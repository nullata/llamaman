# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests as http_requests
from flask import Blueprint, Response, jsonify, request

from config import MODELS_DIR, LLAMAMAN_MAX_MODELS, MODEL_LOAD_TIMEOUT, VERSION, logger
from core.helpers import find_available_port
from api.models import detect_quant, discover_models
from storage import get_storage
from core.state import instances, instances_lock, update_instance_stats
from proxy import get_gate

bp = Blueprint("llamaman", __name__)

# Serialize model launch/evict so one request can't evict a model that
# another request is currently launching or waiting on.
_llamaman_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_name_from_path(path: str) -> str:
    return Path(path).stem.lower()


def _find_model_by_name(name: str) -> dict | None:
    name_lower = name.split(":")[0].lower()
    models = discover_models(MODELS_DIR)
    for m in models:
        if _model_name_from_path(m["path"]) == name_lower:
            return m
    for m in models:
        if name_lower in _model_name_from_path(m["path"]):
            return m
    return None


def _find_running_instance_for_model(model_path: str) -> dict | None:
    with instances_lock:
        for inst in instances.values():
            if inst["model_path"] == model_path and inst["status"] not in ("stopped",):
                return inst
    return None


def _find_any_instance_for_model(model_path: str) -> dict | None:
    with instances_lock:
        for inst in instances.values():
            if inst["model_path"] == model_path:
                return inst
    return None


def _count_running_instances() -> int:
    """Count running non-embedding instances (manual + managed)."""
    with instances_lock:
        return sum(
            1 for inst in instances.values()
            if inst["status"] not in ("stopped",)
            and not inst.get("config", {}).get("embedding_model", False)
        )


def _get_llamaman_managed_instances() -> list[dict]:
    """Return llamaman-managed instances sorted by LRU (eviction candidates)."""
    with instances_lock:
        managed = [
            inst for inst in instances.values()
            if inst.get("_llamaman_managed")
            and inst["status"] not in ("stopped",)
            and not inst.get("config", {}).get("embedding_model", False)
        ]
    managed.sort(key=lambda i: i.get("_last_request_at", i["started_at"]))
    return managed


def _evict_llamaman_instances_if_needed():
    """Evict oldest llamaman-managed instances to stay within limits.

    The limit is checked against ALL running instances (manual + managed),
    but only llamaman-managed instances are evicted.  Manually launched
    instances are never touched.
    """
    from api.instances import stop_instance_by_id

    if LLAMAMAN_MAX_MODELS <= 0:
        return  # 0 = no limit, never evict

    total = _count_running_instances()
    if total < LLAMAMAN_MAX_MODELS:
        return  # still under the limit

    # Need to free at least (total - limit + 1) slots
    managed = _get_llamaman_managed_instances()
    to_free = total - LLAMAMAN_MAX_MODELS + 1
    freed = 0
    while managed and freed < to_free:
        victim = managed.pop(0)
        logger.info(
            "llamaman: evicting %s (port %d) to make room (%d/%d total, max %d)",
            victim["model_name"], victim["port"], total - freed, total, LLAMAMAN_MAX_MODELS,
        )
        stop_instance_by_id(victim["id"])
        freed += 1


def _ensure_model_running(model_name: str) -> tuple[dict | None, str | None]:
    from api.instances import (
        launch_instance, relaunch_inactive_instance, wait_for_healthy,
    )

    model = _find_model_by_name(model_name)
    if model is None:
        return None, f"model '{model_name}' not found"

    # Fast path: model is already healthy - no lock needed
    inst = _find_running_instance_for_model(model["path"])
    if inst and inst["status"] == "healthy":
        with instances_lock:
            if inst["id"] in instances:
                instances[inst["id"]]["_last_request_at"] = time.time()
        return inst, None

    # Slow path: need to launch/relaunch/wait - serialize so concurrent
    # requests for different models don't evict each other's instances.
    with _llamaman_lock:
        # Re-check after acquiring lock (another thread may have launched it)
        inst = _find_running_instance_for_model(model["path"])
        if inst and inst["status"] == "healthy":
            with instances_lock:
                if inst["id"] in instances:
                    instances[inst["id"]]["_last_request_at"] = time.time()
            return inst, None
        if inst and inst["status"] == "starting":
            port = inst.get("_internal_port") or inst["port"]
            if wait_for_healthy(port, timeout=MODEL_LOAD_TIMEOUT):
                with instances_lock:
                    if inst["id"] in instances:
                        instances[inst["id"]]["_last_request_at"] = time.time()
                        instances[inst["id"]]["status"] = "healthy"
                return inst, None
            return None, "model is starting but did not become healthy in time"

        existing = inst or _find_any_instance_for_model(model["path"])
        if existing and existing["status"] in ("sleeping", "stopped"):
            if relaunch_inactive_instance(existing["id"]):
                return existing, None
            return None, "failed to wake model"

        _evict_llamaman_instances_if_needed()

        preset = get_storage().get_preset(model["path"]) or {}

        port = find_available_port()
        if port is None:
            return None, "no ports available"

        inst, err = launch_instance(
            model_path=model["path"],
            port=port,
            n_gpu_layers=preset.get("n_gpu_layers", -1),
            ctx_size=preset.get("ctx_size", 4096),
            threads=preset.get("threads"),
            parallel=preset.get("parallel"),
            extra_args=preset.get("extra_args", ""),
            gpu_devices=preset.get("gpu_devices") or None,
            max_concurrent=preset.get("max_concurrent", 0),
            max_queue_depth=preset.get("max_queue_depth", 200),
            share_queue=preset.get("share_queue", False),
            embedding_model=preset.get("embedding_model", False),
        )
        if err:
            return None, err

        with instances_lock:
            if inst["id"] in instances:
                instances[inst["id"]]["_llamaman_managed"] = True
                instances[inst["id"]]["_last_request_at"] = time.time()

        logger.info("llamaman: auto-launched %s on port %d", model_name, port)

        if not wait_for_healthy(port, timeout=MODEL_LOAD_TIMEOUT):
            return inst, "model launched but did not become healthy in time"

        with instances_lock:
            if inst["id"] in instances:
                instances[inst["id"]]["status"] = "healthy"
                started = instances[inst["id"]].get("started_at", 0)
                if started:
                    stats = instances[inst["id"]].setdefault("stats", {})
                    stats["model_load_time_s"] = round(time.time() - started, 1)

    return inst, None


def _llamaman_model_entry(m: dict) -> dict:
    name = _model_name_from_path(m["path"])
    mtime = datetime.fromtimestamp(
        Path(m["path"]).stat().st_mtime if Path(m["path"]).exists() else 0,
        tz=timezone.utc,
    ).isoformat()
    return {
        "name": name,
        "model": name,
        "modified_at": mtime,
        "size": m["size_bytes"],
        "digest": f"sha256:{uuid.uuid5(uuid.NAMESPACE_URL, m['path']).hex}",
        "details": {
            "parent_model": "",
            "format": m["type"],
            "family": name.split("-")[0] if "-" in name else name,
            "families": [name.split("-")[0]] if "-" in name else [name],
            "parameter_size": "",
            "quantization_level": m.get("quant", ""),
        },
    }


# ---------------------------------------------------------------------------
# Ollama → OpenAI translation
# ---------------------------------------------------------------------------

def _translate_to_openai(body: dict) -> dict:
    openai_body = {
        "model": body.get("model", ""),
        "stream": body.get("stream", True),
    }

    if "messages" in body:
        openai_body["messages"] = body["messages"]

    if "prompt" in body and "messages" not in body:
        msgs = []
        if body.get("system"):
            msgs.append({"role": "system", "content": body["system"]})
        msgs.append({"role": "user", "content": body["prompt"]})
        openai_body["messages"] = msgs

    opts = body.get("options", {})
    if "temperature" in opts:
        openai_body["temperature"] = opts["temperature"]
    if "top_p" in opts:
        openai_body["top_p"] = opts["top_p"]
    if "seed" in opts:
        openai_body["seed"] = opts["seed"]
    if "stop" in opts:
        openai_body["stop"] = opts["stop"]
    if "num_predict" in opts:
        openai_body["max_tokens"] = opts["num_predict"]

    for key in ("temperature", "top_p", "seed", "stop", "max_tokens"):
        if key in body and key not in openai_body:
            openai_body[key] = body[key]

    return openai_body


def _stream_llamaman(port: int, openai_body: dict, model_name: str,
                     mode: str = "chat", inst_id: str | None = None):
    t_start = time.monotonic()
    t_first_token = None
    prompt_tokens = 0
    completion_tokens = 0

    def _content_field(token: str, thinking: str = ""):
        if mode == "chat":
            msg = {"role": "assistant", "content": token}
            if thinking:
                msg["thinking"] = thinking
            return {"message": msg}
        return {"response": (thinking + token) if thinking else token}

    def _done_obj(finish_reason: str = "stop", usage: dict | None = None):
        usage = usage or {}
        elapsed_ns = int((time.monotonic() - t_start) * 1e9)
        p_tokens = usage.get("prompt_tokens", prompt_tokens)
        c_tokens = usage.get("completion_tokens", completion_tokens)
        prompt_dur = int((t_first_token - t_start) * 1e9) if t_first_token else 0
        eval_dur = elapsed_ns - prompt_dur if prompt_dur else elapsed_ns
        return {
            "model": model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **_content_field(""),
            "done": True,
            "done_reason": finish_reason,
            "total_duration": elapsed_ns,
            "load_duration": 0,
            "prompt_eval_count": p_tokens,
            "prompt_eval_duration": prompt_dur,
            "eval_count": c_tokens,
            "eval_duration": eval_dur,
        }

    resp = None
    try:
        resp = http_requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json=openai_body,
            stream=True,
            timeout=300,
        )
        if resp.status_code >= 400:
            error_text = resp.text[:500] if resp.text else f"HTTP {resp.status_code}"
            error_obj = {
                "model": model_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                **_content_field(f"Error: {error_text}"),
                "done": True,
                "done_reason": "stop",
            }
            yield json.dumps(error_obj, ensure_ascii=False) + "\n"
            return
        resp.encoding = "utf-8"

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    yield json.dumps(_done_obj(), ensure_ascii=False) + "\n"
                    return

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                token = delta.get("content", "")
                thinking = delta.get("reasoning_content", "")
                finish = choices[0].get("finish_reason")

                if token or thinking:
                    if t_first_token is None:
                        t_first_token = time.monotonic()
                    completion_tokens += 1

                chunk_obj = {
                    "model": model_name,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **_content_field(token, thinking),
                    "done": False,
                }
                yield json.dumps(chunk_obj, ensure_ascii=False) + "\n"

                if finish:
                    yield json.dumps(_done_obj(finish, chunk.get("usage", {})), ensure_ascii=False) + "\n"
                    return

    except Exception as e:
        error_obj = {
            "model": model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **_content_field(f"Error: {e}"),
            "done": True,
            "done_reason": "stop",
        }
        yield json.dumps(error_obj, ensure_ascii=False) + "\n"
    finally:
        if resp is not None:
            resp.close()
        if inst_id and completion_tokens > 0:
            elapsed = time.monotonic() - t_start
            tps = completion_tokens / elapsed if elapsed > 0 else None
            ttft = ((t_first_token - t_start) * 1000) if t_first_token else None
            update_instance_stats(inst_id, tokens_per_sec=tps, ttft_ms=ttft)


def _proxy_non_streaming(port: int, openai_body: dict, model_name: str,
                         mode: str = "chat", inst_id: str | None = None):
    t_start = time.monotonic()
    openai_body["stream"] = False
    resp = http_requests.post(
        f"http://localhost:{port}/v1/chat/completions",
        json=openai_body,
        timeout=300,
    )
    resp.raise_for_status()
    elapsed_ns = int((time.monotonic() - t_start) * 1e9)
    data = resp.json()
    choices = data.get("choices", [])
    usage = data.get("usage", {})

    result = {
        "model": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "done": True,
        "done_reason": choices[0].get("finish_reason", "stop") if choices else "stop",
        "total_duration": elapsed_ns,
        "load_duration": 0,
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "prompt_eval_duration": 0,
        "eval_count": usage.get("completion_tokens", 0),
        "eval_duration": elapsed_ns,
    }

    if mode == "chat":
        msg = choices[0]["message"] if choices else {"role": "assistant", "content": ""}
        reasoning = msg.pop("reasoning_content", "")
        if reasoning:
            msg["thinking"] = reasoning
        result["message"] = msg
    else:
        result["response"] = choices[0]["message"]["content"] if choices else ""

    if inst_id:
        elapsed = (time.monotonic() - t_start)
        c_tokens = usage.get("completion_tokens", 0)
        tps = c_tokens / elapsed if elapsed > 0 and c_tokens else None
        update_instance_stats(inst_id, tokens_per_sec=tps)

    return result


def _handle_request(mode: str = "chat"):
    body = request.get_json(force=True)
    model_name = body.get("model", "").strip()
    if not model_name:
        return jsonify({"error": "model is required"}), 400

    if mode == "generate" and body.get("keep_alive") == 0 and not body.get("prompt"):
        return jsonify({
            "model": model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "response": "",
            "done": True,
        })

    inst, err = _ensure_model_running(model_name)
    if err:
        return jsonify({"error": err}), 500

    server_port = inst.get("_internal_port") or inst["port"]

    gate = get_gate(inst["id"])
    if gate:
        if not gate.acquire(timeout=300):
            return jsonify({"error": "request queue full"}), 429

    openai_body = _translate_to_openai(body)
    stream_qp = request.args.get("stream", "").lower()
    if stream_qp in ("false", "0", "no"):
        stream = False
    else:
        stream = body.get("stream", True)

    stream_returned = False
    try:
        if stream:
            def _gated_stream():
                try:
                    yield from _stream_llamaman(server_port, openai_body, model_name, mode, inst_id=inst["id"])
                finally:
                    if gate:
                        gate.release()
            stream_returned = True
            return Response(
                _gated_stream(),
                mimetype="application/x-ndjson",
            )

        result = _proxy_non_streaming(server_port, openai_body, model_name, mode, inst_id=inst["id"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if gate and not stream_returned:
            gate.release()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@bp.route("/api/tags", methods=["GET"])
def llamaman_tags():
    models = discover_models(MODELS_DIR)
    return jsonify({"models": [_llamaman_model_entry(m) for m in models]})


@bp.route("/api/version", methods=["GET"])
def llamaman_version():
    return jsonify({"version": VERSION})


@bp.route("/api/show", methods=["POST"])
def llamaman_show():
    body = request.get_json(force=True)
    model_name = body.get("model", body.get("name", "")).strip()
    model = _find_model_by_name(model_name)
    if model is None:
        return jsonify({"error": f"model '{model_name}' not found"}), 404

    entry = _llamaman_model_entry(model)
    return jsonify({
        "modelfile": f"FROM {model['path']}",
        "parameters": "",
        "template": "",
        "details": entry["details"],
        "model_info": {
            "general.architecture": entry["details"]["family"],
            "general.file_type": 0,
            "general.parameter_count": 0,
        },
    })


@bp.route("/api/ps", methods=["GET"])
def llamaman_ps():
    running = []
    with instances_lock:
        for inst in instances.values():
            if inst["status"] in ("stopped",):
                continue
            running.append({
                "name": _model_name_from_path(inst["model_path"]),
                "model": _model_name_from_path(inst["model_path"]),
                "size": 0,
                "digest": "",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "",
                    "families": [],
                    "parameter_size": "",
                    "quantization_level": detect_quant(Path(inst["model_path"]).stem),
                },
                "expires_at": datetime.fromtimestamp(
                    inst["started_at"] + 300, tz=timezone.utc
                ).isoformat(),
                "size_vram": 0,
            })
    return jsonify({"models": running})


@bp.route("/api/chat", methods=["POST"])
def llamaman_chat():
    return _handle_request(mode="chat")


@bp.route("/api/generate", methods=["POST"])
def llamaman_generate():
    return _handle_request(mode="generate")


@bp.route("/v1/models", methods=["GET"])
def llamaman_v1_models():
    models = discover_models(MODELS_DIR)
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": _model_name_from_path(m["path"]),
                "object": "model",
                "created": int(Path(m["path"]).stat().st_mtime) if Path(m["path"]).exists() else 0,
                "owned_by": "local",
            }
            for m in models
        ],
    })


@bp.route("/v1/chat/completions", methods=["POST"])
def llamaman_v1_chat():
    body = request.get_json(force=True)
    model_name = body.get("model", "").strip()
    if not model_name:
        return jsonify({"error": {"message": "model is required"}}), 400

    inst, err = _ensure_model_running(model_name)
    if err:
        return jsonify({"error": {"message": err}}), 500

    server_port = inst.get("_internal_port") or inst["port"]
    inst_id = inst["id"]

    gate = get_gate(inst_id)
    if gate:
        if not gate.acquire(timeout=300):
            return jsonify({"error": {"message": "request queue full"}}), 429

    stream = body.get("stream", False)
    stream_returned = False
    t_start = time.monotonic()
    try:
        resp = http_requests.post(
            f"http://localhost:{server_port}/v1/chat/completions",
            json=body,
            stream=stream,
            timeout=300,
        )
        if stream:
            def _relay():
                try:
                    yield from resp.iter_content(chunk_size=None)
                finally:
                    resp.close()
                    if gate:
                        gate.release()
                    # Can't extract token counts from a consumed stream,
                    # but still update timestamp and request count.
                    _touch_instance(inst_id)
                    update_instance_stats(inst_id)
            stream_returned = True
            return Response(
                _relay(),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        with resp:
            data = resp.json()
            _touch_instance(inst_id)
            # Extract stats from the non-streaming response
            usage = data.get("usage", {})
            c_tokens = usage.get("completion_tokens", 0)
            elapsed = time.monotonic() - t_start
            tps = c_tokens / elapsed if c_tokens and elapsed > 0 else None
            update_instance_stats(inst_id, tokens_per_sec=tps)
            return jsonify(data), resp.status_code
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 500
    finally:
        if gate and not stream_returned:
            gate.release()


def _touch_instance(inst_id: str):
    """Update the last-request timestamp so idle timeout doesn't kill active models."""
    with instances_lock:
        inst = instances.get(inst_id)
        if inst:
            inst["_last_request_at"] = time.time()
