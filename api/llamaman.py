# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests as http_requests
from flask import Blueprint, Response, jsonify, request

from config import (
    HEALTH_CHECK_TIMEOUT,
    MODELS_DIR,
    LLAMAMAN_MAX_MODELS,
    MODEL_LOAD_TIMEOUT,
    REQUEST_TIMEOUT,
    VERSION,
    logger,
)
from core.helpers import (
    find_available_port,
    is_llama_pid,
    is_pid_alive,
    scan_llama_server_processes,
)
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


def _get_all_evictable_instances() -> list[dict]:
    """Return ALL non-embedding running instances sorted by LRU."""
    with instances_lock:
        all_insts = [
            inst for inst in instances.values()
            if inst["status"] not in ("stopped",)
            and not inst.get("config", {}).get("embedding_model", False)
        ]
    all_insts.sort(key=lambda i: i.get("_last_request_at", i["started_at"]))
    return all_insts


def _ollama_can_evict_admin_instances() -> bool:
    return bool(get_storage().get_settings().get("allow_ollama_api_override_admin", False))


def _evict_llamaman_instances_if_needed(incoming_embedding_model: bool = False) -> bool:
    """Evict oldest llamaman-managed instances to stay within limits.

    Returns True if there is room for a new instance after eviction (or if no
    limit is set). Returns False if the cap is still exceeded because admin-UI
    instances are blocking and the override setting is disabled.

    The limit is checked against ALL running instances (manual + managed),
    but only llamaman-managed instances are evicted by default.  Manually
    launched instances are never touched unless allow_ollama_api_override_admin
    is enabled.
    """
    from api.instances import stop_instance_by_id

    if LLAMAMAN_MAX_MODELS <= 0:
        return True  # 0 = no limit, never evict
    if incoming_embedding_model:
        return True  # embedding launches never count toward the chat-model cap

    total = _count_running_instances()
    if total < LLAMAMAN_MAX_MODELS:
        return True  # still under the limit

    # First pass: evict only Ollama-managed instances (LRU order).
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

    # Check if first pass freed enough slots.
    if _count_running_instances() < LLAMAMAN_MAX_MODELS:
        return True

    # Still over limit - only proceed if the override setting allows evicting
    # admin-UI launched instances as well.
    if not _ollama_can_evict_admin_instances():
        return False

    # Second pass: also evict admin-UI instances (LRU order).
    remaining = _get_all_evictable_instances()
    while remaining and _count_running_instances() >= LLAMAMAN_MAX_MODELS:
        victim = remaining.pop(0)
        logger.info(
            "llamaman: evicting admin-launched %s (port %d) - override enabled",
            victim["model_name"], victim["port"],
        )
        stop_instance_by_id(victim["id"])

    return _count_running_instances() < LLAMAMAN_MAX_MODELS


def _wait_for_model_ready(port: int, timeout: float) -> bool:
    """Poll the llama-server /health endpoint until it reports ready.

    Unlike wait_for_healthy (which is used during launch bookkeeping), this
    is designed for the request-forwarding path: it retries with a short
    interval so the first inference request is sent as soon as the model is
    actually ready.
    """
    from api.instances import wait_for_healthy
    return wait_for_healthy(port, timeout=timeout)


def _ensure_model_running(
    model_name: str,
    allow_eviction: bool = True,
) -> tuple[dict | None, str | None]:
    """Ensure a model instance exists and is at least launched.

    Returns the instance as soon as it is launched (status may still be
    ``"starting"``).  Callers are expected to use ``_wait_for_model_ready``
    on the server port before forwarding the actual request.

    allow_eviction controls whether LRU eviction may be used to free a slot.
    The Ollama API sets this True; the OpenAI API sets it False so it never
    displaces a running model — it either finds a free slot or returns 503.
    """
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
        if inst and inst["status"] in ("healthy", "starting"):
            with instances_lock:
                if inst["id"] in instances:
                    instances[inst["id"]]["_last_request_at"] = time.time()
            return inst, None

        existing = inst or _find_any_instance_for_model(model["path"])
        preset = get_storage().get_preset(model["path"]) or {}
        incoming_embedding_model = preset.get("embedding_model", False)

        if allow_eviction:
            # Evict LRU Ollama-managed instances (and admin-UI ones if the
            # override toggle is on) to stay within LLAMAMAN_MAX_MODELS.
            room = _evict_llamaman_instances_if_needed(
                incoming_embedding_model=incoming_embedding_model,
            )
            if not room:
                return None, (
                    f"model limit reached (LLAMAMAN_MAX_MODELS={LLAMAMAN_MAX_MODELS}); "
                    "admin-launched models cannot be evicted via the API"
                )
        else:
            # OpenAI API: never evict — only proceed if there is already room.
            if not incoming_embedding_model and LLAMAMAN_MAX_MODELS > 0:
                if _count_running_instances() >= LLAMAMAN_MAX_MODELS:
                    return None, (
                        f"model limit reached (LLAMAMAN_MAX_MODELS={LLAMAMAN_MAX_MODELS}); "
                        "the OpenAI API does not evict running models"
                    )

        if existing and existing["status"] in ("sleeping", "stopped"):
            # relaunch_inactive_instance blocks until healthy; if it
            # succeeds the instance is ready for requests immediately.
            if relaunch_inactive_instance(existing["id"]):
                return existing, None
            return None, "failed to wake model"

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
            idle_timeout_min=preset.get("idle_timeout_min", 0),
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

    # Return immediately - the model is launched but may still be loading.
    # The caller will poll for readiness before forwarding the request.
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


def _instance_process_alive(inst: dict) -> bool:
    proc = inst.get("_process")
    if proc is not None:
        return proc.poll() is None

    pid = inst.get("pid", 0)
    return pid > 0 and is_pid_alive(pid) and is_llama_pid(pid)


def _probe_server_ready(port: int) -> bool:
    try:
        resp = http_requests.get(
            f"http://localhost:{port}/health",
            timeout=HEALTH_CHECK_TIMEOUT,
        )
        return resp.json().get("status") == "ok"
    except Exception:
        return False


def _llamaman_ps_entry(model_path: str, model_meta: dict | None = None,
                       started_at: float | None = None) -> dict:
    model_meta = model_meta or {}
    model_name = _model_name_from_path(model_path)
    size_bytes = model_meta.get("size_bytes")
    if size_bytes is None:
        try:
            size_bytes = os.path.getsize(model_path)
        except OSError:
            size_bytes = 0

    return {
        "name": model_name,
        "model": model_name,
        "size": size_bytes,
        "digest": f"sha256:{uuid.uuid5(uuid.NAMESPACE_URL, model_path).hex}",
        "details": {
            "parent_model": "",
            "format": model_meta.get("type", "gguf"),
            "family": model_name.split("-")[0] if "-" in model_name else model_name,
            "families": [model_name.split("-")[0]] if "-" in model_name else [model_name],
            "parameter_size": "",
            "quantization_level": model_meta.get("quant", detect_quant(Path(model_path).stem)),
        },
        "expires_at": datetime.fromtimestamp(
            (started_at or time.time()) + 300,
            tz=timezone.utc,
        ).isoformat(),
        "size_vram": 0,
    }


def _list_loaded_models() -> list[dict]:
    model_index = {
        os.path.realpath(m["path"]): m
        for m in discover_models(MODELS_DIR)
    }
    live_by_path: dict[str, dict] = {}

    with instances_lock:
        tracked_instances = [dict(inst) for inst in instances.values()]

    for inst in tracked_instances:
        if not _instance_process_alive(inst):
            continue

        model_path = inst["model_path"]
        server_port = inst.get("_internal_port") or inst["port"]
        ready = _probe_server_ready(server_port)
        key = os.path.realpath(model_path)
        existing = live_by_path.get(key)

        if existing and existing.get("ready") and not ready:
            continue

        live_by_path[key] = {
            "model_path": model_path,
            "started_at": inst.get("started_at"),
            "ready": ready,
        }

    for info in scan_llama_server_processes():
        model_path = info["model_path"]
        key = os.path.realpath(model_path)
        if key in live_by_path and live_by_path[key].get("ready"):
            continue

        live_by_path[key] = {
            "model_path": model_path,
            "started_at": None,
            "ready": _probe_server_ready(info["port"]),
        }

    loaded = [
        _llamaman_ps_entry(
            entry["model_path"],
            model_meta=model_index.get(os.path.realpath(entry["model_path"])),
            started_at=entry.get("started_at"),
        )
        for entry in live_by_path.values()
    ]
    loaded.sort(key=lambda item: item["name"])
    return loaded


# ---------------------------------------------------------------------------
# Ollama >> OpenAI translation
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
        # Retry on connection errors (model may have just finished loading)
        last_err = None
        for _attempt in range(3):
            try:
                resp = http_requests.post(
                    f"http://localhost:{port}/v1/chat/completions",
                    json=openai_body,
                    stream=True,
                    timeout=REQUEST_TIMEOUT,
                )
                break
            except (http_requests.ConnectionError, http_requests.Timeout,
                    ConnectionRefusedError) as e:
                last_err = e
                time.sleep(2)
        if resp is None:
            raise last_err or ConnectionError("failed to connect to model server")
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
    # Retry on connection errors (model may have just finished loading)
    last_err = None
    resp = None
    for _attempt in range(3):
        try:
            resp = http_requests.post(
                f"http://localhost:{port}/v1/chat/completions",
                json=openai_body,
                timeout=REQUEST_TIMEOUT,
            )
            break
        except (http_requests.ConnectionError, http_requests.Timeout,
                ConnectionRefusedError) as e:
            last_err = e
            time.sleep(2)
    if resp is None:
        raise last_err or ConnectionError("failed to connect to model server")
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
        code = 503 if "model limit reached" in err else 500
        return jsonify({"error": err}), code

    server_port = inst.get("_internal_port") or inst["port"]

    # If the model was just launched it may still be loading.  Wait for it
    # to become healthy before forwarding the request so the prompt is not
    # lost to a connection-refused error.
    if inst.get("status") != "healthy":
        if not _wait_for_model_ready(server_port, MODEL_LOAD_TIMEOUT):
            return jsonify({"error": "model launched but did not become healthy in time"}), 500
        with instances_lock:
            if inst["id"] in instances:
                instances[inst["id"]]["status"] = "healthy"
                started = instances[inst["id"]].get("started_at", 0)
                if started:
                    stats = instances[inst["id"]].setdefault("stats", {})
                    stats["model_load_time_s"] = round(time.time() - started, 1)

    gate = get_gate(inst["id"])
    if gate:
        if not gate.acquire(timeout=REQUEST_TIMEOUT):
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
    return jsonify({"models": _list_loaded_models()})


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

    inst, err = _ensure_model_running(model_name, allow_eviction=False)
    if err:
        return jsonify({"error": {"message": err}}), 503

    server_port = inst.get("_internal_port") or inst["port"]
    inst_id = inst["id"]

    # Wait for the model to finish loading before forwarding
    if inst.get("status") != "healthy":
        if not _wait_for_model_ready(server_port, MODEL_LOAD_TIMEOUT):
            return jsonify({"error": {"message": "model launched but did not become healthy in time"}}), 500
        with instances_lock:
            if inst_id in instances:
                instances[inst_id]["status"] = "healthy"
                started = instances[inst_id].get("started_at", 0)
                if started:
                    stats = instances[inst_id].setdefault("stats", {})
                    stats["model_load_time_s"] = round(time.time() - started, 1)

    gate = get_gate(inst_id)
    if gate:
        if not gate.acquire(timeout=REQUEST_TIMEOUT):
            return jsonify({"error": {"message": "request queue full"}}), 429

    stream = body.get("stream", False)
    stream_returned = False
    t_start = time.monotonic()
    try:
        # Retry on connection errors (transient failures right after load)
        last_err = None
        resp = None
        for _attempt in range(3):
            try:
                resp = http_requests.post(
                    f"http://localhost:{server_port}/v1/chat/completions",
                    json=body,
                    stream=stream,
                    timeout=REQUEST_TIMEOUT,
                )
                break
            except (http_requests.ConnectionError, http_requests.Timeout,
                    ConnectionRefusedError) as e:
                last_err = e
                time.sleep(2)
        if resp is None:
            raise last_err or ConnectionError("failed to connect to model server")
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
