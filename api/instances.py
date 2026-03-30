# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import os
import subprocess
import time
import uuid
from pathlib import Path

import requests as http_requests
from flask import Blueprint, Response, jsonify, request

from config import (
    HEALTH_CHECK_TIMEOUT,
    INTERNAL_PORT_RANGE_END,
    INTERNAL_PORT_RANGE_START,
    LLAMAMAN_MAX_MODELS,
    LOGS_DIR,
    MODEL_LOAD_TIMEOUT,
    PORT_RANGE_END,
    PORT_RANGE_START,
    logger,
)
from core.helpers import (
    build_llama_cmd, find_available_port, is_port_available, kill_instance_process,
    public_dict, read_log_file, stream_log_file,
)
from core.state import (
    instances, instances_lock, save_state,
)
from proxy import (
    create_gate, get_gate, remove_gate, refresh_gate,
    start_idle_proxy, stop_idle_proxy,
)
from storage import get_storage

bp = Blueprint("instances", __name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _public_instance(inst: dict) -> dict:
    d = public_dict(inst)
    d["last_request_at"] = inst.get("_last_request_at")
    if inst.get("_internal_port") is not None:
        d["internal_port"] = inst.get("_internal_port")
    gate = get_gate(inst["id"])
    if gate:
        d["queue"] = {
            "active": gate.active,
            "queued": gate.queued,
            "max_concurrent": gate.max_concurrent,
            "max_queue_depth": gate.max_queue_depth,
        }
    return d


def _merge_preset_into_config(model_path: str, config: dict) -> dict:
    """Overlay the latest saved preset onto an instance config."""
    from storage import get_storage

    merged = dict(config)
    preset = get_storage().get_preset(model_path) or {}
    if preset:
        for key in (
            "n_gpu_layers",
            "ctx_size",
            "threads",
            "parallel",
            "extra_args",
            "gpu_devices",
            "idle_timeout_min",
            "max_concurrent",
            "max_queue_depth",
            "share_queue",
            "embedding_model",
        ):
            if key in preset:
                merged[key] = preset[key]
    return merged


def _parse_required_positive_int(body: dict, field_name: str) -> tuple[int | None, str | None]:
    raw = body.get(field_name)
    if raw in (None, ""):
        return None, f"{field_name} is required"
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None, f"{field_name} must be an integer"
    if value <= 0:
        return None, f"{field_name} must be greater than 0"
    return value, None


def _admin_ui_enforces_eviction() -> bool:
    settings = get_storage().get_settings()
    return bool(settings.get("admin_ui_enforce_max_models", False))


def _count_running_chat_instances(exclude_instance_id: str | None = None) -> int:
    with instances_lock:
        return sum(
            1 for inst in instances.values()
            if inst["id"] != exclude_instance_id
            and inst["status"] not in ("stopped",)
            and not inst.get("config", {}).get("embedding_model", False)
        )


def _would_ui_launch_exceed_limit(
    incoming_embedding_model: bool = False,
    exclude_instance_id: str | None = None,
) -> bool:
    if LLAMAMAN_MAX_MODELS <= 0 or incoming_embedding_model:
        return False
    return _count_running_chat_instances(exclude_instance_id=exclude_instance_id) >= LLAMAMAN_MAX_MODELS


def _get_lru_chat_instances(
    exclude_instance_id: str | None = None,
    ollama_managed_first: bool = False,
) -> list[dict]:
    with instances_lock:
        candidates = [
            inst for inst in instances.values()
            if inst["id"] != exclude_instance_id
            and inst["status"] not in ("stopped",)
            and not inst.get("config", {}).get("embedding_model", False)
        ]
    if ollama_managed_first:
        # Ollama-managed (True) sorts before admin-UI (False/absent) so that
        # admin UI launches always evict API-managed models first.
        candidates.sort(key=lambda inst: (
            not inst.get("_llamaman_managed", False),
            inst.get("_last_request_at", inst.get("started_at", 0)),
        ))
    else:
        candidates.sort(key=lambda inst: inst.get("_last_request_at", inst.get("started_at", 0)))
    return candidates


def _evict_instances_for_ui_launch_if_needed(
    incoming_embedding_model: bool = False,
    exclude_instance_id: str | None = None,
) -> None:
    if LLAMAMAN_MAX_MODELS <= 0 or incoming_embedding_model:
        return

    total = _count_running_chat_instances(exclude_instance_id=exclude_instance_id)
    if total < LLAMAMAN_MAX_MODELS:
        return

    to_free = total - LLAMAMAN_MAX_MODELS + 1
    freed = 0
    for victim in _get_lru_chat_instances(
        exclude_instance_id=exclude_instance_id,
        ollama_managed_first=True,
    ):
        if freed >= to_free:
            break
        logger.info(
            "ui: evicting %s (port %d) to make room (%d/%d total, max %d)",
            victim["model_name"], victim["port"], total - freed, total, LLAMAMAN_MAX_MODELS,
        )
        stop_instance_by_id(victim["id"])
        freed += 1


def wait_for_healthy(port: int, timeout: float = 120) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = http_requests.get(f"http://localhost:{port}/health", timeout=HEALTH_CHECK_TIMEOUT)
            if resp.json().get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def relaunch_inactive_instance(inst_id: str) -> bool:
    with instances_lock:
        inst = instances.get(inst_id)
        if inst is None or inst["status"] not in ("sleeping", "stopped"):
            return inst is not None and inst["status"] in ("healthy", "starting")

        prior_status = inst["status"]
        config = _merge_preset_into_config(inst["model_path"], inst["config"])
        model_path = inst["model_path"]
        internal_port = inst.get("_internal_port", inst["port"])

    # Persist the updated config back to the instance record so the UI and
    # future restarts reflect the latest saved preset.
    with instances_lock:
        inst = instances.get(inst_id)
        if inst:
            inst["config"] = config

    gpu_devices = config.get("gpu_devices")

    refresh_gate(inst_id)

    log_file = os.path.join(LOGS_DIR, f"{inst_id}.log")
    cmd = build_llama_cmd(model_path, internal_port, config)

    if not is_port_available(internal_port):
        logger.warning(
            "Cannot relaunch inactive instance %s: port %d is already occupied",
            inst_id, internal_port,
        )
        with instances_lock:
            inst = instances.get(inst_id)
            if inst:
                inst["status"] = "stopped"
        save_state()
        return False

    env = {**os.environ}
    if gpu_devices:
        env["CUDA_VISIBLE_DEVICES"] = gpu_devices

    logger.info(
        "Relaunching %s instance %s on server port %d",
        prior_status, inst_id, internal_port,
    )

    try:
        log_fh = open(log_file, "a")
        log_fh.write(f"\n--- Relaunched at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT, close_fds=True,
        )
    except Exception as e:
        log_fh.close()
        logger.error("Failed to relaunch %s: %s", inst_id, e)
        return False

    with instances_lock:
        inst = instances.get(inst_id)
        if inst:
            inst["status"] = "starting"
            inst["pid"] = proc.pid
            inst["_process"] = proc
            inst["_log_fh"] = log_fh
            inst["started_at"] = time.time()
            inst["_last_request_at"] = time.time()

    save_state()

    if not wait_for_healthy(internal_port, timeout=MODEL_LOAD_TIMEOUT):
        logger.warning("Relaunched %s but it did not become healthy", inst_id)
        return False

    with instances_lock:
        inst = instances.get(inst_id)
        if inst:
            inst["status"] = "healthy"
    save_state()
    return True


def relaunch_sleeping_instance(inst_id: str) -> bool:
    return relaunch_inactive_instance(inst_id)


# ---------------------------------------------------------------------------
# Launch / Stop / Sleep
# ---------------------------------------------------------------------------

def launch_instance(model_path, port, n_gpu_layers=-1, ctx_size=4096,
                    threads=None, parallel=None, extra_args="",
                    gpu_devices=None, idle_timeout_min=0,
                    max_concurrent=0, max_queue_depth=200,
                    share_queue=False, embedding_model=False):
    with instances_lock:
        used_ports = {i["port"] for i in instances.values() if i["status"] not in ("stopped",)}
    if port in used_ports:
        return None, f"Port {port} is already in use"

    needs_proxy = idle_timeout_min > 0 or max_concurrent > 0
    if needs_proxy:
        internal_port = find_available_port(
            exclude={port},
            range_start=INTERNAL_PORT_RANGE_START,
            range_end=INTERNAL_PORT_RANGE_END,
        )
        if internal_port is None:
            return None, "no internal ports available for proxy"
        server_port = internal_port
    else:
        server_port = port
        internal_port = None

    config = {
        "n_gpu_layers": n_gpu_layers,
        "ctx_size": ctx_size,
        "threads": threads,
        "parallel": parallel,
        "extra_args": extra_args,
        "gpu_devices": gpu_devices,
        "idle_timeout_min": idle_timeout_min,
        "max_concurrent": max_concurrent,
        "max_queue_depth": max_queue_depth,
        "share_queue": share_queue,
        "embedding_model": embedding_model,
    }

    inst_id = str(uuid.uuid4())
    log_file = os.path.join(LOGS_DIR, f"{inst_id}.log")
    model_name = Path(model_path).name
    cmd = build_llama_cmd(model_path, server_port, config)

    if not is_port_available(port):
        return None, f"Port {port} is already occupied by another process"
    if internal_port and not is_port_available(internal_port):
        return None, f"Internal port {internal_port} is already occupied by another process"

    env = {**os.environ}
    if gpu_devices:
        env["CUDA_VISIBLE_DEVICES"] = gpu_devices

    logger.info("Launching: %s (CUDA_VISIBLE_DEVICES=%s)", " ".join(cmd), gpu_devices or "all")

    try:
        log_fh = open(log_file, "w")
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT, close_fds=True,
        )
    except FileNotFoundError:
        log_fh.close()
        return None, "llama-server binary not found. Is llama.cpp installed?"
    except Exception as e:
        log_fh.close()
        return None, str(e)

    instance = {
        "id": inst_id,
        "model_name": model_name,
        "model_path": model_path,
        "port": port,
        "status": "starting",
        "pid": proc.pid,
        "log_file": log_file,
        "config": config,
        "started_at": time.time(),
        "_process": proc,
        "_log_fh": log_fh,
        "_last_request_at": time.time(),
        "stats": {
            "model_load_time_s": None,
            "last_tokens_per_sec": None,
            "last_ttft_ms": None,
            "total_requests": 0,
            "crash_count": 0,
        },
    }

    if internal_port:
        instance["_internal_port"] = internal_port

    with instances_lock:
        instances[inst_id] = instance

    if needs_proxy and internal_port:
        start_idle_proxy(inst_id, port, internal_port)

    if max_concurrent > 0:
        create_gate(inst_id, max_concurrent, max_queue_depth,
                    model_path=model_path, share_queue=share_queue)

    save_state()
    return instance, None


def stop_instance_by_id(inst_id: str) -> bool:
    with instances_lock:
        inst = instances.get(inst_id)
        if inst is None:
            return False
        kill_instance_process(inst)
        inst["status"] = "stopped"
    release_instance_reservations(inst_id)
    save_state()
    return True


def sleep_instance_by_id(inst_id: str) -> bool:
    with instances_lock:
        inst = instances.get(inst_id)
        if inst is None:
            return False
        kill_instance_process(inst)
        inst["status"] = "sleeping"
    refresh_gate(inst_id)
    save_state()
    logger.info("Instance %s put to sleep (idle timeout)", inst_id)
    return True


def _restore_restarted_instance(old: dict) -> None:
    """Restore a removed instance record if restart launch fails."""
    with instances_lock:
        instances[old["id"]] = old

    if old["status"] == "sleeping":
        internal_port = old.get("_internal_port")
        if internal_port:
            start_idle_proxy(old["id"], old["port"], internal_port)
        max_concurrent = old.get("config", {}).get("max_concurrent", 0)
        if max_concurrent > 0:
            create_gate(
                old["id"],
                max_concurrent,
                old.get("config", {}).get("max_queue_depth", 200),
                model_path=old["model_path"],
                share_queue=old.get("config", {}).get("share_queue", False),
            )


def release_instance_reservations(inst_id: str) -> None:
    """Release proxy/gate resources tied to a public instance port."""
    stop_idle_proxy(inst_id)
    remove_gate(inst_id)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@bp.route("/api/next-port")
def api_next_port():
    port = find_available_port()
    if port is None:
        return jsonify({"error": "No ports available", "port": None}), 409
    return jsonify({"port": port})


@bp.route("/api/instances", methods=["GET"])
def api_instances_list():
    with instances_lock:
        safe = [_public_instance(inst) for inst in instances.values()]
    return jsonify(safe)


@bp.route("/api/instances", methods=["POST"])
def api_instances_create():
    body = request.get_json(force=True)
    model_path = body.get("model_path", "").strip()
    if not model_path:
        return jsonify({"error": "model_path is required"}), 400

    ctx_size, ctx_err = _parse_required_positive_int(body, "ctx_size")
    if ctx_err:
        return jsonify({"error": ctx_err}), 400

    incoming_embedding_model = bool(body.get("embedding_model", False))
    confirm_overcommit = bool(body.get("confirm_overcommit", False))
    if _admin_ui_enforces_eviction():
        _evict_instances_for_ui_launch_if_needed(
            incoming_embedding_model=incoming_embedding_model,
        )
    elif _would_ui_launch_exceed_limit(incoming_embedding_model=incoming_embedding_model) and not confirm_overcommit:
        return jsonify({
            "error": f"You're about to launch an instance beyond LLAMAMAN_MAX_MODELS={LLAMAMAN_MAX_MODELS}. Do you want to proceed?",
            "confirm_required": True,
        }), 409

    inst, err = launch_instance(
        model_path=model_path,
        port=int(body.get("port", 8000)),
        n_gpu_layers=int(body.get("n_gpu_layers", -1)),
        ctx_size=ctx_size,
        threads=body.get("threads"),
        parallel=body.get("parallel"),
        extra_args=body.get("extra_args", "").strip(),
        gpu_devices=body.get("gpu_devices", "").strip() or None,
        idle_timeout_min=int(body.get("idle_timeout_min", 0)),
        max_concurrent=int(body.get("max_concurrent", 0)),
        max_queue_depth=int(body.get("max_queue_depth", 200)),
        share_queue=bool(body.get("share_queue", False)),
        embedding_model=bool(body.get("embedding_model", False)),
    )
    if err:
        code = 409 if "already in use" in err else 500
        return jsonify({"error": err}), code
    return jsonify(_public_instance(inst)), 201


@bp.route("/api/instances/<inst_id>", methods=["DELETE"])
def api_instances_delete(inst_id):
    if not stop_instance_by_id(inst_id):
        return jsonify({"error": "Not found"}), 404
    return jsonify({"status": "stopped"})


@bp.route("/api/instances/<inst_id>/restart", methods=["POST"])
def api_instances_restart(inst_id):
    body = request.get_json(silent=True) or {}
    with instances_lock:
        old = instances.get(inst_id)
        if old is None:
            return jsonify({"error": "Not found"}), 404
        if old["status"] not in ("stopped", "sleeping"):
            return jsonify({"error": "Instance must be stopped or sleeping before restarting"}), 409
        old = {
            **old,
            "config": _merge_preset_into_config(old["model_path"], old.get("config", {})),
            "stats": dict(old.get("stats", {})),
        }
        model_path = old["model_path"]
        config = old["config"]
        preferred_port = old["port"]

    incoming_embedding_model = bool(config.get("embedding_model", False))
    confirm_overcommit = bool(body.get("confirm_overcommit", False))
    if _admin_ui_enforces_eviction():
        _evict_instances_for_ui_launch_if_needed(
            incoming_embedding_model=incoming_embedding_model,
            exclude_instance_id=inst_id,
        )
    elif _would_ui_launch_exceed_limit(
        incoming_embedding_model=incoming_embedding_model,
        exclude_instance_id=inst_id,
    ) and not confirm_overcommit:
        return jsonify({
            "error": f"You're about to launch an instance beyond LLAMAMAN_MAX_MODELS={LLAMAMAN_MAX_MODELS}. Do you want to proceed?",
            "confirm_required": True,
        }), 409

    # Release any old proxy/gate reservations before we pick ports for the
    # replacement instance. This lets restarts reuse the same public port and
    # avoids stale reservations influencing internal-port selection.
    release_instance_reservations(inst_id)
    with instances_lock:
        instances.pop(inst_id, None)

    port = preferred_port if is_port_available(preferred_port) else None
    if port is None:
        for p in range(PORT_RANGE_START, PORT_RANGE_END + 1):
            if is_port_available(p):
                port = p
                break
    if port is None:
        _restore_restarted_instance(old)
        save_state()
        return jsonify({"error": "No ports available"}), 409

    inst, err = launch_instance(
        model_path=model_path,
        port=port,
        n_gpu_layers=config.get("n_gpu_layers", -1),
        ctx_size=config.get("ctx_size", 4096),
        threads=config.get("threads"),
        parallel=config.get("parallel"),
        extra_args=config.get("extra_args", ""),
        gpu_devices=config.get("gpu_devices"),
        idle_timeout_min=config.get("idle_timeout_min", 0),
        max_concurrent=config.get("max_concurrent", 0),
        max_queue_depth=config.get("max_queue_depth", 200),
        share_queue=config.get("share_queue", False),
        embedding_model=config.get("embedding_model", False),
    )
    if err:
        _restore_restarted_instance(old)
        save_state()
        code = 409 if "already in use" in err else 500
        return jsonify({"error": err}), code

    return jsonify(_public_instance(inst)), 201


@bp.route("/api/instances/<inst_id>", methods=["GET"])
def api_instances_get(inst_id):
    with instances_lock:
        inst = instances.get(inst_id)
        if inst is None:
            return jsonify({"error": "Not found"}), 404
        d = _public_instance(inst)
    return jsonify(d)


@bp.route("/api/instances/<inst_id>/remove", methods=["DELETE"])
def api_instances_remove(inst_id):
    with instances_lock:
        inst = instances.get(inst_id)
        if inst is None:
            return jsonify({"error": "Not found"}), 404
        if inst["status"] not in ("stopped",):
            return jsonify({"error": "Instance must be stopped before removing"}), 409
    release_instance_reservations(inst_id)
    with instances_lock:
        instances.pop(inst_id, None)
    save_state()
    return jsonify({"status": "removed"})


@bp.route("/api/instances/<inst_id>/logs")
def api_instance_logs(inst_id):
    with instances_lock:
        inst = instances.get(inst_id)
    if inst is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"lines": read_log_file(inst["log_file"])})


@bp.route("/api/instances/<inst_id>/logs/stream")
def api_instance_logs_stream(inst_id):
    with instances_lock:
        inst = instances.get(inst_id)
    if inst is None:
        return jsonify({"error": "Not found"}), 404
    return Response(stream_log_file(inst["log_file"]),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
