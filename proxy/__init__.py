# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import json
import threading
import time

import requests
from werkzeug.serving import make_server
from werkzeug.wrappers import Request as WerkzeugRequest

from config import logger
from core.state import instances, instances_lock


# ---------------------------------------------------------------------------
# Per-instance request gate (concurrency limiter)
# ---------------------------------------------------------------------------

_instance_gates: dict[str, "RequestGate"] = {}
_shared_queue_gates: dict[str, "RequestGate"] = {}  # model_path -> gate
_gates_lock = threading.Lock()


class RequestGate:
    """Concurrency limiter with a bounded waiting queue.

    - max_concurrent: how many requests may be in-flight at once.
    - max_queue_depth: how many additional requests may wait.
    Requests beyond the queue depth are rejected immediately (429).
    """

    def __init__(self, max_concurrent: int, max_queue_depth: int = 200):
        self.max_concurrent = max_concurrent
        self.max_queue_depth = max_queue_depth
        self._semaphore = threading.Semaphore(max_concurrent)
        self._waiting = 0
        self._lock = threading.Lock()
        self.active = 0
        self.queued = 0

    def acquire(self, timeout: float = 300) -> bool:
        with self._lock:
            if self._waiting >= self.max_queue_depth:
                return False
            self._waiting += 1
            self.queued = self._waiting

        got = self._semaphore.acquire(timeout=timeout)

        with self._lock:
            self._waiting -= 1
            if got:
                self.active += 1
            self.queued = self._waiting
        return got

    def release(self):
        with self._lock:
            self.active = max(0, self.active - 1)
        self._semaphore.release()


def get_gate(inst_id: str) -> "RequestGate | None":
    with _gates_lock:
        return _instance_gates.get(inst_id)


def create_gate(inst_id: str, max_concurrent: int, max_queue_depth: int,
                model_path: str | None = None, share_queue: bool = False):
    if max_concurrent <= 0:
        return
    with _gates_lock:
        if share_queue and model_path:
            if model_path in _shared_queue_gates:
                gate = _shared_queue_gates[model_path]
            else:
                gate = RequestGate(max_concurrent, max_queue_depth)
                _shared_queue_gates[model_path] = gate
            _instance_gates[inst_id] = gate
        else:
            gate = RequestGate(max_concurrent, max_queue_depth)
            _instance_gates[inst_id] = gate


def remove_gate(inst_id: str):
    with _gates_lock:
        _instance_gates.pop(inst_id, None)


# ---------------------------------------------------------------------------
# Idle proxy servers: { inst_id: { server, thread, proxy_port, internal_port } }
# ---------------------------------------------------------------------------

idle_proxies: dict[str, dict] = {}
idle_proxies_lock = threading.Lock()

_GATED_PATHS = frozenset({
    "/v1/chat/completions", "/v1/completions", "/v1/embeddings",
    "/completion", "/chat/completions",
})


def _is_inference_request(environ: dict) -> bool:
    method = environ.get("REQUEST_METHOD", "GET").upper()
    if method != "POST":
        return False
    path = environ.get("PATH_INFO", "")
    return path in _GATED_PATHS


def _check_proxy_auth(environ, start_response):
    """Enforce bearer token auth on proxy ports when require_auth is enabled.

    Returns True if the request should be blocked (response already sent),
    False if the request is allowed to proceed.
    """
    from api.auth import is_require_auth_enabled, verify_bearer_token

    if not is_require_auth_enabled():
        return False

    error = verify_bearer_token(environ.get("HTTP_AUTHORIZATION", ""), strict=True)
    if not error:
        return False

    start_response("401 Unauthorized",
                   [("Content-Type", "application/json")])
    return True


def make_proxy_app(inst_id: str, internal_port: int):
    """Create a WSGI app that proxies requests to the llama-server,
    waking it from sleep if necessary."""
    def proxy_app(environ, start_response):
        # Lazy import to break circular dependency
        from api.instances import relaunch_sleeping_instance

        # Auth check - blocks the request if token is missing/invalid
        if _check_proxy_auth(environ, start_response):
            return [json.dumps({"error": "API key required"}).encode()]

        with instances_lock:
            inst = instances.get(inst_id)
            if inst:
                inst["_last_request_at"] = time.time()
                stats = inst.setdefault("stats", {})
                stats["total_requests"] = stats.get("total_requests", 0) + 1

        with instances_lock:
            inst = instances.get(inst_id)
            status = inst["status"] if inst else "stopped"

        if status == "sleeping":
            if not relaunch_sleeping_instance(inst_id):
                start_response("503 Service Unavailable",
                               [("Content-Type", "application/json")])
                return [json.dumps({"error": "failed to wake model"}).encode()]

        gate = get_gate(inst_id) if _is_inference_request(environ) else None
        if gate:
            if not gate.acquire(timeout=300):
                start_response("429 Too Many Requests",
                               [("Content-Type", "application/json")])
                return [json.dumps({"error": "request queue full"}).encode()]

        try:
            req = WerkzeugRequest(environ)
            target = f"http://localhost:{internal_port}{req.path}"
            if req.query_string:
                target += f"?{req.query_string.decode()}"

            headers = {k: v for k, v in req.headers if k.lower() != "host"}

            try:
                resp = requests.request(
                    method=req.method,
                    url=target,
                    headers=headers,
                    data=req.get_data(),
                    stream=True,
                    timeout=300,
                )
            except Exception as e:
                start_response("502 Bad Gateway",
                               [("Content-Type", "application/json")])
                return [json.dumps({"error": str(e)}).encode()]

            resp_headers = [
                (k, v) for k, v in resp.headers.items()
                if k.lower() not in ("transfer-encoding", "connection")
            ]
            start_response(f"{resp.status_code} {resp.reason}", resp_headers)

            def _relay_and_close():
                try:
                    yield from resp.iter_content(chunk_size=None)
                finally:
                    resp.close()
                    if gate:
                        gate.release()
            return _relay_and_close()
        except Exception:
            if gate:
                gate.release()
            raise

    return proxy_app


def start_idle_proxy(inst_id: str, proxy_port: int, internal_port: int):
    proxy_app = make_proxy_app(inst_id, internal_port)
    server = make_server("0.0.0.0", proxy_port, proxy_app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    with idle_proxies_lock:
        idle_proxies[inst_id] = {
            "server": server,
            "thread": thread,
            "proxy_port": proxy_port,
            "internal_port": internal_port,
        }
    logger.info("Idle proxy: port %d -> internal %d (instance %s)",
                proxy_port, internal_port, inst_id)


def stop_idle_proxy(inst_id: str):
    with idle_proxies_lock:
        proxy = idle_proxies.pop(inst_id, None)
    if proxy:
        try:
            proxy["server"].shutdown()
        except Exception:
            pass
        logger.info("Idle proxy stopped for instance %s", inst_id)
