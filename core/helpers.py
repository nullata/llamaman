# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path


def is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)  # signal 0 = just check existence
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but we can't signal it


def format_size(size_bytes: int) -> str:
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.1f} GB"
    return f"{size_bytes / (1024**2):.0f} MB"


def public_dict(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def cleanup_download_dir(dest_path: str) -> None:
    """Delete a partial/failed download directory, guarded to stay inside MODELS_DIR."""
    from config import MODELS_DIR
    try:
        real_dest = os.path.realpath(dest_path)
        real_models = os.path.realpath(MODELS_DIR)
        if not real_dest.startswith(real_models + os.sep) and real_dest != real_models:
            return  # never delete outside models dir
        if os.path.isdir(real_dest):
            shutil.rmtree(real_dest)
        elif os.path.isfile(real_dest):
            os.remove(real_dest)
    except Exception:
        pass


def build_llama_cmd(model_path: str, port: int, config: dict) -> list[str]:
    cmd = [
        "/app/llama-server",
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--n-gpu-layers", str(config.get("n_gpu_layers", -1)),
        "--ctx-size", str(config.get("ctx_size", 4096)),
    ]
    if config.get("threads"):
        cmd += ["--threads", str(int(config["threads"]))]
    if config.get("parallel"):
        cmd += ["--parallel", str(int(config["parallel"]))]
    if config.get("extra_args"):
        cmd += shlex.split(config["extra_args"])
    return cmd


def kill_pid(pid: int) -> None:
    """Best-effort kill of a process by PID. SIGTERM first, SIGKILL after 10s."""
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception:
        pass


def kill_instance_process(inst: dict):
    proc = inst.get("_process")
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    elif proc is None and inst.get("pid", 0) > 0:
        kill_pid(inst["pid"])
    fh = inst.get("_log_fh")
    if fh:
        try:
            fh.close()
        except Exception:
            pass
    inst["_process"] = None
    inst["_log_fh"] = None


def read_log_file(log_path: str, tail: int = 100) -> list[str]:
    try:
        with open(log_path, "r", errors="replace") as f:
            lines = f.readlines()
        return lines[-tail:]
    except FileNotFoundError:
        return []


def stream_log_file(log_file):
    """Generator that tails a log file and yields SSE events."""
    try:
        with open(log_file, "r", errors="replace") as f:
            content = f.read()
            if content:
                yield f"data: {json.dumps({'lines': content.splitlines(keepends=True)})}\n\n"
            while True:
                line = f.readline()
                if line:
                    yield f"data: {json.dumps({'lines': [line]})}\n\n"
                else:
                    time.sleep(0.5)
                    yield ": keepalive\n\n"
    except GeneratorExit:
        return
    except Exception:
        return


def _parse_llama_cmdline(pid: int) -> dict | None:
    """Read /proc/<pid>/cmdline and parse it if it's a llama-server process.

    Returns a dict with model_path, port, and config keys, or None if the
    process is not a llama-server or the cmdline can't be read.
    """
    try:
        exe = os.readlink(f"/proc/{pid}/exe")
        if Path(exe).name != "llama-server":
            return None
    except (FileNotFoundError, PermissionError, OSError):
        return None

    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            args = f.read().decode("utf-8", errors="replace").split("\x00")
    except (FileNotFoundError, PermissionError):
        return None

    model_path = None
    port = None
    n_gpu_layers = -1
    ctx_size = 4096
    threads = None
    parallel = None

    i = 0
    while i < len(args):
        a = args[i]
        nxt = args[i + 1] if i + 1 < len(args) else None
        if a in ("--model", "-m") and nxt:
            model_path = nxt; i += 2
        elif a == "--port" and nxt:
            try: port = int(nxt)
            except ValueError: pass
            i += 2
        elif a in ("--n-gpu-layers", "-ngl") and nxt:
            try: n_gpu_layers = int(nxt)
            except ValueError: pass
            i += 2
        elif a in ("--ctx-size", "-c") and nxt:
            try: ctx_size = int(nxt)
            except ValueError: pass
            i += 2
        elif a in ("--threads", "-t") and nxt:
            try: threads = int(nxt)
            except ValueError: pass
            i += 2
        elif a in ("--parallel", "-np") and nxt:
            try: parallel = int(nxt)
            except ValueError: pass
            i += 2
        else:
            i += 1

    if not model_path or not port:
        return None

    return {
        "pid": pid,
        "model_path": model_path,
        "port": port,
        "config": {
            "n_gpu_layers": n_gpu_layers,
            "ctx_size": ctx_size,
            "threads": threads,
            "parallel": parallel,
            "extra_args": "",
            "gpu_devices": None,
            "idle_timeout_min": 0,
            "max_concurrent": 0,
            "max_queue_depth": 200,
            "share_queue": False,
            "embedding_model": False,
            "proxy_sampling_override_enabled": False,
            "proxy_sampling_temperature": 0.8,
            "proxy_sampling_top_k": 40,
            "proxy_sampling_top_p": 0.95,
            "proxy_sampling_presence_penalty": 0.0,
        },
    }


def is_llama_pid(pid: int) -> bool:
    """Return True if the given PID is a running llama-server process."""
    return _parse_llama_cmdline(pid) is not None


def scan_llama_server_processes() -> list[dict]:
    """Return a list of parsed config dicts for all running llama-server processes."""
    results = []
    try:
        entries = os.listdir("/proc")
    except PermissionError:
        return results
    for entry in entries:
        if not entry.isdigit():
            continue
        info = _parse_llama_cmdline(int(entry))
        if info:
            results.append(info)
    return results


def is_port_available(port: int, host: str = "0.0.0.0") -> bool:
    """Return True if the TCP port can be bound locally.

    Note: inherent TOCTOU race - the port may be taken between this check
    and the actual bind by the child process. Callers must handle Popen/bind
    failures gracefully.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def find_available_port(
    exclude: set[int] | None = None,
    range_start: int | None = None,
    range_end: int | None = None,
) -> int | None:
    from core.state import instances, instances_lock
    from proxy import idle_proxies, idle_proxies_lock
    from config import PORT_RANGE_START, PORT_RANGE_END

    exclude = exclude or set()
    range_start = PORT_RANGE_START if range_start is None else range_start
    range_end = PORT_RANGE_END if range_end is None else range_end
    with instances_lock:
        used = set()
        for i in instances.values():
            if i["status"] not in ("stopped",):
                used.add(i["port"])
                if i.get("_internal_port"):
                    used.add(i["_internal_port"])
    with idle_proxies_lock:
        for p in idle_proxies.values():
            used.add(p["internal_port"])
    used |= exclude
    for p in range(range_start, range_end + 1):
        # Only return ports that are both untracked by LlamaMan and
        # actually bindable on the host right now.
        if p not in used and is_port_available(p):
            return p
    return None
