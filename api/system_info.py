# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import subprocess

import psutil
from flask import Blueprint, jsonify

bp = Blueprint("system_info", __name__)


def _read_cgroup_value(path: str) -> int | None:
    try:
        with open(path) as f:
            val = int(f.read().strip())
            return val
    except (FileNotFoundError, ValueError, OSError):
        return None


def _read_cgroup_stats(path: str) -> dict[str, int] | None:
    try:
        stats = {}
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                key, value = parts
                stats[key] = int(value)
        return stats
    except (FileNotFoundError, ValueError, OSError):
        return None


def _get_container_cpu_limit() -> float | None:
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            parts = f.read().strip().split()
            if parts[0] == "max":
                return None
            quota, period = int(parts[0]), int(parts[1])
            return quota / period
    except (FileNotFoundError, OSError, ValueError, IndexError):
        pass
    quota = _read_cgroup_value("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_cgroup_value("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota and period and quota > 0:
        return quota / period
    return None


def _get_container_memory_limit() -> int | None:
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            val = f.read().strip()
            if val == "max":
                return None
            return int(val)
    except (FileNotFoundError, OSError, ValueError):
        pass
    limit = _read_cgroup_value("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if limit and limit < (1 << 62):
        return limit
    return None


def _get_container_memory_usage() -> int | None:
    usage = _read_cgroup_value("/sys/fs/cgroup/memory.current")
    if usage is not None:
        return usage
    return _read_cgroup_value("/sys/fs/cgroup/memory/memory.usage_in_bytes")


def _get_container_inactive_file_bytes() -> int:
    stats = _read_cgroup_stats("/sys/fs/cgroup/memory.stat")
    if stats:
        return max(
            stats.get("inactive_file", stats.get("total_inactive_file", 0)),
            0,
        )

    legacy_stats = _read_cgroup_stats("/sys/fs/cgroup/memory/memory.stat")
    if legacy_stats:
        return max(legacy_stats.get("total_inactive_file", 0), 0)

    return 0


@bp.route("/api/system-info")
def api_system_info():
    try:
        cpu_limit = _get_container_cpu_limit()
        cpu_cores = cpu_limit if cpu_limit else (psutil.cpu_count(logical=True) or 1)
        cpu_percent = psutil.cpu_percent(interval=0.3)

        mem_limit = _get_container_memory_limit()
        mem_usage = _get_container_memory_usage()

        if mem_limit and mem_usage is not None:
            # cgroup usage includes page cache for recently-read GGUF files,
            # which can stay charged after a model exits. Subtract inactive
            # file cache so the UI tracks the active working set instead.
            inactive_file = min(_get_container_inactive_file_bytes(), mem_usage)
            working_set = max(mem_usage - inactive_file, 0)
            ram_total_mb = round(mem_limit / (1024 * 1024))
            ram_used_mb = round(working_set / (1024 * 1024))
            ram_free_mb = ram_total_mb - ram_used_mb
            ram_percent = round(working_set / mem_limit * 100, 1) if mem_limit > 0 else 0
        else:
            vm = psutil.virtual_memory()
            ram_total_mb = round(vm.total / (1024 * 1024))
            ram_used_mb = round(vm.used / (1024 * 1024))
            ram_free_mb = round(vm.available / (1024 * 1024))
            ram_percent = vm.percent

        return jsonify({
            "cpu_percent": cpu_percent,
            "cpu_cores": round(cpu_cores, 1),
            "ram_total_mb": ram_total_mb,
            "ram_used_mb": ram_used_mb,
            "ram_free_mb": ram_free_mb,
            "ram_percent": ram_percent,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _query_nvidia() -> list[dict] | None:
    """Query NVIDIA GPUs via nvidia-smi. Returns list of dicts or None."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": int(parts[2]),
                    "memory_total_mb": int(parts[3]),
                    "memory_free_mb": int(parts[4]),
                    "utilization_pct": int(parts[5]),
                })
        return gpus
    except FileNotFoundError:
        return None


def _query_rocm() -> list[dict] | None:
    """Query AMD GPUs via rocm-smi. Returns list of dicts or None."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--showuse", "--showproductname", "--csv"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None

        # rocm-smi --csv output varies by version; parse what we can
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if len(lines) < 2:
            return None

        # Try CSV header-based parsing
        header = [h.strip().lower() for h in lines[0].split(",")]
        gpus = []
        for line in lines[1:]:
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < len(header):
                continue
            row = dict(zip(header, cols))
            # Field names differ across rocm-smi versions
            idx = row.get("device", row.get("gpu", str(len(gpus))))
            name = row.get("card series", row.get("card model", row.get("name", "AMD GPU")))
            vram_used = row.get("vram used", row.get("used vram (b)", "0"))
            vram_total = row.get("vram total", row.get("total vram (b)", "0"))
            gpu_use = row.get("gpu use (%)", row.get("gpu busy", "0"))

            # rocm-smi may report bytes or MB depending on flags
            used = int(vram_used)
            total = int(vram_total)
            # If values look like bytes (> 1GB), convert to MB
            if total > 1_000_000:
                used = used // (1024 * 1024)
                total = total // (1024 * 1024)

            gpus.append({
                "index": int(idx) if idx.isdigit() else len(gpus),
                "name": name,
                "memory_used_mb": used,
                "memory_total_mb": total,
                "memory_free_mb": total - used,
                "utilization_pct": int(gpu_use.replace("%", "")) if gpu_use.replace("%", "").isdigit() else 0,
            })
        return gpus if gpus else None
    except FileNotFoundError:
        return None


@bp.route("/api/gpu-info")
def api_gpu_info():
    try:
        # Try NVIDIA first, then AMD ROCm
        gpus = _query_nvidia()
        if gpus is not None:
            return jsonify({"gpus": gpus})

        gpus = _query_rocm()
        if gpus is not None:
            return jsonify({"gpus": gpus})

        return jsonify({"error": "No GPU tools found (nvidia-smi / rocm-smi)", "gpus": []}), 500
    except Exception as e:
        return jsonify({"error": str(e), "gpus": []}), 500
