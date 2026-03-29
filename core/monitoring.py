# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import threading
import time

import requests

from config import LLAMAMAN_IDLE_TIMEOUT, HEALTH_CHECK_TIMEOUT, logger
from core.helpers import cleanup_download_dir, is_pid_alive, kill_instance_process, is_llama_pid
from core.state import (
    instances, instances_lock,
    downloads, downloads_lock,
    save_state,
)
from proxy import idle_proxies, idle_proxies_lock, cleanup_orphan_idle_proxies, refresh_gate

_last_cleanup_at: float = 0.0
_CLEANUP_INTERVAL = 3600  # run at most once per hour

_last_orphan_scan_at: float = 0.0
_ORPHAN_SCAN_INTERVAL = 60  # seconds


def _run_cleanup() -> None:
    from storage import get_storage
    from api.instances import release_instance_reservations
    storage = get_storage()
    cleanup = storage.get_settings().get("cleanup", {})
    changed = False
    cleanup_patch = {}
    now = time.time()

    if cleanup.get("downloads_enabled"):
        cleanup_patch["downloads_last_run_at"] = now
        max_age_s = (cleanup.get("downloads_max_age_hours") or 24) * 3600
        cutoff = now - max_age_s
        with downloads_lock:
            to_remove = [
                did for did, dl in downloads.items()
                if dl["status"] in ("completed", "failed", "cancelled")
                and dl.get("started_at", 0) < cutoff
            ]
            for did in to_remove:
                del downloads[did]
                logger.info("Cleanup: removed download %s", did)
        if to_remove:
            changed = True

    if cleanup.get("instances_enabled"):
        cleanup_patch["instances_last_run_at"] = now
        max_age_s = (cleanup.get("instances_max_age_hours") or 24) * 3600
        cutoff = now - max_age_s
        with instances_lock:
            to_remove = [
                iid for iid, inst in instances.items()
                if inst["status"] == "stopped"
                and inst.get("started_at", 0) < cutoff
            ]
            for iid in to_remove:
                release_instance_reservations(iid)
                del instances[iid]
                logger.info("Cleanup: removed instance %s", iid)
        if to_remove:
            changed = True

    if changed:
        save_state()
    if cleanup_patch:
        storage.merge_settings({"cleanup": cleanup_patch})




def _background_poller():
    global _last_cleanup_at, _last_orphan_scan_at
    while True:
        time.sleep(5)

        now = time.time()

        # --- Periodic cleanup ---
        if now - _last_cleanup_at >= _CLEANUP_INTERVAL:
            _last_cleanup_at = now
            try:
                _run_cleanup()
            except Exception as e:
                logger.warning("Cleanup error: %s", e)

        # --- Periodic orphan scan ---
        if now - _last_orphan_scan_at >= _ORPHAN_SCAN_INTERVAL:
            _last_orphan_scan_at = now
            try:
                from core.state import adopt_orphans
                n = adopt_orphans()
                if n:
                    logger.info("Orphan scan: adopted %d untracked llama-server instance(s)", n)
            except Exception as e:
                logger.warning("Orphan scan error: %s", e)

        # --- Instance health ---
        with instances_lock:
            inst_ids = list(instances.keys())

        removed_orphans = cleanup_orphan_idle_proxies(set(inst_ids))
        if removed_orphans:
            logger.info("Proxy cleanup: removed %d orphan idle proxy listener(s)", removed_orphans)

        for inst_id in inst_ids:
            with instances_lock:
                inst = instances.get(inst_id)
                if inst is None or inst["status"] in ("stopped", "sleeping"):
                    continue
                port = inst.get("_internal_port") or inst["port"]
                proc = inst.get("_process")
                pid = inst.get("pid", 0)

            # Detect process death - via Popen handle or PID check.
            # When relying on PID alone, also verify the process is still
            # llama-server (guards against PID recycling after a crash).
            process_dead = False
            if proc is not None:
                process_dead = proc.poll() is not None
            elif pid > 0:
                process_dead = not is_pid_alive(pid) or not is_llama_pid(pid)

            if process_dead:
                with instances_lock:
                    inst = instances.get(inst_id)
                    if inst and inst["status"] not in ("stopped", "sleeping"):
                        kill_instance_process(inst)
                        inst["status"] = "stopped"
                        stats = inst.setdefault("stats", {})
                        stats["crash_count"] = stats.get("crash_count", 0) + 1
                        logger.info("Instance %s auto-stopped (process died, crashes: %d)",
                                    inst_id, stats["crash_count"])
                refresh_gate(inst_id)
                save_state()
                continue

            try:
                resp = requests.get(f"http://localhost:{port}/health", timeout=HEALTH_CHECK_TIMEOUT)
                new_status = "healthy" if resp.json().get("status") == "ok" else "starting"
            except Exception:
                new_status = "starting"

            status_changed = False
            with instances_lock:
                if inst_id in instances and instances[inst_id]["status"] not in ("stopped", "sleeping"):
                    old_status = instances[inst_id]["status"]
                    if new_status != old_status:
                        instances[inst_id]["status"] = new_status
                        status_changed = True
                    if new_status == "healthy" and old_status == "starting":
                        started = instances[inst_id].get("started_at", 0)
                        if started:
                            stats = instances[inst_id].setdefault("stats", {})
                            stats["model_load_time_s"] = round(time.time() - started, 1)
                        logger.info("Instance %s is now healthy (was %s)", inst_id, old_status)

            # Persist status transitions so they survive restarts
            if status_changed:
                save_state()

        # --- Idle timeout reaper ---
        for inst_id in inst_ids:
            with instances_lock:
                inst = instances.get(inst_id)
                if inst is None or inst["status"] not in ("healthy",):
                    continue

                config = inst.get("config", {})
                timeout_min = config.get("idle_timeout_min", 0)
                if inst.get("_llamaman_managed") and LLAMAMAN_IDLE_TIMEOUT > 0:
                    timeout_min = timeout_min or LLAMAMAN_IDLE_TIMEOUT

                if timeout_min <= 0:
                    continue

                last_req = inst.get("_last_request_at", inst.get("started_at", now))
                idle_secs = now - last_req

                if idle_secs < timeout_min * 60:
                    continue

                with idle_proxies_lock:
                    has_proxy = inst_id in idle_proxies
                is_llamaman = inst.get("_llamaman_managed", False)

            # Lazy import to break circular dependency
            from api.instances import stop_instance_by_id, sleep_instance_by_id

            if has_proxy or is_llamaman:
                sleep_instance_by_id(inst_id)
                logger.info("Idle reaper: %s slept after %d min idle",
                            inst_id, int(idle_secs / 60))
            else:
                stop_instance_by_id(inst_id)
                logger.info("Idle reaper: %s stopped after %d min idle",
                            inst_id, int(idle_secs / 60))

        # --- Download process monitoring ---
        with downloads_lock:
            dl_ids = list(downloads.keys())

        for dl_id in dl_ids:
            with downloads_lock:
                dl = downloads.get(dl_id)
                if dl is None or dl["status"] in ("completed", "failed", "cancelled", "paused"):
                    continue
                proc = dl.get("_process")

            if proc is None:
                continue

            exit_code = proc.poll()
            if exit_code is None:
                continue

            dest_path = ""
            with downloads_lock:
                dl = downloads.get(dl_id)
                if dl and dl["status"] not in ("cancelled",):
                    kill_instance_process(dl)
                    dl["status"] = "completed" if exit_code == 0 else "failed"
                    if exit_code != 0:
                        dest_path = dl.get("dest_path", "")
            save_state()
            if dest_path:
                cleanup_download_dir(dest_path)
                logger.info("Download %s failed - cleaned up %s", dl_id, dest_path)


def start_background_poller():
    logger.info(
        "Background poller started (cleanup=%ds, orphan-scan=%ds, poll=5s)",
        _CLEANUP_INTERVAL, _ORPHAN_SCAN_INTERVAL,
    )
    thread = threading.Thread(target=_background_poller, daemon=True)
    thread.start()
    return thread
