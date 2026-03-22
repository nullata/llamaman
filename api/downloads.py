# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import json
import os
import sys
import subprocess
import time
import uuid
from pathlib import Path

from flask import Blueprint, Response, jsonify, request

from config import LOGS_DIR, MODELS_DIR, logger
from core.helpers import kill_instance_process, public_dict, read_log_file, stream_log_file
from core.state import downloads, downloads_lock, save_state

bp = Blueprint("downloads", __name__)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@bp.route("/api/downloads", methods=["GET"])
def api_downloads_list():
    with downloads_lock:
        safe = [public_dict(dl) for dl in downloads.values()]
    return jsonify(safe)


@bp.route("/api/downloads", methods=["POST"])
def api_downloads_create():
    body = request.get_json(force=True)
    repo_id = body.get("repo_id", "").strip()
    if not repo_id:
        return jsonify({"error": "repo_id is required"}), 400

    filename = body.get("filename", "").strip()
    token = body.get("hf_token", "").strip()
    speed_limit_mbps = body.get("speed_limit_mbps", 0)

    dest_name = repo_id.split("/")[-1]
    if filename:
        dest_name = Path(filename).stem
    dest_path = os.path.join(MODELS_DIR, dest_name)
    os.makedirs(dest_path, exist_ok=True)

    dl_id = str(uuid.uuid4())
    log_file = os.path.join(LOGS_DIR, f"dl-{dl_id}.log")

    env = {
        **os.environ,
        "HF_REPO_ID": repo_id,
        "HF_LOCAL_DIR": dest_path,
        "HF_FILENAME": filename,
        "HF_TOKEN": token,
        "HF_SPEED_LIMIT": str(int(float(speed_limit_mbps) * 1_000_000 / 8)) if speed_limit_mbps else "0",
        "PYTHONUNBUFFERED": "1",
    }

    try:
        log_fh = open(log_file, "w", buffering=1)
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "core.downloader"],
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    dl = {
        "id": dl_id,
        "repo_id": repo_id,
        "filename": filename,
        "dest_path": dest_path,
        "status": "downloading",
        "pid": proc.pid,
        "log_file": log_file,
        "started_at": time.time(),
        "_process": proc,
        "_log_fh": log_fh,
    }

    with downloads_lock:
        downloads[dl_id] = dl

    logger.info("Download started: %s -> %s (pid %d)", repo_id, dest_path, proc.pid)
    save_state()
    return jsonify(public_dict(dl)), 201


@bp.route("/api/downloads/<dl_id>", methods=["GET"])
def api_downloads_get(dl_id):
    with downloads_lock:
        dl = downloads.get(dl_id)
        if dl is None:
            return jsonify({"error": "Not found"}), 404
        return jsonify(public_dict(dl))


@bp.route("/api/downloads/<dl_id>", methods=["DELETE"])
def api_downloads_cancel(dl_id):
    with downloads_lock:
        dl = downloads.get(dl_id)
        if dl is None:
            return jsonify({"error": "Not found"}), 404
        kill_instance_process(dl)
        dl["status"] = "cancelled"

    save_state()
    return jsonify({"status": "cancelled"})


@bp.route("/api/downloads/<dl_id>/remove", methods=["DELETE"])
def api_downloads_remove(dl_id):
    with downloads_lock:
        dl = downloads.get(dl_id)
        if dl is None:
            return jsonify({"error": "Not found"}), 404
        if dl["status"] == "downloading":
            return jsonify({"error": "Cannot remove active download, cancel it first"}), 409
        del downloads[dl_id]
    save_state()
    return jsonify({"status": "removed"})


@bp.route("/api/downloads/<dl_id>/logs")
def api_download_logs(dl_id):
    with downloads_lock:
        dl = downloads.get(dl_id)
    if dl is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"lines": read_log_file(dl["log_file"], tail=200)})


@bp.route("/api/downloads/<dl_id>/logs/stream")
def api_download_logs_stream(dl_id):
    with downloads_lock:
        dl = downloads.get(dl_id)
    if dl is None:
        return jsonify({"error": "Not found"}), 404
    return Response(stream_log_file(dl["log_file"]),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
