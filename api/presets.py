# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

from flask import Blueprint, jsonify, request

from core.proxy_sampling import parse_proxy_sampling_config
from storage import get_storage

bp = Blueprint("presets", __name__)


def _normalize_model_path(model_path: str) -> str:
    """Ensure model_path is an absolute path (leading /).

    Flask's <path:> converter strips the leading / from the URL, so
    /api/presets/models/foo.gguf yields model_path='models/foo.gguf'
    but the storage key is '/models/foo.gguf'.
    """
    if not model_path.startswith("/"):
        model_path = "/" + model_path
    return model_path


@bp.route("/api/presets", methods=["GET"])
def api_presets_list():
    return jsonify(get_storage().get_all_presets())


@bp.route("/api/presets/<path:model_path>", methods=["GET"])
def api_preset_get(model_path):
    model_path = _normalize_model_path(model_path)
    preset = get_storage().get_preset(model_path)
    if preset is None:
        return jsonify({"error": "No preset for this model"}), 404
    return jsonify(preset)


@bp.route("/api/presets/<path:model_path>", methods=["PUT"])
def api_preset_save(model_path):
    model_path = _normalize_model_path(model_path)
    body = request.get_json(force=True)
    ctx_size = body.get("ctx_size")
    if ctx_size in (None, ""):
        return jsonify({"error": "ctx_size is required"}), 400
    try:
        ctx_size = int(ctx_size)
    except (TypeError, ValueError):
        return jsonify({"error": "ctx_size must be an integer"}), 400
    if ctx_size <= 0:
        return jsonify({"error": "ctx_size must be greater than 0"}), 400
    proxy_sampling_config, proxy_sampling_err = parse_proxy_sampling_config(body)
    if proxy_sampling_err:
        return jsonify({"error": proxy_sampling_err}), 400
    # Preserve existing meta fields (favorite, note) that aren't part of the launch form
    existing = get_storage().get_preset(model_path) or {}
    data = {
        "n_gpu_layers": body.get("n_gpu_layers", -1),
        "ctx_size": ctx_size,
        "threads": body.get("threads"),
        "memory_limit": body.get("memory_limit", ""),
        "parallel": body.get("parallel"),
        "extra_args": body.get("extra_args", ""),
        "gpu_devices": body.get("gpu_devices", ""),
        "idle_timeout_min": body.get("idle_timeout_min", 0),
        "max_concurrent": body.get("max_concurrent", 0),
        "max_queue_depth": body.get("max_queue_depth", 200),
        "share_queue": body.get("share_queue", False),
        "embedding_model": body.get("embedding_model", False),
        "favorite": body.get("favorite", existing.get("favorite", False)),
        "note": body.get("note", existing.get("note", "")),
        **proxy_sampling_config,
    }
    get_storage().save_preset(model_path, data)
    return jsonify({"status": "saved"})


@bp.route("/api/presets/<path:model_path>", methods=["PATCH"])
def api_preset_patch(model_path):
    """Partially update preset fields (e.g. favorite, note) without requiring a full preset."""
    model_path = _normalize_model_path(model_path)
    body = request.get_json(force=True)
    storage = get_storage()
    preset = storage.get_preset(model_path) or {}
    allowed = {"favorite", "note"}
    for key in allowed:
        if key in body:
            preset[key] = body[key]
    storage.save_preset(model_path, preset)
    return jsonify({"status": "saved"})


@bp.route("/api/presets/<path:model_path>", methods=["DELETE"])
def api_preset_delete(model_path):
    model_path = _normalize_model_path(model_path)
    get_storage().delete_preset(model_path)
    return jsonify({"status": "deleted"})
