# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import os
import re
import shutil
import struct
from pathlib import Path

from flask import Blueprint, jsonify, request

from config import MODELS_DIR
from core.helpers import format_size
from core.state import instances, instances_lock

bp = Blueprint("models", __name__)

_QUANT_PATTERN = re.compile(
    r'(?i)(bf16|f16|f32|q[0-9]_[0-9]|q[0-9]+_k(?:_[sml])?|iq[0-9]+_[a-z]+|q[0-9]+)',
)


def detect_quant(name: str) -> str:
    m = _QUANT_PATTERN.search(name)
    return m.group(1).upper() if m else ""


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except OSError:
        pass
    return total


def discover_models(models_dir: str) -> list[dict]:
    found = []
    base = Path(models_dir)
    if not base.exists():
        return found

    for config_file in base.rglob("config.json"):
        model_dir = config_file.parent
        size = _dir_size(model_dir)
        found.append({
            "name": model_dir.name,
            "path": str(model_dir),
            "type": "hf",
            "quant": "",
            "size_bytes": size,
            "size_display": format_size(size),
        })

    for gguf_file in base.rglob("*.gguf"):
        size = gguf_file.stat().st_size
        found.append({
            "name": gguf_file.stem,
            "path": str(gguf_file),
            "type": "gguf",
            "quant": detect_quant(gguf_file.stem),
            "size_bytes": size,
            "size_display": format_size(size),
        })

    seen = set()
    unique = []
    for m in found:
        if m["path"] not in seen:
            seen.add(m["path"])
            unique.append(m)
    return unique


# ---------------------------------------------------------------------------
# GGUF metadata
# ---------------------------------------------------------------------------

def _read_gguf_string(f):
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")


def _read_gguf_value(f, vtype: int):
    if vtype == 0: return struct.unpack("<B", f.read(1))[0]
    if vtype == 1: return struct.unpack("<b", f.read(1))[0]
    if vtype == 2: return struct.unpack("<H", f.read(2))[0]
    if vtype == 3: return struct.unpack("<h", f.read(2))[0]
    if vtype == 4: return struct.unpack("<I", f.read(4))[0]
    if vtype == 5: return struct.unpack("<i", f.read(4))[0]
    if vtype == 6: return struct.unpack("<f", f.read(4))[0]
    if vtype == 7: return bool(struct.unpack("<B", f.read(1))[0])
    if vtype == 8: return _read_gguf_string(f)
    if vtype == 9:
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        return [_read_gguf_value(f, elem_type) for _ in range(count)]
    if vtype == 10: return struct.unpack("<Q", f.read(8))[0]
    if vtype == 11: return struct.unpack("<q", f.read(8))[0]
    if vtype == 12: return struct.unpack("<d", f.read(8))[0]
    raise ValueError(f"Unknown GGUF type: {vtype}")


def get_gguf_metadata(filepath: str) -> dict:
    """Read key architecture metadata from a GGUF file for layer/VRAM calculation.

    Returns a dict with fields: block_count, embedding_length, feed_forward_length,
    head_count, head_count_kv, vocab_size (all int or None if not found).
    """
    meta: dict = {
        "block_count": None,
        "embedding_length": None,
        "feed_forward_length": None,
        "head_count": None,
        "head_count_kv": None,
        "vocab_size": None,
    }
    _required = {"block_count", "embedding_length", "feed_forward_length", "head_count"}
    _wanted = _required | {"head_count_kv", "vocab_size"}
    _found: set[str] = set()
    try:
        with open(filepath, "rb") as f:
            if f.read(4) != b"GGUF":
                return meta
            struct.unpack("<I", f.read(4))   # version
            struct.unpack("<Q", f.read(8))   # tensor_count
            kv_count = struct.unpack("<Q", f.read(8))[0]
            for _ in range(kv_count):
                key = _read_gguf_string(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                value = _read_gguf_value(f, vtype)
                if key.endswith(".block_count"):
                    meta["block_count"] = int(value); _found.add("block_count")
                elif key.endswith(".embedding_length"):
                    meta["embedding_length"] = int(value); _found.add("embedding_length")
                elif key.endswith(".feed_forward_length"):
                    meta["feed_forward_length"] = int(value); _found.add("feed_forward_length")
                elif key.endswith(".attention.head_count"):
                    meta["head_count"] = int(value); _found.add("head_count")
                elif key.endswith(".attention.head_count_kv"):
                    meta["head_count_kv"] = int(value); _found.add("head_count_kv")
                elif key.endswith(".vocab_size"):
                    meta["vocab_size"] = int(value); _found.add("vocab_size")
                if _found >= _wanted:
                    break
                # Once past architecture keys into tokenizer arrays, stop if
                # all required fields are already found.
                if key.startswith("tokenizer.") and _required <= _found:
                    break
    except Exception:
        pass
    return meta


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@bp.route("/api/models")
def api_models():
    models = discover_models(MODELS_DIR)
    return jsonify(models)


@bp.route("/api/models/delete", methods=["POST"])
def api_models_delete():
    body = request.get_json(force=True)
    model_path = body.get("path", "").strip()
    if not model_path:
        return jsonify({"error": "path is required"}), 400

    try:
        resolved = os.path.realpath(model_path)
        models_real = os.path.realpath(MODELS_DIR)
        if not resolved.startswith(models_real + os.sep) and resolved != models_real:
            return jsonify({"error": "path is outside models directory"}), 403
    except Exception:
        return jsonify({"error": "invalid path"}), 400

    if not os.path.exists(resolved):
        return jsonify({"error": "path does not exist"}), 404

    with instances_lock:
        for inst in instances.values():
            if inst["status"] in ("stopped",):
                continue
            if os.path.realpath(inst["model_path"]) == resolved or \
               resolved.startswith(os.path.realpath(inst["model_path"]) + os.sep):
                return jsonify({"error": f"model is in use by instance on port {inst['port']}"}), 409

    try:
        if os.path.isdir(resolved):
            shutil.rmtree(resolved)
        else:
            os.remove(resolved)
            parent = os.path.dirname(resolved)
            if parent != models_real and os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    from config import logger
    logger.info("Deleted model: %s", resolved)
    return jsonify({"status": "deleted"})


@bp.route("/api/model-layers")
def api_model_layers():
    model_path = request.args.get("path", "").strip()
    if not model_path:
        return jsonify({"error": "path is required"}), 400
    if not model_path.lower().endswith(".gguf"):
        return jsonify({"layers": None})
    meta = get_gguf_metadata(model_path)
    meta["layers"] = meta["block_count"]  # backward-compat alias
    meta["quant"] = detect_quant(Path(model_path).stem)
    return jsonify(meta)


@bp.route("/api/disk-space")
def api_disk_space():
    try:
        usage = shutil.disk_usage(MODELS_DIR)
        return jsonify({
            "total_gb": round(usage.total / (1024**3), 1),
            "used_gb": round(usage.used / (1024**3), 1),
            "free_gb": round(usage.free / (1024**3), 1),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
