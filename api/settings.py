# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import time
import uuid

from flask import Blueprint, jsonify, request

from storage import get_storage

bp = Blueprint("settings", __name__)

DEFAULT_GLOBAL_SPEED_LIMIT_MBPS = 0.0
DEFAULT_RETRY_COUNT_PER_FAILED_DOWNLOAD = 3


def _get_hf_tokens() -> list[dict]:
    settings = get_storage().get_settings()
    tokens = settings.get("huggingface_tokens", [])
    return [t for t in tokens if isinstance(t, dict) and t.get("id")]


def _save_hf_tokens(tokens: list[dict]) -> None:
    storage = get_storage()
    settings = storage.get_settings()
    settings["huggingface_tokens"] = tokens
    storage.save_settings(settings)


def _mask_hf_token(token: str) -> str:
    if not token:
        return ""
    if len(token) <= 10:
        return token[:3] + "..." + token[-2:]
    return token[:6] + "..." + token[-4:]


def _coerce_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off", ""):
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _coerce_non_negative_float(value, default: float = 0.0) -> float:
    try:
        coerced = float(value or 0)
    except (TypeError, ValueError):
        return default
    return max(0.0, coerced)


def _coerce_min_int(value, default: int, minimum: int) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, coerced)


def _normalize_settings_patch(settings: dict) -> dict:
    normalized = dict(settings)
    if "global_speed_limit_mbps" in normalized:
        normalized["global_speed_limit_mbps"] = _coerce_non_negative_float(
            normalized.get("global_speed_limit_mbps"),
            default=DEFAULT_GLOBAL_SPEED_LIMIT_MBPS,
        )
    if "auto_retry_failed_downloads" in normalized:
        normalized["auto_retry_failed_downloads"] = _coerce_bool(
            normalized.get("auto_retry_failed_downloads"),
            default=False,
        )
    if "retry_count_per_failed_download" in normalized:
        normalized["retry_count_per_failed_download"] = _coerce_min_int(
            normalized.get("retry_count_per_failed_download"),
            default=DEFAULT_RETRY_COUNT_PER_FAILED_DOWNLOAD,
            minimum=1,
        )
    return normalized


def _apply_settings_defaults(settings: dict) -> dict:
    normalized = dict(settings)
    normalized["global_speed_limit_mbps"] = _coerce_non_negative_float(
        normalized.get("global_speed_limit_mbps"),
        default=DEFAULT_GLOBAL_SPEED_LIMIT_MBPS,
    )
    normalized["auto_retry_failed_downloads"] = _coerce_bool(
        normalized.get("auto_retry_failed_downloads"),
        default=False,
    )
    normalized["retry_count_per_failed_download"] = _coerce_min_int(
        normalized.get("retry_count_per_failed_download"),
        default=DEFAULT_RETRY_COUNT_PER_FAILED_DOWNLOAD,
        minimum=1,
    )
    return normalized


def serialize_hf_token(token_entry: dict) -> dict:
    return {
        "id": token_entry["id"],
        "name": token_entry.get("name", "Untitled"),
        "preview": _mask_hf_token(token_entry.get("token", "")),
        "created_at": token_entry.get("created_at", 0),
    }


def get_hf_token_secret(token_id: str) -> str | None:
    for token in _get_hf_tokens():
        if token.get("id") == token_id:
            secret = token.get("token", "").strip()
            return secret or None
    return None


def _sanitize_settings(settings: dict) -> dict:
    safe = _apply_settings_defaults(settings)
    if "huggingface_tokens" in safe:
        safe["huggingface_tokens"] = [serialize_hf_token(token) for token in _get_hf_tokens()]
    return safe


@bp.route("/api/settings")
def get_settings():
    return jsonify(_sanitize_settings(get_storage().get_settings()))


@bp.route("/api/settings", methods=["POST"])
def save_settings():
    data = request.get_json(silent=True) or {}
    data.pop("huggingface_tokens", None)
    settings = get_storage().merge_settings(_normalize_settings_patch(data))
    return jsonify({"ok": True, "settings": _sanitize_settings(settings)})


@bp.route("/api/settings/huggingface-tokens")
def list_huggingface_tokens():
    return jsonify([serialize_hf_token(token) for token in _get_hf_tokens()])


@bp.route("/api/settings/huggingface-tokens", methods=["POST"])
def create_huggingface_token():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip() or "Untitled"
    token = data.get("token", "").strip()
    if not token:
        return jsonify({"error": "token is required"}), 400

    tokens = _get_hf_tokens()
    entry = {
        "id": uuid.uuid4().hex,
        "name": name,
        "token": token,
        "created_at": int(time.time()),
    }
    tokens.append(entry)
    _save_hf_tokens(tokens)
    return jsonify(serialize_hf_token(entry)), 201


@bp.route("/api/settings/huggingface-tokens/<token_id>", methods=["DELETE"])
def delete_huggingface_token(token_id):
    tokens = _get_hf_tokens()
    kept = [token for token in tokens if token.get("id") != token_id]
    if len(kept) == len(tokens):
        return jsonify({"error": "Not found"}), 404
    _save_hf_tokens(kept)
    return jsonify({"ok": True})
