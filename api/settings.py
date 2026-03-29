# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import time
import uuid

from flask import Blueprint, jsonify, request

from storage import get_storage

bp = Blueprint("settings", __name__)


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
    safe = dict(settings)
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
    settings = get_storage().merge_settings(data)
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
