# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

from flask import Blueprint, jsonify, request

from storage import get_storage

bp = Blueprint("request_log", __name__)


@bp.route("/api/request-log/conversations", methods=["GET"])
def list_conversations():
    try:
        limit = max(1, min(int(request.args.get("limit", 100)), 500))
    except (TypeError, ValueError):
        limit = 100
    return jsonify(get_storage().list_conversations(limit=limit))


@bp.route("/api/request-log/conversations/<conversation_id>", methods=["GET"])
def get_conversation(conversation_id: str):
    turns = get_storage().get_conversation_turns(conversation_id)
    if not turns:
        return jsonify({"error": "not found"}), 404
    return jsonify({"conversation_id": conversation_id, "turns": turns})
