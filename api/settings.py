# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

from flask import Blueprint, jsonify, request

from storage import get_storage

bp = Blueprint("settings", __name__)


@bp.route("/api/settings")
def get_settings():
    return jsonify(get_storage().get_settings())


@bp.route("/api/settings", methods=["POST"])
def save_settings():
    data = request.get_json(silent=True) or {}
    settings = get_storage().merge_settings(data)
    return jsonify({"ok": True, "settings": settings})
