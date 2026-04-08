# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import hashlib
import os
import threading

from flask import Flask, jsonify, make_response, render_template
from werkzeug.serving import make_server

from config import LLAMAMAN_PROXY_PORT, SECRET_KEY, logger
from core.state import load_state
from proxy import start_idle_proxy
from core.monitoring import start_background_poller

import api.auth as auth
import api.models as models
import api.presets as presets
import api.instances as instances
import api.downloads as downloads
import api.system_info as system_info
import api.llamaman as llamaman
import api.settings as settings
import api.api_keys as api_keys
import api.images as images


def create_app() -> Flask:
    application = Flask(__name__)

    # Secret key for session cookies - derived from SECRET_KEY env var,
    # or auto-generated from machine-id for zero-config single-user setups.
    if SECRET_KEY:
        application.secret_key = SECRET_KEY
    else:
        seed = "llamaman"
        try:
            with open("/etc/machine-id", "r") as f:
                seed = f.read().strip()
        except FileNotFoundError:
            pass
        application.secret_key = hashlib.sha256(seed.encode()).hexdigest()

    application.register_blueprint(auth.bp)
    application.register_blueprint(models.bp)
    application.register_blueprint(presets.bp)
    application.register_blueprint(instances.bp)
    application.register_blueprint(downloads.bp)
    application.register_blueprint(system_info.bp)
    application.register_blueprint(llamaman.bp)
    application.register_blueprint(settings.bp)
    application.register_blueprint(api_keys.bp)
    application.register_blueprint(images.bp)

    auth.init_auth(application)

    @application.route("/")
    def index():
        resp = make_response(render_template("index.html"))
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @application.route("/health")
    def health():
        return jsonify({"status": "ok"})

    return application


# ---------------------------------------------------------------------------
# Startup - runs on import (works for both gunicorn and python app.py)
# ---------------------------------------------------------------------------

# Load persisted state (instances, downloads) and collect proxies to restore
_deferred_proxies = load_state()

# Start background health/download poller
start_background_poller()

# Create the Flask app
app = create_app()

# Restore idle proxies from previous state
for _inst_id, _proxy_port, _internal_port in _deferred_proxies:
    try:
        start_idle_proxy(_inst_id, _proxy_port, _internal_port)
    except Exception as _e:
        logger.warning("Failed to restore proxy for %s: %s", _inst_id, _e)

# Start the llamaman proxy port (Ollama-compatible API) in a background thread.
# OpenWebUI connects here via OLLAMA_BASE_URL=http://llamaman:42069
_proxy_server = make_server("0.0.0.0", LLAMAMAN_PROXY_PORT, app, threaded=True)
_proxy_thread = threading.Thread(target=_proxy_server.serve_forever, daemon=True)
_proxy_thread.start()
logger.info("Llamaman proxy listening on port %d", LLAMAMAN_PROXY_PORT)


if __name__ == "__main__":
    # Direct execution: run the Flask dev server on port 5000
    app.run(host="0.0.0.0", port=5000, debug=False)
