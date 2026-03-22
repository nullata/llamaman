# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import os
import logging
from pathlib import Path

VERSION = (Path(__file__).parent / "VERSION").read_text().strip()

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
LOGS_DIR = os.environ.get("LOGS_DIR", "/tmp/llama-logs")
PORT_RANGE_START = int(os.environ.get("PORT_RANGE_START", 8000))
PORT_RANGE_END = int(os.environ.get("PORT_RANGE_END", 8020))

PRESETS_FILE = os.path.join(DATA_DIR, "presets.json")
LLAMAMAN_MAX_MODELS = int(os.environ.get("LLAMAMAN_MAX_MODELS", 0))
LLAMAMAN_PROXY_PORT = int(os.environ.get("LLAMAMAN_PROXY_PORT", 42069))
LLAMAMAN_IDLE_TIMEOUT = int(os.environ.get("LLAMAMAN_IDLE_TIMEOUT", 0))  # minutes, 0=disabled
HEALTH_CHECK_TIMEOUT = int(os.environ.get("HEALTH_CHECK_TIMEOUT", 3))

STATE_FILE = os.path.join(DATA_DIR, "state.json")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
SECRET_KEY = os.environ.get("SECRET_KEY", "")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llamaman")
