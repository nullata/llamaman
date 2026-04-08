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
INTERNAL_PORT_RANGE_START = int(os.environ.get("INTERNAL_PORT_RANGE_START", 9000))
INTERNAL_PORT_RANGE_END = int(os.environ.get("INTERNAL_PORT_RANGE_END", 9020))

PRESETS_FILE = os.path.join(DATA_DIR, "presets.json")
LLAMAMAN_MAX_MODELS = int(os.environ.get("LLAMAMAN_MAX_MODELS", 0))
LLAMAMAN_PROXY_PORT = int(os.environ.get("LLAMAMAN_PROXY_PORT", 42069))
LLAMAMAN_IDLE_TIMEOUT = int(os.environ.get("LLAMAMAN_IDLE_TIMEOUT", 0))  # minutes, 0=disabled
HEALTH_CHECK_TIMEOUT = int(os.environ.get("HEALTH_CHECK_TIMEOUT", 3))
MODEL_LOAD_TIMEOUT = int(os.environ.get("MODEL_LOAD_TIMEOUT", 300))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", 300))

STATE_FILE = os.path.join(DATA_DIR, "state.json")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
SECRET_KEY = os.environ.get("SECRET_KEY", "")

# Docker-in-Docker settings
# Fixed port llama-server listens on inside every spawned container.
LLAMA_CONTAINER_PORT = 8080
LLAMA_IMAGE = os.environ.get("LLAMA_IMAGE", "ghcr.io/ggml-org/llama.cpp:server-cuda")
LLAMA_NETWORK = os.environ.get("LLAMA_NETWORK", "llamaman-net")
LLAMA_CONTAINER_PREFIX = os.environ.get("LLAMA_CONTAINER_PREFIX", "llamaman-")
GPU_TYPE = os.environ.get("GPU_TYPE", "cuda")  # "cuda" or "rocm"
# Comma-separated GPU indices visible to all llama-server containers, e.g. "0,1,3".
# Empty (default) means all GPUs. Per-instance gpu_devices overrides this when set.
LLAMA_GPU_DEVICES = os.environ.get("LLAMA_GPU_DEVICES", "").strip()

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llamaman")
