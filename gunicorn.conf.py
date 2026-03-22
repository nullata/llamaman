# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

# Gunicorn configuration for LlamaMan
#
# Usage:
#   gunicorn app:app
#
# Or with explicit config:
#   gunicorn -c gunicorn.conf.py app:app

import os

# Bind to port 5000 (UI + API), overridable via env
bind = os.environ.get("GUNICORN_BIND", "0.0.0.0:5000")

# IMPORTANT: Must use exactly 1 worker. The app uses in-memory state
# (instances, downloads, locks) that cannot be shared across processes.
# Use threads for concurrency instead.
workers = 1
threads = int(os.environ.get("GUNICORN_THREADS", 8))

# Worker class - gthread supports threading within a single worker
worker_class = "gthread"

# Timeout - model launches can take a while
timeout = int(os.environ.get("GUNICORN_TIMEOUT", 300))

# Graceful shutdown timeout
graceful_timeout = 30

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("GUNICORN_LOG_LEVEL", "info")

# IMPORTANT: preload_app MUST be False.  With preload_app=True the module-level
# code (load_state, background poller, proxy thread) runs in the *master*
# process.  The single worker is a fork with its own copy of memory, so it
# never sees status updates from the poller, instances launched by the proxy,
# or any state changes.  Worker crashes also reset state to the master's
# original snapshot instead of reading from disk.
preload_app = False
