# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import os

from config import STATE_FILE, PRESETS_FILE, USERS_FILE, SETTINGS_FILE

_backend = None


def get_storage():
    """Return the singleton storage backend.

    Uses MariaDBBackend if DATABASE_URL is set, otherwise JsonBackend.
    """
    global _backend
    if _backend is not None:
        return _backend

    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        from storage.mariadb_backend import MariaDBBackend
        _backend = MariaDBBackend(database_url)
    else:
        from storage.json_backend import JsonBackend
        _backend = JsonBackend(STATE_FILE, PRESETS_FILE, USERS_FILE, SETTINGS_FILE)

    return _backend
