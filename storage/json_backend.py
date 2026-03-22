# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import hashlib
import json
import logging
import os
import tempfile
import threading

from storage.base import StorageBackend

logger = logging.getLogger("llamaman")


def _atomic_write_json(path: str, data: dict) -> None:
    """Write JSON to a file atomically (write tmp + rename).

    If the process crashes mid-write, the original file is untouched.
    """
    dir_name = os.path.dirname(path) or "."
    try:
        fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
        except BaseException:
            os.unlink(tmp)
            raise
    except Exception as e:
        logger.warning("Failed to write %s: %s", path, e)


class JsonBackend(StorageBackend):
    """Stores data in local JSON files. Zero dependencies, works out of the box."""

    def __init__(self, state_file: str, presets_file: str, users_file: str,
                 settings_file: str, api_keys_file: str = ""):
        self._state_file = state_file
        self._presets_file = presets_file
        self._users_file = users_file
        self._settings_file = settings_file
        self._api_keys_file = api_keys_file or os.path.join(
            os.path.dirname(settings_file), "api_keys.json"
        )
        self._state_lock = threading.Lock()

    # -- State helpers --

    def _read_state(self) -> dict:
        try:
            with open(self._state_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    # -- State (atomic read/write) --

    def save_state(self, instances: list[dict], downloads: list[dict]) -> None:
        with self._state_lock:
            state = {"instances": instances, "downloads": downloads}
            _atomic_write_json(self._state_file, state)

    def load_instances(self) -> list[dict]:
        return self._read_state().get("instances", [])

    def load_downloads(self) -> list[dict]:
        return self._read_state().get("downloads", [])

    # -- Presets --

    def get_all_presets(self) -> dict[str, dict]:
        try:
            with open(self._presets_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def get_preset(self, model_path: str) -> dict | None:
        return self.get_all_presets().get(model_path)

    def save_preset(self, model_path: str, data: dict) -> None:
        presets = self.get_all_presets()
        presets[model_path] = data
        _atomic_write_json(self._presets_file, presets)

    def delete_preset(self, model_path: str) -> None:
        presets = self.get_all_presets()
        if model_path in presets:
            del presets[model_path]
            _atomic_write_json(self._presets_file, presets)

    # -- Auth --

    def _read_users(self) -> dict[str, dict]:
        try:
            with open(self._users_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write_users(self, users: dict[str, dict]) -> None:
        _atomic_write_json(self._users_file, users)

    def get_user(self, username: str) -> dict | None:
        return self._read_users().get(username)

    def save_user(self, username: str, password_hash: str) -> None:
        users = self._read_users()
        users[username] = {"username": username, "password_hash": password_hash}
        self._write_users(users)

    def user_count(self) -> int:
        return len(self._read_users())

    # -- Settings --

    def get_settings(self) -> dict:
        try:
            with open(self._settings_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_settings(self, settings: dict) -> None:
        _atomic_write_json(self._settings_file, settings)

    # -- API Keys --

    @staticmethod
    def _hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def _read_api_keys(self) -> list[dict]:
        try:
            with open(self._api_keys_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _write_api_keys(self, keys: list[dict]) -> None:
        _atomic_write_json(self._api_keys_file, keys)

    def get_api_keys(self) -> list[dict]:
        return self._read_api_keys()

    def save_api_key(self, key_entry: dict) -> None:
        keys = self._read_api_keys()
        # Update existing or append
        for i, k in enumerate(keys):
            if k["id"] == key_entry["id"]:
                keys[i] = key_entry
                self._write_api_keys(keys)
                return
        keys.append(key_entry)
        self._write_api_keys(keys)

    def delete_api_key(self, key_id: str) -> None:
        keys = self._read_api_keys()
        keys = [k for k in keys if k["id"] != key_id]
        self._write_api_keys(keys)

    def verify_api_key(self, raw_key: str) -> bool:
        hashed = self._hash_key(raw_key)
        for k in self._read_api_keys():
            if k.get("key_hash") == hashed:
                return True
        return False
