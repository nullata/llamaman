# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import hashlib
import json
import logging
import os
import tempfile
import threading
from copy import deepcopy
from datetime import datetime, timezone

from storage.base import StorageBackend

logger = logging.getLogger("llamaman")


def _merge_dicts(base: dict, patch: dict) -> dict:
    merged = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


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
                 settings_file: str, api_keys_file: str = "",
                 recordings_dir: str = ""):
        self._state_file = state_file
        self._presets_file = presets_file
        self._users_file = users_file
        self._settings_file = settings_file
        self._api_keys_file = api_keys_file or os.path.join(
            os.path.dirname(settings_file), "api_keys.json"
        )
        self._recordings_dir = recordings_dir or os.path.join(
            os.path.dirname(settings_file), "request_log"
        )
        self._state_lock = threading.Lock()
        self._settings_lock = threading.Lock()
        self._request_log_lock = threading.Lock()

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
        with self._settings_lock:
            try:
                with open(self._settings_file, "r") as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return {}

    def save_settings(self, settings: dict) -> None:
        with self._settings_lock:
            _atomic_write_json(self._settings_file, settings)

    def merge_settings(self, patch: dict) -> dict:
        with self._settings_lock:
            try:
                with open(self._settings_file, "r") as f:
                    current = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                current = {}
            merged = _merge_dicts(current, patch)
            _atomic_write_json(self._settings_file, merged)
            return merged

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

    # -- Request Log --

    def append_request_log(self, record: dict, mode: str) -> None:
        conv_id = record.get("conversation_id")
        created_at = record.get("created_at")
        if not conv_id or not created_at:
            logger.warning("request_log append skipped: missing conversation_id or created_at")
            return

        date = datetime.fromtimestamp(int(created_at) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        date_dir = os.path.join(self._recordings_dir, date)

        if mode == "per_request":
            # Unique filename per record; each turn is its own single-line file.
            filename = f"{int(created_at)}_{conv_id[:8]}.jsonl"
        else:
            # per_conversation: append to the conversation's file.
            filename = f"{conv_id}.jsonl"

        path = os.path.join(date_dir, filename)
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"

        with self._request_log_lock:
            try:
                os.makedirs(date_dir, exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line)
            except Exception as e:
                logger.warning("request_log append failed (%s): %s", path, e)

    def _iter_request_log_records(self):
        """Yield every record dict across all date dirs, with (date_dir, filename, line_no)."""
        if not os.path.isdir(self._recordings_dir):
            return
        for date in sorted(os.listdir(self._recordings_dir), reverse=True):
            date_dir = os.path.join(self._recordings_dir, date)
            if not os.path.isdir(date_dir):
                continue
            for fn in os.listdir(date_dir):
                if not fn.endswith(".jsonl"):
                    continue
                full = os.path.join(date_dir, fn)
                try:
                    with open(full, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                yield json.loads(line), full
                            except (json.JSONDecodeError, ValueError):
                                continue
                except OSError:
                    continue

    def list_conversations(self, limit: int = 100) -> list[dict]:
        rollup: dict[str, dict] = {}
        for rec, _ in self._iter_request_log_records():
            cid = rec.get("conversation_id")
            if not cid:
                continue
            created = rec.get("created_at") or 0
            entry = rollup.get(cid)
            if entry is None:
                entry = {
                    "conversation_id": cid,
                    "model": rec.get("model", ""),
                    "first_seen_at": created,
                    "last_seen_at": created,
                    "turn_count": 0,
                    "title": "",
                }
                rollup[cid] = entry
            entry["turn_count"] += 1
            if created < entry["first_seen_at"] or not entry["first_seen_at"]:
                entry["first_seen_at"] = created
            if created > entry["last_seen_at"]:
                entry["last_seen_at"] = created
            # Title = first user message from the first (earliest) request body
            if not entry["title"]:
                try:
                    req = json.loads(rec.get("request_body") or "{}")
                    msgs = req.get("messages") or []
                    for m in msgs:
                        if isinstance(m, dict) and m.get("role") == "user":
                            content = m.get("content", "")
                            if isinstance(content, str) and content:
                                entry["title"] = content[:120]
                                break
                    if not entry["title"]:
                        prompt = req.get("prompt")
                        if isinstance(prompt, str):
                            entry["title"] = prompt[:120]
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
        out = sorted(rollup.values(), key=lambda e: e["last_seen_at"], reverse=True)
        return out[:limit]

    def get_conversation_turns(self, conversation_id: str) -> list[dict]:
        turns = []
        for rec, _ in self._iter_request_log_records():
            if rec.get("conversation_id") == conversation_id:
                turns.append(rec)
        turns.sort(key=lambda r: r.get("created_at") or 0)
        return turns

    def prune_request_log(self, older_than_ms: int) -> int:
        if not os.path.isdir(self._recordings_dir):
            return 0
        pruned = 0
        cutoff_date = datetime.fromtimestamp(older_than_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        for date in os.listdir(self._recordings_dir):
            date_dir = os.path.join(self._recordings_dir, date)
            if not os.path.isdir(date_dir):
                continue
            if date >= cutoff_date:
                continue
            # Whole day is older than cutoff — nuke it
            try:
                for fn in os.listdir(date_dir):
                    full = os.path.join(date_dir, fn)
                    # Rough count: one turn per line
                    try:
                        with open(full, "r", encoding="utf-8") as f:
                            pruned += sum(1 for _ in f)
                    except OSError:
                        pass
                    try:
                        os.unlink(full)
                    except OSError:
                        pass
                os.rmdir(date_dir)
            except OSError as e:
                logger.warning("failed to prune %s: %s", date_dir, e)
        return pruned
