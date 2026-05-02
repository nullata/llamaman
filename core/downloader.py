# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import json
import os
import re
import sys
import time

import requests

MULTIPART_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)

repo_id = os.environ.get("HF_REPO_ID", "")
local_dir = os.environ.get("HF_LOCAL_DIR", "")
filename = os.environ.get("HF_FILENAME", "").strip()
token = os.environ.get("HF_TOKEN", "").strip() or None
speed_limit = int(os.environ.get("HF_SPEED_LIMIT", "0"))        # effective at launch (for log)
per_model_limit = int(os.environ.get("HF_PER_MODEL_SPEED_LIMIT", "0"))  # per-model fallback

_SUBPROCESS_SETTINGS_FILE = os.path.join(
    os.environ.get("DATA_DIR", "/data"), "subprocess_settings.json"
)

CHUNK_SIZE = 64 * 1024  # 64 KB
HF_API = "https://huggingface.co"


def _read_global_speed_limit() -> int:
    """Read the current global speed limit from the subprocess-facing settings
    snapshot (bytes/sec). Maintained by the main process regardless of storage
    backend, so live UI updates are picked up within 1 second by running
    downloads on both JSON and MariaDB backends.

    Returns:
        > 0  - global limit active
          0  - global limit explicitly disabled
         -1  - snapshot missing or unreadable (first run before any save,
               or the main process hasn't written it yet)
    """
    try:
        with open(_SUBPROCESS_SETTINGS_FILE) as f:
            s = json.load(f)
        mbps = float(s.get("global_speed_limit_mbps", 0) or 0)
        return int(mbps * 1_000_000 / 8) if mbps > 0 else 0
    except FileNotFoundError:
        return -1
    except Exception:
        return -1


def _effective_limit() -> int:
    """Return the current effective speed limit in bytes/sec (0 = unlimited).

    Priority:
      1. Global setting from settings.json if > 0
      2. Per-model limit if global is explicitly 0
      3. Launch-time effective limit (speed_limit) if settings.json is unreadable
    """
    g = _read_global_speed_limit()
    if g > 0:
        return g
    if g == 0:
        return per_model_limit
    return speed_limit  # settings.json unreadable >> launch-time fallback


def _headers(tok=None):
    h = {"User-Agent": "llamaman/1.0"}
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _fmt_size(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _throttled_iter(resp):
    """Yield chunks, throttling to the current effective speed limit.

    Re-reads the global speed limit control file every second so that changes
    made in the UI take effect immediately on running downloads.
    Token bucket: when the limit changes, the bucket resets to the new rate.
    """
    lim = _effective_limit()
    last_check = time.monotonic()
    last = time.monotonic()
    bucket = float(lim) if lim > 0 else 0.0

    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
        if not chunk:
            continue
        now = time.monotonic()

        # Re-read limit every second
        if now - last_check >= 1.0:
            new_lim = _effective_limit()
            if new_lim != lim:
                lim = new_lim
                bucket = float(lim) if lim > 0 else 0.0
            last_check = now

        if lim > 0:
            nbytes = len(chunk)
            bucket += (now - last) * lim
            if bucket > lim:
                bucket = float(lim)
            last = now
            bucket -= nbytes
            if bucket < 0:
                time.sleep(-bucket / lim)
                bucket = 0.0
                last = time.monotonic()
        else:
            last = now  # keep last current so bucket math is correct if limit is set later

        yield chunk


def list_repo_files(rid=None, tok=None):
    """Fetch file list from HuggingFace API.

    Defaults to module-level repo_id/token (script use); pass args when
    imported from another module.
    """
    rid = rid or repo_id
    tok = tok if tok is not None else token
    url = f"{HF_API}/api/models/{rid}"
    r = requests.get(url, headers=_headers(tok), timeout=30)
    if r.status_code in (401, 403):
        raise RuntimeError(f"Authentication failed ({r.status_code}). Check your HF token.")
    if r.status_code == 404:
        raise RuntimeError(f"Repository not found: {rid}")
    r.raise_for_status()
    siblings = r.json().get("siblings", [])
    return [{"name": s["rfilename"], "size": s.get("size") or s.get("lfs", {}).get("size")} for s in siblings]


def download_file(fname, file_num=None, total_files=None):
    """Download a single file with resume support and rate limiting."""
    local_path = os.path.join(local_dir, fname)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    url = f"{HF_API}/{repo_id}/resolve/main/{fname}"

    prefix = f"[{file_num}/{total_files}] " if file_num else ""
    print(f"{prefix}{fname}", flush=True)

    existing = 0
    if os.path.isfile(local_path):
        existing = os.path.getsize(local_path)

    hdrs = _headers(token)
    if existing > 0:
        hdrs["Range"] = f"bytes={existing}-"

    r = requests.get(url, headers=hdrs, stream=True, timeout=30, allow_redirects=True)

    if r.status_code == 416:
        print(f"  Already complete ({_fmt_size(existing)})", flush=True)
        return

    if existing > 0 and r.status_code == 200:
        existing = 0

    if r.status_code not in (200, 206):
        raise RuntimeError(f"HTTP {r.status_code} downloading {fname}: {r.text[:200]}")

    content_length = r.headers.get("Content-Length")
    total = int(content_length) + existing if content_length else None

    if existing > 0 and r.status_code == 206:
        print(f"  Resuming from {_fmt_size(existing)}", flush=True)
        mode = "ab"
    else:
        mode = "wb"
        existing = 0

    downloaded = existing
    last_print = time.monotonic()
    start_time = last_print

    with open(local_path, mode) as f:
        for chunk in _throttled_iter(r):
            f.write(chunk)
            downloaded += len(chunk)
            now = time.monotonic()
            if now - last_print >= 1.0:
                elapsed = now - start_time
                speed = (downloaded - existing) / elapsed if elapsed > 0 else 0
                pct = f"  {downloaded * 100 / total:.0f}%" if total else ""
                print(f"  {_fmt_size(downloaded)}{f' / {_fmt_size(total)}' if total else ''}{pct}  {_fmt_size(speed)}/s", flush=True)
                last_print = now

    elapsed = time.monotonic() - start_time
    speed = (downloaded - existing) / elapsed if elapsed > 0 else 0
    print(f"  {_fmt_size(downloaded)}{f' / {_fmt_size(total)}' if total else ''}  100%  {_fmt_size(speed)}/s  done", flush=True)


def resolve_filename(requested: str, repo_files: list[dict], rid: str = "") -> list[dict]:
    """Resolve a user-supplied filename to one or more repo-relative paths.

    Handles two UI gotchas: HF's file browser shows only basenames even when
    the file is nested in a subfolder, and multipart GGUFs ("...-00001-of-00008.gguf")
    are useless without their siblings. Matches by exact path first, then by
    basename; if the resolved name is a multipart shard, expands to every shard
    in that group so the user only has to pick one.
    """
    rid = rid or repo_id
    requested_norm = requested.strip().lstrip("/")
    by_path = {f["name"]: f for f in repo_files}

    match = by_path.get(requested_norm)
    if match is None:
        basename = os.path.basename(requested_norm)
        basename_matches = [f for f in repo_files if os.path.basename(f["name"]) == basename]
        if len(basename_matches) == 1:
            match = basename_matches[0]
        elif len(basename_matches) > 1:
            paths = ", ".join(f["name"] for f in basename_matches)
            raise RuntimeError(f"Ambiguous filename '{basename}' matches multiple paths: {paths}")
        else:
            raise RuntimeError(f"File '{requested}' not found in {rid}")

    m = MULTIPART_RE.match(match["name"])
    if not m:
        return [match]

    stem, _, total_str = m.group(1), m.group(2), m.group(3)
    total = int(total_str)
    shards = []
    for i in range(1, total + 1):
        shard_name = f"{stem}-{i:05d}-of-{total_str}.gguf"
        shard = by_path.get(shard_name)
        if shard is None:
            raise RuntimeError(f"Missing multipart shard in repo: {shard_name}")
        shards.append(shard)
    return shards


def main():
    print(f"Starting download: {repo_id}", flush=True)
    if filename:
        print(f"Single file: {filename}", flush=True)
    print(f"Destination: {local_dir}", flush=True)
    if speed_limit:
        print(f"Speed limit: {speed_limit * 8 / 1_000_000:.0f} Mbps ({speed_limit / 1024 / 1024:.1f} MB/s)", flush=True)
    print("", flush=True)

    try:
        if filename:
            print("Fetching file list...", flush=True)
            all_files = list_repo_files()
            if not all_files:
                raise RuntimeError(f"No files found in {repo_id}")
            targets = resolve_filename(filename, all_files)
            if len(targets) == 1 and targets[0]["name"] == filename.strip().lstrip("/"):
                download_file(targets[0]["name"])
            else:
                print(f"Resolved to {len(targets)} file(s)", flush=True)
                print("", flush=True)
                for i, finfo in enumerate(targets, 1):
                    download_file(finfo["name"], file_num=i, total_files=len(targets))
        else:
            print("Fetching file list...", flush=True)
            files = list_repo_files()
            if not files:
                raise RuntimeError(f"No files found in {repo_id}")
            print(f"Found {len(files)} files", flush=True)
            print("", flush=True)
            for i, finfo in enumerate(files, 1):
                download_file(finfo["name"], file_num=i, total_files=len(files))

        print(f"\nCompleted: {local_dir}", flush=True)
        sys.exit(0)

    except Exception as e:
        print(f"\nError: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
