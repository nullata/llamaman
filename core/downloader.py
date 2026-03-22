# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import os
import sys
import time

import requests

repo_id = os.environ["HF_REPO_ID"]
local_dir = os.environ["HF_LOCAL_DIR"]
filename = os.environ.get("HF_FILENAME", "").strip()
token = os.environ.get("HF_TOKEN", "").strip() or None
speed_limit = int(os.environ.get("HF_SPEED_LIMIT", "0"))  # bytes/sec, 0=unlimited

CHUNK_SIZE = 64 * 1024  # 64 KB
HF_API = "https://huggingface.co"


def _headers():
    h = {"User-Agent": "llamaman/1.0"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _fmt_size(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _throttled_iter(resp):
    """Yield chunks, sleeping as needed to enforce speed_limit."""
    if speed_limit <= 0:
        yield from resp.iter_content(chunk_size=CHUNK_SIZE)
        return
    bucket = float(speed_limit)
    last = time.monotonic()
    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
        if not chunk:
            continue
        nbytes = len(chunk)
        now = time.monotonic()
        bucket += (now - last) * speed_limit
        if bucket > speed_limit:
            bucket = float(speed_limit)
        last = now
        bucket -= nbytes
        if bucket < 0:
            time.sleep(-bucket / speed_limit)
            bucket = 0.0
            last = time.monotonic()
        yield chunk


def list_repo_files():
    """Fetch file list from HuggingFace API."""
    url = f"{HF_API}/api/models/{repo_id}"
    r = requests.get(url, headers=_headers(), timeout=30)
    if r.status_code in (401, 403):
        raise RuntimeError(f"Authentication failed ({r.status_code}). Check your HF token.")
    if r.status_code == 404:
        raise RuntimeError(f"Repository not found: {repo_id}")
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

    hdrs = _headers()
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


print(f"Starting download: {repo_id}", flush=True)
if filename:
    print(f"Single file: {filename}", flush=True)
print(f"Destination: {local_dir}", flush=True)
if speed_limit:
    print(f"Speed limit: {speed_limit * 8 / 1_000_000:.0f} Mbps ({speed_limit / 1024 / 1024:.1f} MB/s)", flush=True)
print("", flush=True)

try:
    if filename:
        download_file(filename)
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
