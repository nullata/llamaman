# <img src="static/images/logo.svg" alt="logo" width="24"> LlamaMan

<p align="center">
  <img src="docs/llamaman.jpg" alt="LlamaMan" width="400">
</p>

A browser-based UI for launching, monitoring, and managing multiple [llama.cpp](https://github.com/ggerganov/llama.cpp) server instances from inside a Docker container. Includes an Ollama-compatible API proxy so it works as a drop-in replacement for Ollama with [Open WebUI](https://github.com/open-webui/open-webui).

## Features

- **Model library** - scans `/models` for GGUF files, shows quant type and file size
- **One-click launch** - configure GPU layers, context size, threads, multi-GPU, extra args
- **Preset configs** - save/load per-model launch settings
- **Download manager** - pull models from HuggingFace with speed throttling
- **Instance management** - stop, restart, remove, view live-streamed logs
- **GPU VRAM indicator** - per-GPU usage bars via nvidia-smi or rocm-smi
- **Idle timeout** - auto-sleep instances after configurable idle period, wake on next request
- **Ollama-compatible proxy** - OpenWebUI discovers models and auto-starts servers on demand
- **Authentication** - user accounts with session login, API key management with bearer tokens
- **Require auth toggle** - enforce bearer token authentication on all endpoints (including model loading) or leave model endpoints open
- **Persistent state** - instance history and configs survive container restarts
- **Storage backends** - JSON files (default) or MariaDB/MySQL via SQLAlchemy

## Requirements

- Docker with **one** of:
  - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for CUDA / NVIDIA GPUs)
  - [ROCm-compatible setup](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) (for AMD GPUs) - **experimental, not tested**
- A supported GPU (llama.cpp can offload to CPU/RAM when VRAM is insufficient)

## Quick Start

**NVIDIA (CUDA):**

```bash
docker compose up --build
```

**AMD (ROCm)** - experimental, not tested:

```bash
docker compose --profile rocm up --build llamaman-rocm
```

- **Management UI**: http://localhost:5000
- **Llamaman proxy** (Ollama-compatible API): http://localhost:42069
- **llama-server public instance ports**: 8000-8020

On first launch, visit the UI to create an admin account via `/setup`.

## Authentication

LlamaMan has a built-in auth system with two layers:

### User accounts (session-based)

On first launch, `/setup` lets you create an admin account. After that, all browser access requires login. Session cookies authenticate UI requests.

### API keys (bearer tokens)

Create API keys in the **API Keys** section of the UI. External clients (OpenWebUI, scripts, etc.) authenticate with:

```
Authorization: Bearer llm-xxxxxxxxxx
```

### Require authentication toggle

The **"Require authentication for all endpoints"** toggle (on by default) controls whether model-serving endpoints require a bearer token:

| Toggle | Model endpoints (`/api/chat`, `/v1/chat/completions`, etc.) | Management endpoints (`/api/instances`, etc.) | Per-instance proxy ports |
|--------|--------------------------------------------------------------|-----------------------------------------------|--------------------------|
| **ON** (default) | Bearer token required | Bearer token or session required | Bearer token required |
| **OFF** | Open (no auth) | Bearer token or session required | Open (no auth) |

When the toggle is **ON**, all three port surfaces are protected:
- **Port 5000** (management UI + API) - Flask `before_request` hook
- **Port 42069** (Ollama-compatible proxy) - same Flask app, same hook
- **Ports 8000-8020** (per-instance proxies) - WSGI-level auth check

### OpenWebUI with authentication

When `require_auth` is on, configure OpenWebUI to send a valid API key:

```yaml
open-webui:
  environment:
    - OLLAMA_BASE_URL=http://llamaman:42069
    - OPENAI_API_BASE_URLS=http://llamaman:42069/v1
    - OPENAI_API_KEYS=llm-your-api-key-here
```

## Models

Place models inside the `models/` volume:

- **GGUF files**: any `.gguf` file (recommended - llama.cpp native format)
- **HuggingFace repos**: directories containing `config.json`

Or use the **Download** button in the UI to pull from HuggingFace.

## Launching Instances

1. Select a model from the sidebar
2. Configure launch settings (GPU layers, context size, idle timeout, etc.)
3. Click **Launch** - the instance appears with a status badge
4. Optionally click **Save Preset** to remember settings for that model

Each instance exposes an OpenAI-compatible API on its assigned port.

### Layer autodetection

When you select a GGUF model, LlamaMan reads the file's metadata to detect the total number of layers (block count). This is displayed next to the **GPU Layers** input so you can see exactly how many layers are available to offload (e.g. `/ 32`). Set GPU Layers to `-1` to offload all layers to GPU.

### Launch settings reference

| Setting | Default | Description |
|---|---|---|
| **GPU Layers** | `-1` | Number of layers to offload to GPU. `-1` = all layers, `0` = CPU only. Total layers are autodetected from the GGUF file. |
| **Context Size** | `4096` | Maximum context window in tokens (`--ctx-size`). |
| **Parallel** | `1` | Number of parallel sequences the llama-server can process simultaneously (`--parallel`). Controls KV cache slot allocation inside the server itself. |
| **Idle Timeout min** | `0` | Minutes of inactivity before the server is stopped to free VRAM. `0` = disabled. See [Idle Timeout](#idle-timeout). |
| **Max Concurrent** | `0` | Maximum number of inference requests allowed in-flight at once. `0` = unlimited. When set, incoming requests are queued and gated by a semaphore. |
| **Max Queue Depth** | `200` | Maximum number of requests that can wait in the queue when `Max Concurrent` is active. Requests beyond this limit are rejected with HTTP 429. |
| **Share Queue** | off | When enabled, multiple proxy-managed instances of the **same model** share a single request queue. Incoming requests are distributed across instances as slots become available, providing simple load balancing. |
| **Embedding Model** | off | Marks the instance as an embedding model. Embedding instances are **excluded** from the `LLAMAMAN_MAX_MODELS` count and will never be evicted by the proxy's LRU policy. |
| **GPU Devices** | `0` | Comma-separated GPU indices for multi-GPU setups (e.g. `0,1`). |
| **Extra Args** | _(empty)_ | Additional flags passed directly to llama-server (e.g. `--flash-attn`). |

### Concurrency and queueing

When **Max Concurrent** is set to a value greater than 0, LlamaMan places a concurrency gate in front of the instance. Requests that exceed the limit are held in a FIFO queue (up to **Max Queue Depth**). If the queue is also full, new requests are rejected with HTTP 429.

The gate tracks active and queued request counts, which are visible in the instance list via the API.

**Parallel vs Max Concurrent:** `Parallel` controls how many sequences the llama-server processes internally (KV cache slots). `Max Concurrent` is an external gate that limits how many requests LlamaMan forwards to the server at once. You can use both together — for example, `Parallel=4` with `Max Concurrent=4` ensures the server always has enough KV slots for the requests it receives.

## Idle Timeout

Set **Idle Timeout min** in the launch form (0 = disabled). When enabled:

- The manager proxies the instance port (transparent to clients)
- After N minutes of no requests, the llama-server is stopped to free VRAM
- On the next request, the server auto-relaunches with the same config
- Client sees the same port/API with just a cold-start delay

For instances managed by the llamaman proxy (OpenWebUI), use the `LLAMAMAN_IDLE_TIMEOUT` env var instead.

## Cleanup Settings

The UI provides automatic cleanup of stale records under **Settings**:

- **Auto-clean completed/failed downloads** - removes download records older than a configurable number of hours (default: 24). Only affects completed, failed, or cancelled downloads — active downloads are never touched.
- **Auto-clean stopped instances** - removes stopped instance records older than a configurable number of hours (default: 24). Only affects stopped instances — running instances are never removed.

Cleanup runs periodically in the background. These settings only remove records from the UI/state — they do not delete model files or stop running processes.

## OpenWebUI Integration (llamaman proxy)

The llamaman proxy exposes an Ollama-compatible API on port **42069** (configurable). Point OpenWebUI at it:

```yaml
open-webui:
  environment:
    - OLLAMA_BASE_URL=http://llamaman:42069
```

**How it works:**

1. OpenWebUI calls `/api/tags` -> LlamaMan returns all available GGUF models
2. User selects a model in OpenWebUI -> `/api/chat` request arrives
3. LlamaMan auto-launches a llama-server (using saved preset or defaults)
4. Waits for healthy, then proxies the request with format translation
5. When `LLAMAMAN_MAX_MODELS` limit is reached, the least-recently-used model is evicted

Supported Ollama endpoints: `/api/tags`, `/api/chat`, `/api/generate`, `/api/show`, `/api/version`, `/api/ps`

Also supports OpenAI-compatible endpoints with auto-start: `/v1/models`, `/v1/chat/completions`

### Model eviction policy

The `LLAMAMAN_MAX_MODELS` limit controls how many **chat** models the proxy will keep loaded simultaneously. When a new model is requested and the limit is reached, the least-recently-used chat model is evicted.

Key details:

- **All running instances count toward the limit** - both manually launched instances (from the LlamaMan UI) and proxy-managed ones. If you manually launch 2 models and `LLAMAMAN_MAX_MODELS=1`, the proxy sees you're already over the limit.
- **Proxy auto-launch only evicts proxy-managed instances.** Normal incoming inference requests will never kill manually launched instances.
- **Admin UI launches are configurable.** A dashboard toggle controls whether launching from the admin UI should evict the least-recently-used non-embedding instance to stay within the cap, or instead prompt you and allow launching beyond the cap.
- **Embedding models are excluded.** Instances marked as **Embedding Model** do not count toward the limit and are never evicted. This lets you keep an embedding model loaded permanently alongside your chat models.
- **`LLAMAMAN_MAX_MODELS=0` (default) disables eviction entirely.** The proxy will launch models on demand without ever stopping existing ones.

## Storage Backends

### JSON (default)

Zero-config. Stores data in JSON files under `DATA_DIR` (`/data`):
- `state.json` - instances and downloads
- `presets.json` - per-model launch presets
- `users.json` - user accounts
- `settings.json` - global settings
- `api_keys.json` - API key hashes

Instance and download logs are written to `LOGS_DIR` (`/tmp/llama-logs`), which is separate from persistent data.

### MariaDB / MySQL

Set `DATABASE_URL` to enable:

```yaml
environment:
  - DATABASE_URL=mysql+pymysql://user:password@host:3306/llamaman
```

Tables are auto-created on first connection. Requires `sqlalchemy` and `pymysql` (included in requirements).

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODELS_DIR` | `/models` | Directory scanned for model files |
| `DATA_DIR` | `/data` | Directory for persistent config/state (JSON files) |
| `LOGS_DIR` | `/tmp/llama-logs` | Directory for instance and download logs |
| `PORT_RANGE_START` | `8000` | Start of public llama-server/proxy port pool |
| `PORT_RANGE_END` | `8020` | End of public llama-server/proxy port pool |
| `INTERNAL_PORT_RANGE_START` | `9000` | Start of internal llama-server port pool used when proxy mode is enabled |
| `INTERNAL_PORT_RANGE_END` | `9020` | End of internal llama-server port pool used when proxy mode is enabled |
| `LLAMAMAN_PROXY_PORT` | `42069` | Port for the Ollama-compatible proxy |
| `LLAMAMAN_MAX_MODELS` | `0` | Max concurrent **chat** models via the proxy (LRU eviction, 0 = unlimited) |
| `LLAMAMAN_IDLE_TIMEOUT` | `0` | Idle timeout in minutes for proxy-managed instances (0 = disabled) |
| `SECRET_KEY` | _(auto)_ | Flask session secret. Auto-derived from machine-id if unset. Set this for multi-replica deployments. |
| `DATABASE_URL` | _(unset)_ | MariaDB/MySQL connection string. Unset = use JSON files. |
| `HEALTH_CHECK_TIMEOUT` | `3` | Timeout in seconds for instance health checks |
| `MODEL_LOAD_TIMEOUT` | `300` | Seconds to wait for a model to become healthy during launch/relaunch. Increase for very large models. |
| `REQUEST_TIMEOUT` | `300` | Timeout in seconds for upstream requests to llama-server and gate acquire waits. Increase if requests are being cut off under heavy concurrency. |

## REST API

All endpoints return and accept JSON.

**Authentication:** Management endpoints require either a session cookie (from browser login) or an `Authorization: Bearer <key>` header. When `require_auth` is enabled (default), model-serving endpoints also require a bearer token.

### Authentication

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/login` | Login page |
| `POST` | `/login` | Authenticate (`username`, `password` form data) |
| `GET` | `/setup` | First-run setup page |
| `POST` | `/setup` | Create first user account |
| `GET` | `/logout` | End session |

### API Keys

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/api-keys` | List all API keys (hashes stripped) |
| `POST` | `/api/api-keys` | Create a new API key (`{"name": "..."}`) |
| `DELETE` | `/api/api-keys/<id>` | Revoke an API key |

### Instances

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/instances` | List all instances |
| `POST` | `/api/instances` | Launch a new instance |
| `GET` | `/api/instances/<id>` | Get a single instance |
| `DELETE` | `/api/instances/<id>` | Stop and remove an instance |
| `POST` | `/api/instances/<id>/restart` | Restart a stopped/sleeping instance |
| `DELETE` | `/api/instances/<id>/remove` | Remove a stopped instance from the list |
| `GET` | `/api/instances/<id>/logs` | Last N log lines |
| `GET` | `/api/instances/<id>/logs/stream` | SSE live log tail |
| `GET` | `/api/next-port` | Get next available port from the pool |

**Launch body** (`POST /api/instances`):
```json
{
  "model_path": "/models/my-model.gguf",
  "port": 8000,
  "n_gpu_layers": -1,
  "ctx_size": 4096,
  "threads": null,
  "parallel": null,
  "extra_args": "--flash-attn",
  "gpu_devices": "0",
  "idle_timeout_min": 0,
  "max_concurrent": 0,
  "max_queue_depth": 200,
  "share_queue": false
}
```

### Downloads

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/downloads` | List all downloads |
| `POST` | `/api/downloads` | Start a new download |
| `GET` | `/api/downloads/<id>` | Get a single download |
| `DELETE` | `/api/downloads/<id>` | Cancel an active download |
| `DELETE` | `/api/downloads/<id>/remove` | Remove a completed/failed entry |
| `GET` | `/api/downloads/<id>/logs` | Download log output |
| `GET` | `/api/downloads/<id>/logs/stream` | SSE live log tail |

**Download body** (`POST /api/downloads`):
```json
{
  "repo_id": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
  "filename": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
  "hf_token": "hf_...",
  "speed_limit_mbps": 0
}
```

Leave `filename` blank to download the full repository.

### Models

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/models` | List discovered models in `MODELS_DIR` |
| `POST` | `/api/models/delete` | Delete a model from disk (`{"path": "/models/..."}`) |
| `GET` | `/api/model-layers?path=<path>` | Read layer count from GGUF metadata |
| `GET` | `/api/disk-space` | Free/used space on the models volume |

### Presets

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/presets` | List all saved presets |
| `GET` | `/api/presets/<model_path>` | Get preset for a model |
| `PUT` | `/api/presets/<model_path>` | Save/update a preset |
| `DELETE` | `/api/presets/<model_path>` | Delete a preset |

### Settings

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/settings` | Get global settings |
| `POST` | `/api/settings` | Save global settings |

**Settings body** (`POST /api/settings`):
```json
{
  "require_auth": true,
  "cleanup": {
    "downloads_enabled": true,
    "downloads_max_age_hours": 24,
    "downloads_last_run_at": 1710000000,
    "instances_enabled": false,
    "instances_max_age_hours": 48,
    "instances_last_run_at": 1710000000
  }
}
```

### System

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/system-info` | CPU usage, core count, RAM usage |
| `GET` | `/api/gpu-info` | Per-GPU VRAM usage via nvidia-smi |
| `GET` | `/health` | Health check (`{"status": "ok"}`) - always open, no auth required |

### Ollama-compatible (llamaman)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/tags` | List available models (Ollama format) |
| `GET` | `/api/version` | Version info |
| `POST` | `/api/show` | Model metadata |
| `GET` | `/api/ps` | Running models |
| `POST` | `/api/chat` | Chat completion (auto-starts model) |
| `POST` | `/api/generate` | Text generation (auto-starts model) |
| `GET` | `/v1/models` | List models (OpenAI format) |
| `POST` | `/v1/chat/completions` | Chat completion (OpenAI format, auto-starts model) |

## Troubleshooting

| Symptom | Fix |
|---|---|
| _"llama-server binary not found"_ | The base image must be `ghcr.io/ggml-org/llama.cpp:server-cuda` (or `server-rocm` for AMD). Rebuild with `--no-cache`. |
| Instance stuck on **starting** | Check logs via the Logs button. Common causes: OOM, model path typo, corrupt GGUF. |
| No GPU / CUDA error | Ensure the NVIDIA Container Toolkit is installed and `docker run --gpus all` works. |
| No GPU / ROCm error | Ensure `/dev/kfd` and `/dev/dri` exist on the host and your user is in the `video`/`render` groups. The ROCm image is experimental and not tested. |
| Port conflict | The form auto-suggests an unused port; adjust if needed. |
| Model not showing in OpenWebUI | Ensure `OLLAMA_BASE_URL` points to `http://llamaman:42069`. Check `/api/tags` returns models. |
| OpenWebUI gets 401 errors | `require_auth` is on (default). Create an API key in the UI and set `OPENAI_API_KEYS` in OpenWebUI's environment. |
| _"API key required"_ on all requests | Either create an API key, or turn off the "Require authentication" toggle in the API Keys section. |

## License

LlamaMan is licensed under the [Elastic License 2.0](LICENSE). You may use, copy, distribute, and modify the software, subject to the following limitations:

- You may **not** provide it as a hosted or managed service
- You may **not** remove or circumvent license key functionality
- You may **not** alter or remove licensing, copyright, or other notices
