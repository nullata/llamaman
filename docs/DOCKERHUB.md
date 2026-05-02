# LlamaMan

![LlamaMan](https://raw.githubusercontent.com/nullata/llamaman/main/docs/llamaman-0.9.6.jpg)

A browser-based UI for launching, monitoring, and managing multiple [llama.cpp](https://github.com/ggerganov/llama.cpp) server instances from inside a Docker container. Includes an Ollama-compatible API proxy so it works as a drop-in replacement for Ollama with [Open WebUI](https://github.com/open-webui/open-webui).

## Features

- **Model library** - scans `/models` for GGUF files, shows quant type and file size
- **One-click launch** - configure GPU layers, context size, threads, multi-GPU, extra args
- **Preset configs** - save/load per-model launch settings
- **Download manager** - pull models from HuggingFace with speed throttling and auto-retry on failure
- **Model backup and restore** - export model metadata and presets to JSON, restore on any instance with downloads queued automatically for missing models
- **Instance management** - stop, restart, remove, view live-streamed logs
- **Container resource monitoring** - live CPU%, core quota, RAM usage with thin progress bars, and GPU assignment per running instance card
- **GPU VRAM indicator** - per-GPU VRAM and utilization, queried natively (no running instance required)
- **Idle timeout** - auto-sleep instances after configurable idle period, wake on next request
- **Ollama-compatible proxy** - OpenWebUI discovers models and auto-starts servers on demand
- **Authentication** - user accounts with session login, API key management with bearer tokens
- **Require auth toggle** - enforce bearer token authentication on all endpoints (including model loading) or leave model endpoints open
- **Persistent state** - instance history and configs survive container restarts
- **Storage backends** - JSON files (default) or MariaDB/MySQL via SQLAlchemy
- **Proxy sampling overrides** - force temperature, top-k, top-p, presence penalty, and repeat penalty on all proxied requests, configurable per model preset
- **Docker image management** - pull any llama.cpp image by name, delete old local images from the UI

## What's New

- **Universal GPU support** - single image for NVIDIA, AMD (ROCm), Intel Arc, and CPU. GPU vendor auto-detected at startup; `LLAMA_IMAGE` auto-selected from the detected vendor. `GPU_TYPE` overrides if needed.
- **Native GPU monitoring** - VRAM and utilization queried inside the llamaman container (pynvml for NVIDIA, `/sys/class/drm` sysfs for AMD/Intel Arc). GPU panel works without a running llama-server instance.
- **Container resource monitoring** - each running instance card shows live CPU%, core quota, RAM used/limit, and GPU assignment with thin usage bars under each value.
- **Docker image management** - pull any llama.cpp image by name, delete old local images from the Settings UI.
- **Model backup and restore** - export model metadata and presets to JSON; restore on any instance with downloads queued automatically for missing models.
- **Repeat penalty in proxy sampling overrides** - configurable per preset, default 0 (disabled).
- **CPU quota + memory limit** - CPU Threads setting now also applies a Docker CPU quota; new Memory Limit field caps container RAM.

## Tags

- `latest`, `<version>` - Universal image, auto-detects GPU vendor (NVIDIA / AMD / Intel Arc / CPU)

## Quick Start

Pull the llama.cpp image for your GPU first, then run LlamaMan.

`HOST_MODELS_DIR` and `HOST_LOGS_DIR` must be the **absolute paths on the Docker host** that match your volume mounts. LlamaMan passes these to the Docker daemon when spawning sibling llama-server containers.

### NVIDIA

```bash
docker pull ghcr.io/ggml-org/llama.cpp:server-cuda

docker network create llamaman-net

docker run -d \
  --name llamaman \
  --network llamaman-net \
  -p 5000:5000 \
  -p 42069:42069 \
  -p 8000-8020:9000-9020 \
  -v /path/to/models:/models \
  -v /path/to/data:/data \
  -v /path/to/logs:/tmp/llama-logs \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /sys/class/drm:/sys/class/drm:ro \
  -e LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda \
  -e HOST_MODELS_DIR=/path/to/models \
  -e HOST_LOGS_DIR=/path/to/logs \
  --restart unless-stopped \
  nullata/llamaman:latest
```

For native GPU monitoring (pynvml), add `--gpus` with utility capability:
```bash
  --gpus '"driver=nvidia,capabilities=utility"' \
```

### AMD (ROCm)

```bash
docker pull ghcr.io/ggml-org/llama.cpp:server-rocm

docker network create llamaman-net

docker run -d \
  --name llamaman \
  --network llamaman-net \
  -p 5000:5000 \
  -p 42069:42069 \
  -p 8000-8020:9000-9020 \
  -v /path/to/models:/models \
  -v /path/to/data:/data \
  -v /path/to/logs:/tmp/llama-logs \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /sys/class/drm:/sys/class/drm:ro \
  -e LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-rocm \
  -e HOST_MODELS_DIR=/path/to/models \
  -e HOST_LOGS_DIR=/path/to/logs \
  --restart unless-stopped \
  nullata/llamaman:latest
```

### Intel Arc

```bash
docker pull ghcr.io/ggml-org/llama.cpp:server-sycl

docker network create llamaman-net

docker run -d \
  --name llamaman \
  --network llamaman-net \
  -p 5000:5000 \
  -p 42069:42069 \
  -p 8000-8020:9000-9020 \
  -v /path/to/models:/models \
  -v /path/to/data:/data \
  -v /path/to/logs:/tmp/llama-logs \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /sys/class/drm:/sys/class/drm:ro \
  -e LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-sycl \
  -e HOST_MODELS_DIR=/path/to/models \
  -e HOST_LOGS_DIR=/path/to/logs \
  --restart unless-stopped \
  nullata/llamaman:latest
```

### Docker Compose

```yaml
services:
  llamaman:
    image: nullata/llamaman:latest
    ports:
      - "5000:5000"
      - "42069:42069"
      - "8000-8020:9000-9020"
    volumes:
      - /path/to/models:/models
      - /path/to/data:/data
      - /path/to/logs:/tmp/llama-logs
      - /var/run/docker.sock:/var/run/docker.sock
      - /sys/class/drm:/sys/class/drm:ro
    environment:
      - LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda
      # Must be the absolute host-side paths matching the volume mounts above.
      - HOST_MODELS_DIR=/path/to/models
      - HOST_LOGS_DIR=/path/to/logs
    # NVIDIA native GPU monitoring (pynvml) - uncomment on NVIDIA hosts.
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           capabilities: [utility]
    networks:
      - llamaman-net
    restart: unless-stopped

networks:
  llamaman-net:
    driver: bridge
    name: llamaman-net
```

## Ports

| Port | Description |
|---|---|
| `5000` | Management UI and REST API |
| `42069` | Ollama-compatible API proxy |
| `8000-8020` | Individual llama-server instances |

## Volumes

| Path | Description |
|---|---|
| `/models` | GGUF model files. Place your models here or use the built-in download manager. |
| `/data` | Persistent state: instance configs, presets, user accounts, settings, API keys. |
| `/tmp/llama-logs` | Instance and download logs. Optional - mount to preserve logs across restarts. |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLAMA_IMAGE` | *(auto)* | llama.cpp server image for spawned containers. Auto-selected from detected GPU vendor if not set. Set explicitly to pin a version or backend (`server-cuda`, `server-rocm`, `server-sycl`, `server`). |
| `GPU_TYPE` | *(auto-detect)* | Override GPU vendor detection: `cuda`, `rocm`, or `intel`. Leave unset to auto-detect. |
| `LLAMA_GPU_DEVICES` | *(all)* | Comma-separated GPU indices visible to spawned containers, e.g. `0,1`. Not supported on Intel Arc. |
| `LLAMAMAN_MAX_MODELS` | `0` | Max concurrent **chat** models via the proxy. Uses LRU eviction when the limit is reached. `0` = unlimited. |
| `LLAMAMAN_IDLE_TIMEOUT` | `0` | Idle timeout in minutes for proxy-managed instances. Stopped instances auto-restart on next request. `0` = disabled. |
| `LLAMAMAN_PROXY_PORT` | `42069` | Port for the Ollama-compatible proxy. |
| `MODELS_DIR` | `/models` | Directory scanned for model files (container path). |
| `DATA_DIR` | `/data` | Directory for persistent config/state. |
| `LOGS_DIR` | `/tmp/llama-logs` | Directory for instance and download logs (container path). |
| `HOST_MODELS_DIR` | *(same as `MODELS_DIR`)* | **Host-side** absolute path of the models volume. Must match the left side of `-v /host/path/models:/models`. LlamaMan passes this to the Docker daemon when spawning sibling containers. |
| `HOST_LOGS_DIR` | *(same as `LOGS_DIR`)* | **Host-side** absolute path of the logs volume. Same requirement as `HOST_MODELS_DIR`. |
| `PORT_RANGE_START` | `8000` | Start of public llama-server/proxy port pool. |
| `PORT_RANGE_END` | `8020` | End of public llama-server/proxy port pool. |
| `INTERNAL_PORT_RANGE_START` | `9000` | Start of internal llama-server port pool used for proxied instances. |
| `INTERNAL_PORT_RANGE_END` | `9020` | End of internal llama-server port pool used for proxied instances. |
| `SECRET_KEY` | *(auto)* | Flask session secret. Auto-derived from machine-id if unset. |
| `DATABASE_URL` | *(unset)* | MariaDB/MySQL connection string (e.g. `mysql+pymysql://user:pass@host/db`). Unset = JSON file storage. |
| `HEALTH_CHECK_TIMEOUT` | `3` | Timeout in seconds for instance health checks. |
| `MODEL_LOAD_TIMEOUT` | `300` | Seconds to wait for a model to become healthy during launch/relaunch. Increase for very large models. |
| `REQUEST_TIMEOUT` | `300` | Timeout in seconds for upstream requests to llama-server and gate acquire waits. |

## First Launch

1. Start the container
2. Open **http://localhost:5000** in your browser
3. Create an admin account on the `/setup` page
4. Place GGUF model files in the `models/` volume, or download from HuggingFace via the UI

## Cleanup Settings

The UI provides automatic cleanup under **Settings >> Cleanup Settings**:

- **Auto-clean completed/failed downloads** - removes download records older than a configurable number of hours (default: 24). Only affects completed, failed, or cancelled downloads - active downloads are never touched.
- **Auto-clean stopped instances** - removes stopped instance records older than a configurable number of hours (default: 24). Only affects stopped instances - running instances are never removed.
- **Auto-remove stale instance records** - periodically checks all `starting`/`healthy`/`sleeping` instance records against their backing Docker container. Records whose container is no longer running are marked stopped. Configurable check interval (default: 5 minutes).

Cleanup runs periodically in the background. These settings only remove or update records in the UI/state - they do not delete model files.

## OpenWebUI Integration

Point OpenWebUI at the Ollama-compatible proxy:

```yaml
open-webui:
  environment:
    - OLLAMA_BASE_URL=http://llamaman:42069
```

LlamaMan auto-launches models on demand:

1. OpenWebUI calls `/api/tags` and gets the available models.
2. A request to `/api/chat` or `/api/generate` starts the selected model automatically using saved presets or defaults.
3. When `LLAMAMAN_MAX_MODELS` is reached, the proxy evicts the least-recently-used **Ollama-managed** chat model first.

Supported Ollama endpoints: `/api/tags`, `/api/chat`, `/api/generate`, `/api/show`, `/api/version`, `/api/ps`

Also supports OpenAI-compatible auto-start endpoints: `/v1/models`, `/v1/chat/completions`

### With authentication enabled (default)

Create an API key in the LlamaMan UI, then configure OpenWebUI:

```yaml
open-webui:
  environment:
    - OLLAMA_BASE_URL=http://llamaman:42069
    - OPENAI_API_BASE_URLS=http://llamaman:42069/v1
    - OPENAI_API_KEYS=llm-your-api-key-here
```

### Model eviction policy

The `LLAMAMAN_MAX_MODELS` limit controls how many **chat** models the proxy keeps loaded simultaneously.

| Launcher | Eviction behavior | Cannot evict |
|---|---|---|
| **Admin UI** | Evicts Ollama-managed models first (LRU), then admin-launched models if needed | - |
| **Ollama API** (`/api/chat`, `/api/generate`) | Evicts Ollama-managed models (LRU) | Admin-launched models by default |
| **OpenAI API** (`/v1/chat/completions`) | Does not evict; only starts a model if a slot is free | Everything |

Two settings under **Settings >> App Settings** control this behavior:

- **Enforce `LLAMAMAN_MAX_MODELS` for admin UI launches** - when on, the admin UI evicts the LRU model before launching. When off (default), the UI prompts before exceeding the cap.
- **Allow Ollama API to evict admin-launched models** - when on, the Ollama API may evict admin-launched models as a fallback. Off by default. This does not affect the OpenAI API, which never evicts.

Other details:

- All running chat instances count toward the limit, including admin-launched and proxy-managed instances.
- Embedding models are excluded from the limit and are never evicted.
- `LLAMAMAN_MAX_MODELS=0` disables eviction entirely.

## MariaDB / MySQL Setup

By default LlamaMan uses JSON files. To use MariaDB/MySQL, create a database and dedicated user:

```sql
CREATE DATABASE llamaman CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'llamaman'@'%' IDENTIFIED BY 'yourpassword';
GRANT ALL PRIVILEGES ON llamaman.* TO 'llamaman'@'%';
FLUSH PRIVILEGES;
```

Then set `DATABASE_URL` in your container environment:

```
DATABASE_URL=mysql+pymysql://llamaman:yourpassword@host:3306/llamaman
```

Tables are auto-created on first connection.

## Per-Instance Proxy

When **Idle Timeout**, **Max Concurrent**, or **Proxy Sampling Overrides** are enabled for an instance, LlamaMan places a proxy in front of that instance's port. The proxy handles auth, concurrency gating, wake-on-request, and model name validation.

Saving a preset propagates idle-timeout, queue, and proxy-sampling fields to running instances live without a relaunch. If the instance was launched with all three of the above off, no proxy was spawned, so toggling **Proxy Sampling Overrides** on live applies only to requests routed through the main app's Ollama/OpenAI compat endpoints; direct hits to the public port require a relaunch to take effect.

On inference endpoints, if the request body includes a `"model"` field, the proxy validates it against the loaded model's filename stem. A prefix match is accepted (e.g. `"qwen2.5-0.5b-instruct-q2"` matches `"qwen2.5-0.5b-instruct-q2_k"`). A mismatch returns HTTP 404. Requests without a `"model"` field are forwarded unconditionally.

For sleeping instances, a mismatched model name returns 404 without waking the instance.

## Requirements

- Docker with access to `/var/run/docker.sock`
- GPU support (one of):
  - **NVIDIA**: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed (`docker run --gpus all` must work)
  - **AMD**: [ROCm-compatible setup](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
  - **Intel Arc**: `/dev/dri` accessible, user in `video`/`render` groups
  - **CPU only**: no GPU required

## Links

- **Source**: [GitHub](https://github.com/nullata/llamaman)

## License

LlamaMan is licensed under the [Elastic License 2.0](https://github.com/nullata/llamaman/blob/main/LICENSE). You may use, copy, distribute, and modify the software, subject to the following limitations:

- You may not provide the software to third parties as a hosted or managed service where the service gives users access to a substantial set of its features or functionality.
- You may not remove or obscure any licensing, copyright, or other notices of the licensor.
