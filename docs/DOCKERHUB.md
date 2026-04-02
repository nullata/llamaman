# LlamaMan

![LlamaMan](https://raw.githubusercontent.com/nullata/llamaman/main/docs/llamaman.jpg)

A browser-based UI for launching, monitoring, and managing multiple [llama.cpp](https://github.com/ggerganov/llama.cpp) server instances from inside a Docker container. Includes an Ollama-compatible API proxy so it works as a drop-in replacement for Ollama with [Open WebUI](https://github.com/open-webui/open-webui).

## Features

- **Model library** - scans `/models` for GGUF files, shows quant type and file size
- **One-click launch** - configure GPU layers, context size, threads, multi-GPU, extra args
- **Preset configs** - save/load per-model launch settings
- **Download manager** - pull models from HuggingFace with speed throttling and auto-retry on failure
- **Instance management** - stop, restart, remove, view live-streamed logs
- **GPU VRAM indicator** - per-GPU usage bars via nvidia-smi or rocm-smi
- **Idle timeout** - auto-sleep instances after configurable idle period, wake on next request
- **Ollama-compatible proxy** - OpenWebUI discovers models and auto-starts servers on demand
- **Authentication** - user accounts with session login, API key management with bearer tokens
- **Require auth toggle** - enforce bearer token authentication on all endpoints (including model loading) or leave model endpoints open
- **Persistent state** - instance history and configs survive container restarts
- **Storage backends** - JSON files (default) or MariaDB/MySQL via SQLAlchemy
- **Proxy sampling overrides** - force temperature, top-k, top-p, and presence penalty on all proxied requests, configurable per model preset

## Tags

- `cuda-latest`, `cuda-<version>` - NVIDIA GPU (CUDA) support
- `rocm-latest`, `rocm-<version>` - AMD GPU (ROCm) support *(experimental, not tested)*
- 🆕 `turboquant-cuda-latest`, `turboquant-cuda-<version>` 🆕 - [Experimental branch](https://github.com/nullata/llamaman/tree/tq-cuda-experimental) for the llama.cpp TurboQuant implementation. *(⚠️ early stage experimental feature)*

⚠️ **Dev note:** I do not own an AMD GPU and I am unable to test the ROCm functionality. I encourage users who can test the ROCm images to leave some feedback on the GitHub page for the project. For that matter, **ALL user feedback is welcome.**

## Quick Start

### NVIDIA (CUDA)

```bash
docker run -d \
  --name llamaman \
  --gpus all \
  -p 5000:5000 \
  -p 42069:42069 \
  -p 8000-8020:8000-8020 \
  -v ./models:/models \
  -v ./data:/data \
  -v ./logs:/tmp/llama-logs \
  --restart unless-stopped \
  nullata/llamaman:cuda-latest
```

### AMD (ROCm) - experimental (untested)

```bash
docker run -d \
  --name llamaman \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --group-add render \
  -p 5000:5000 \
  -p 42069:42069 \
  -p 8000-8020:8000-8020 \
  -v ./models:/models \
  -v ./data:/data \
  -v ./logs:/tmp/llama-logs \
  --restart unless-stopped \
  nullata/llamaman:rocm-latest
```

### Docker Compose

```yaml
services:
  llamaman:
    image: nullata/llamaman:cuda-latest
    ports:
      - "5000:5000"
      - "42069:42069"
      - "8000-8020:8000-8020"
    volumes:
      - ./models:/models
      - ./data:/data
      - ./logs:/tmp/llama-logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
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
| `LLAMAMAN_MAX_MODELS` | `0` | Max concurrent **chat** models via the proxy. Uses LRU eviction when the limit is reached. `0` = unlimited. |
| `LLAMAMAN_IDLE_TIMEOUT` | `0` | Idle timeout in minutes for proxy-managed instances. Stopped instances auto-restart on next request. `0` = disabled. |
| `LLAMAMAN_PROXY_PORT` | `42069` | Port for the Ollama-compatible proxy. |
| `MODELS_DIR` | `/models` | Directory scanned for model files. |
| `DATA_DIR` | `/data` | Directory for persistent config/state. |
| `LOGS_DIR` | `/tmp/llama-logs` | Directory for instance and download logs. |
| `PORT_RANGE_START` | `8000` | Start of public llama-server/proxy port pool. |
| `PORT_RANGE_END` | `8020` | End of public llama-server/proxy port pool. |
| `INTERNAL_PORT_RANGE_START` | `9000` | Start of internal llama-server port pool used for proxied instances. |
| `INTERNAL_PORT_RANGE_END` | `9020` | End of internal llama-server port pool used for proxied instances. |
| `SECRET_KEY` | *(auto)* | Flask session secret. Auto-derived from machine-id if unset. |
| `DATABASE_URL` | *(unset)* | MariaDB/MySQL connection string (e.g. `mysql+pymysql://user:pass@host/db`). Unset = JSON file storage. |
| `HEALTH_CHECK_TIMEOUT` | `3` | Timeout in seconds for instance health checks. |
| `MODEL_LOAD_TIMEOUT` | `300` | Seconds to wait for a model to become healthy during launch/relaunch. Increase for very large models. |
| `REQUEST_TIMEOUT` | `300` | Timeout in seconds for upstream requests to llama-server and gate acquire waits. Increase if requests are being cut off under heavy concurrency. |

## First Launch

1. Start the container
2. Open **http://localhost:5000** in your browser
3. Create an admin account on the `/setup` page
4. Place GGUF model files in the `models/` volume, or download from HuggingFace via the UI

## Cleanup Settings

The UI provides automatic cleanup under **Settings >> Cleanup Settings**:

- **Auto-clean completed/failed downloads** - removes download records older than a configurable number of hours (default: 24). Only affects completed, failed, or cancelled downloads - active downloads are never touched.
- **Auto-clean stopped instances** - removes stopped instance records older than a configurable number of hours (default: 24). Only affects stopped instances - running instances are never removed.
- **Auto-remove stale instance records** - periodically checks all `starting`/`healthy`/`sleeping` instance records against their actual OS process. Records whose backing process is no longer alive are marked stopped. Configurable check interval (default: 5 minutes).

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

## Requirements

- Docker with GPU support:
  - **NVIDIA**: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed and working (`docker run --gpus all` must work)
  - **AMD**: [ROCm-compatible setup](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) *(experimental, not tested)*
- A supported GPU (llama.cpp can fall back to CPU/RAM when VRAM is insufficient)

## Links

- **Source**: [GitHub](https://github.com/nullata/llamaman)

## License

[Elastic License 2.0](https://github.com/nullata/llamaman/blob/main/LICENSE)
