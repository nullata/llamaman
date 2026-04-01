# TurboQuant+ KV Cache Compression

LlamaMan includes optional support for [TurboQuant+](https://github.com/TheTom/turboquant_plus), an implementation of [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026) that compresses the KV cache using PolarQuant + Walsh-Hadamard rotation. The result is 3.8–6.4x smaller KV cache with near-zero quality loss, letting you run much larger contexts in the same VRAM.

### ⚠️ Experimental early stage feature. ⚠️

## Credits

This development would not be possible without the work of:
- [Google Research](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) - original TurboQuant paper (ICLR 2026)
- [0xSero/turboquant](https://github.com/0xSero/turboquant)
- [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)
- [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch)
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)

More info on the base implementation: [TheTom/turboquant_plus - Getting Started](https://github.com/TheTom/turboquant_plus/blob/main/docs/getting-started.md)

## What it does

The KV cache stores attention keys and values across all tokens in the context window. At long contexts it becomes the main VRAM bottleneck. TurboQuant+ adds three new cache formats on top of the standard llama.cpp types:

| Format | Bits/val | Compression | Quality vs q8_0 |
|--------|----------|-------------|-----------------|
| `turbo4` | 4.25 | 3.8x | +0.23% PPL |
| `turbo3` | 3.5 | 4.6x | +1.06% PPL |
| `turbo2` | 2.5 | 6.4x | +6.48% PPL |

`turbo4` is the recommended starting point - better quality than `q4_0` at better compression, and within 0.23% of `q8_0`.

## Building

TurboQuant+ is not yet in mainline llama.cpp. The turbo cache types and CUDA kernels live on the `feature/turboquant-kv-cache` branch of `TheTom/llama-cpp-turboquant`. A dedicated Dockerfile is included that compiles llama-server from that branch:

```bash
docker build -f Dockerfile.turboquant -t llamaman-turboquant .
docker run --gpus all ... llamaman-turboquant
```

The build compiles for all major CUDA architectures (Pascal through Hopper) so the resulting image is portable across GPU generations. Build time is longer than the standard image since llama-server is compiled from source.

## Usage

Once running on the TurboQuant image, set **KV Cache Type K** and **KV Cache Type V** in the Launch Instance form before starting a server. The values are saved with presets.

**Recommended configs:**

- **Best quality:** `-ctk turbo4 -ctv turbo4` - nearly identical to q8_0, 3.8x compression
- **Maximum compression:** `-ctk q8_0 -ctv turbo3` - keep K at full precision, compress V aggressively
- **Extreme memory pressure:** `-ctk q8_0 -ctv turbo2` - combine with boundary-layer protection (automatic on recent builds)

**For low-bit weight models (Q4_K_M):** symmetric turbo can degrade quality on some models. Use asymmetric K/V instead:

```
K: q8_0    V: turbo4
```

This is safe for most Q4_K_M models. Larger models (70B+) generally absorb symmetric turbo without issue.

## Asymmetric K/V

K and V compression are independent. K controls attention routing via softmax and is the dominant quality factor - compressing it too aggressively causes most of the quality loss. V compression is effectively free at any precision when K is maintained.

This means you can usually set V to `turbo2` or `turbo3` while keeping K at `q8_0` and see almost no quality change, recovering the majority of the VRAM savings.

## Flash Attention

TurboQuant+ works with flash attention. Add `--flash-attn` to Extra Args for additional memory and speed improvements alongside turbo cache types.

## Compatibility

- Validated on CUDA (RTX 3090, 4090, 5090) and Apple Silicon (Metal)
- Works with Llama, Qwen, Mistral, Command-R, and Phi model families
- Standard `q8_0`, `q4_0`, and `f16` cache types continue to work on the TurboQuant image

## Tested models
- GLM-4.7-Flash-UD-Q8_K_XL
- Qwen3.5-35B-A3B-GGUF
- TBA

## Model Selection Guide

TurboQuant+ compresses the **KV cache**, not the model weights. It extends context length within the same VRAM budget. The right turbo config depends on your model's weight quantization, size, and architecture.

**Choosing a config based on weight quantization:**

| Weight quant | Recommended KV config | Why |
|---|---|---|
| Q8_0 or higher | Symmetric: `-ctk turbo4 -ctv turbo4` | Higher-precision weights tolerate symmetric K compression |
| Q4_K_M / Q5_K_M | Asymmetric: `-ctk q8_0 -ctv turbo4` | Low-bit weights are sensitive to K compression - keep K at full precision |
| Uncertain | Asymmetric: `-ctk q8_0 -ctv turbo4` | Safe default; all quality loss comes from K compression, V is nearly free |

**Model size and compression tolerance:**

Larger models absorb turbo compression better than small ones. Upstream benchmarks show:
- 104B Command-R+: +1.9–3.6% PPL degradation
- 70B Llama-3.1: +6.3–11.4% PPL degradation
- 7B Qwen2.5: +1.0–2.0% PPL (with asymmetric config)

MoE models with small active parameter footprints (e.g. Qwen3.5-35B-A3B) benefit heavily - weights are cheap, context is the bottleneck.

**CPU-offloaded and large models:**

Note that some newer architectures (e.g. MoE ISWA models like GPT-OSS) may hit bugs in the experimental branch unrelated to turbo cache types. If you see assertion failures during model loading (particularly in `llama_params_fit` or `build_attn`), try `-fit off` first to rule out the auto-sizing logic, then try standard KV cache types (`f16` / `q8_0`) to isolate whether the crash is turbo-specific or an architecture support issue in the branch.

**Architecture requirements:**
- Works with full-attention transformer models (dense, hybrid, MoE)
- Models with linear-attention or Mamba-style layers have reduced benefits
- Must be in GGUF format
- Validated families: Llama, Qwen, Mistral, Command-R, Phi, Gemma, Falcon
- Newer architectures (MoE ISWA) may work but are less tested - expect rough edges on the experimental branch

**Qwen2.5 note:** Qwen models have extreme K/V norm asymmetry (key norms 172–778 vs value norms 2–4). Symmetric turbo compression on Q4_K_M weights will degrade quality significantly. Always use asymmetric config (`-ctk q8_0 -ctv turbo4`) for Qwen2.5 with low-bit weights.

**Always test with real prompts:** perplexity numbers and attention similarity metrics (even 99.5%+) don't guarantee working generation. Model sensitivity varies even within families - always validate a turbo config with actual conversation before relying on it.