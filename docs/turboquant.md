# TurboQuant+ KV Cache Compression

LlamaMan includes optional support for [TurboQuant+](https://github.com/TheTom/turboquant_plus), an implementation of [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026) that compresses the KV cache using PolarQuant + Walsh-Hadamard rotation. The result is 3.8–6.4x smaller KV cache with near-zero quality loss, letting you run much larger contexts in the same VRAM.

## What it does

The KV cache stores attention keys and values across all tokens in the context window. At long contexts it becomes the main VRAM bottleneck. TurboQuant+ adds three new cache formats on top of the standard llama.cpp types:

| Format | Bits/val | Compression | Quality vs q8_0 |
|--------|----------|-------------|-----------------|
| `turbo4` | 4.25 | 3.8x | +0.23% PPL |
| `turbo3` | 3.5 | 4.6x | +1.06% PPL |
| `turbo2` | 2.5 | 6.4x | +6.48% PPL |

`turbo4` is the recommended starting point — better quality than `q4_0` at better compression, and within 0.23% of `q8_0`.

## Building

TurboQuant+ is not yet in mainline llama.cpp. A dedicated Dockerfile is included that compiles llama-server from the fork with CUDA support:

```bash
docker build -f Dockerfile.turboquant -t llamaman-turboquant .
docker run --gpus all ... llamaman-turboquant
```

The build compiles for all major CUDA architectures (Pascal through Hopper) so the resulting image is portable across GPU generations. Build time is longer than the standard image since llama-server is compiled from source.

## Usage

Once running on the TurboQuant image, set **KV Cache Type K** and **KV Cache Type V** in the Launch Instance form before starting a server. The values are saved with presets.

**Recommended configs:**

- **Best quality:** `-ctk turbo4 -ctv turbo4` — nearly identical to q8_0, 3.8x compression
- **Maximum compression:** `-ctk q8_0 -ctv turbo3` — keep K at full precision, compress V aggressively
- **Extreme memory pressure:** `-ctk q8_0 -ctv turbo2` — combine with boundary-layer protection (automatic on recent builds)

**For low-bit weight models (Q4_K_M):** symmetric turbo can degrade quality on some models. Use asymmetric K/V instead:

```
K: q8_0    V: turbo4
```

This is safe for most Q4_K_M models. Larger models (70B+) generally absorb symmetric turbo without issue.

## Asymmetric K/V

K and V compression are independent. K controls attention routing via softmax and is the dominant quality factor — compressing it too aggressively causes most of the quality loss. V compression is effectively free at any precision when K is maintained.

This means you can usually set V to `turbo2` or `turbo3` while keeping K at `q8_0` and see almost no quality change, recovering the majority of the VRAM savings.

## Flash Attention

TurboQuant+ works with flash attention. Add `--flash-attn` to Extra Args for additional memory and speed improvements alongside turbo cache types.

## Compatibility

- Validated on CUDA (RTX 3090, 4090, 5090) and Apple Silicon (Metal)
- Works with Llama, Qwen, Mistral, Command-R, and Phi model families
- Standard `q8_0`, `q4_0`, and `f16` cache types continue to work on the TurboQuant image
