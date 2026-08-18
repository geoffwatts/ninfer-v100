# NInfer for NVIDIA V100

> Fast, single-GPU Qwen inference on Volta (`sm_70`).

This repository is a hardware-validated NVIDIA V100 port of
[NInfer](https://github.com/Neroued/ninfer), a from-scratch C++/CUDA inference engine for a small
set of explicitly registered Qwen checkpoints. It provides a local CLI plus OpenAI- and
Anthropic-compatible HTTP APIs for text, image, and video prompts.

The upstream RTX 5090 (`sm_120a`) path remains available. This fork adds a separate compile-time
Volta path rather than weakening or emulating the Blackwell kernels.

## Volta status

The port is tested on a Tesla V100-SXM2-32GB with CUDA 12.8.61. The following groupwise-integer
artifacts run end to end on the V100:

| Model | Artifact | Text | Vision | MTP | DFlash | BF16 / INT8 KV |
|---|---|:---:|:---:|:---:|:---:|:---:|
| [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | `qwen3_6_27b.ninfer` | yes | yes | yes | — | yes |
| [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | `qwen3_8_27b.ninfer` | yes | yes | yes | — | yes |
| [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | `qwen3_6_35b_a3b.ninfer` | yes | yes | yes | yes | yes |

NVFP4 is intentionally unsupported on Volta: V100 has no FP4 hardware. Use the `groupwise-int`
artifacts above.

Implemented Volta routes include FP16 tensor-core prefill, BF16 and INT8-G64 KV cache, CUDA Graph
decode, MTP speculative decoding, A3B grouped sparse MoE, DFlash for A3B, vision attention, prefix
reuse, and small-scale batched serving. See the [Volta technical notes](docs/volta-port.md) for the
design, validation scope, benchmark method, and remaining limitations.

## V100 performance

Representative single-request measurements on the V100-SXM2-32GB are shown below. Prefill values
use 512-, 4,096-, and 12,000-token fixtures after warm-up. Decode uses CUDA Graphs and the optimized
proposal head; acceptance varies by prompt, so real serving rates will vary.

| Model and configuration | pp512 | pp4096 | pp12000 | Decode |
|---|---:|---:|---:|---:|
| Qwen3.8-27B, BF16 KV | 953.7 tok/s | 1,118.2 tok/s | 1,053.0 tok/s | 51.8 tok/s MTP3 |
| Qwen3.6-35B-A3B, INT8 KV | 625.9 tok/s | 749.3 tok/s | 743.1 tok/s | 167.9 tok/s MTP3 |

On a natural A3B prompt, DFlash with two draft tokens measured 146.3 tok/s; MTP3 remains the default
V100 recommendation at 167.9 tok/s on the corresponding decode benchmark. These are model-specific
measurements, not a cross-model quality or speed comparison.

### Concurrent serving

The latest dense-model tuning run used Qwen3.8-27B with INT8-G64 KV, CUDA Graphs, the optimized
proposal head, MTP3, greedy sampling, 512 generated tokens per request, and prefix reuse disabled.
Each point is aggregate committed throughput while the requested concurrency remained fully
decode-saturated. Route comparisons were warmed, interleaved, and checked with the graphics clock
locked at 1,530 MHz; the representative final values are:

| C | Verify width | Aggregate decode | Aggregate / C | Speedup vs C1 |
|---:|---:|---:|---:|---:|
| 1 | 4 | **51.8 tok/s** | 51.8 tok/s | 1.00x |
| 2 | 8 | **68.5 tok/s** | 34.3 tok/s | 1.32x |
| 4 | 16 | **105.9 tok/s** | 26.5 tok/s | 2.04x |
| 8 | 32 | **181.1 tok/s** | 22.6 tok/s | **3.50x** |

Across the tuning session, the corresponding points moved from 48.9 to 51.8 tok/s at C1, 58.0 to
68.5 at C2, 76.9 to 105.9 at C4, and 106.9 to 181.1 at C8. The measured session gains were
approximately **6%, 18%, 38%, and 65%**, respectively. The largest improvements come from fused
dequantization on Volta tensor cores, exact-shape split-K tuning, and narrow-width quadpair-split-N
kernels; the C1 path is already much closer to its single-pass weight-streaming limit.

For a separate long-running stochastic reference, Qwen3.6-35B-A3B with the same INT8-G64 KV and
MTP3 configuration measured 152.8, 181.6, 205.4, and 220.5 aggregate tok/s at C1, C2, C4, and C8.
Those requests generated 8,192 tokens each, so the A3B figures should not be compared directly with
the shorter greedy Qwen3.8 campaign above.

Real serving throughput varies with prompt, sampling, speculative acceptance, context length, and
clock state; benchmark figures from different protocols are intentionally reported separately.

## Requirements

- 64-bit Linux;
- a Tesla V100 (`sm_70`), with 32GB recommended for the published artifacts;
- CUDA Toolkit 12.8 and a compatible NVIDIA driver;
- CMake 3.28 or newer and a C++20-capable host compiler;
- Ninja, `pkg-config`, `libcurl >= 7.85`, and FFmpeg development libraries
  (`libavformat >= 60`, `libavcodec >= 60`, `libavutil >= 58`, `libswscale >= 7`).

CUDA 12.8 is the final toolkit release that can build `sm_70`; CUDA 13 removed the target. The
Blackwell build instead requires CUDA 13.1 or newer.

## Build for V100

```bash
git clone https://github.com/geoffwatts/ninfer-v100.git
cd ninfer-v100

cmake -S . -B build-v100 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build-v100 --parallel
```

The Volta configuration fetches the pinned, header-only CUTLASS dependency during CMake setup. The
default build produces:

```text
build-v100/apps/ninfer
build-v100/apps/ninfer-serve
```

To build the unchanged upstream Blackwell path, omit the architecture option and use CUDA 13.1+;
the default architecture is `120a`.

## Download a model

Install the Hugging Face CLI, then download one of the registered artifacts:

```bash
hf download neroued/Qwen3.8-27B-NInfer \
  qwen3_8_27b.ninfer \
  --local-dir models

# Or the mixture-of-experts model:
hf download neroued/Qwen3.6-35B-A3B-NInfer \
  qwen3_6_35b_a3b.ninfer \
  --local-dir models
```

Each `.ninfer` file contains the weights and frontend resources required by NInfer. It is not a
Transformers checkpoint, Safetensors distribution, or GGUF file.

## Serve Qwen3.8-27B on port 8080

This is the recommended dense-model starting point for a 32GB V100:

```bash
./build-v100/apps/ninfer-serve models/qwen3_8_27b.ninfer \
  --host 0.0.0.0 \
  --port 8080 \
  --device 0 \
  --max-context 24576 \
  --kv-capacity auto \
  --prefill-chunk 2048 \
  --kv-dtype bf16 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Then send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Explain speculative decoding briefly."}],
    "max_tokens": 256
  }'
```

Use `--kv-dtype int8` when KV capacity matters more than maximum numerical fidelity. On a 32GB V100,
the dense model with MTP and the optimized proposal head has been exercised at 196,608 tokens with
BF16 KV and at the model's 262,144-token ceiling with INT8 KV. Prefill latency, rather than memory,
becomes the practical constraint at those sizes.

Vision is disabled at process startup unless `--vision` is present. DFlash is text-only and is
available only for Qwen3.6-35B-A3B. Startup capability choices cannot be changed by a later request.

## Run the CLI

```bash
./build-v100/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Use `--messages FILE --vision` for image, video, or structured chat input. Generated content goes to
stdout; load, timing, memory, and speculative-decoding statistics go to stderr.

## Tests

```bash
cmake -S . -B build-v100-tests -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DBUILD_TESTING=ON
cmake --build build-v100-tests --parallel
ctest --test-dir build-v100-tests --output-on-failure
```

The V100 qualification includes focused numerical oracles and real-artifact tests in addition to
the general suite. Some tests remain Blackwell-specific or require local model artifacts; the exact
validated inventory and exceptions are recorded in the [Volta technical notes](docs/volta-port.md).

## Documentation

- [Volta port: design, measurements, and limitations](docs/volta-port.md)
- [CLI guide](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [CLI examples](examples/cli/)
- [Upstream RTX 5090 performance](docs/performance.md)
- [Documentation index](docs/README.md)
- [Contributing](CONTRIBUTING.md)

## Scope and license

NInfer is deliberately a specialized runtime for registered artifacts, not a general model loader.
It supports one resident model on one CUDA device and does not provide multi-GPU execution,
CPU/GPU offload, distributed serving, or large-scale preemptive continuous batching.

NInfer is licensed under the [Apache License 2.0](LICENSE). Vendored dependencies retain their own
licenses under `third_party/`. The Volta flash-attention source derived from llama.cpp is MIT
licensed and records its exact provenance in
[`third_party/llama_cpp_fattn`](third_party/llama_cpp_fattn/README.md).
