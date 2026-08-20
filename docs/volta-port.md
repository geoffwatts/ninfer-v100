# NVIDIA Volta (`sm_70`) port

This document describes the architecture, validation, and measured behavior of the Tesla V100 port
in this repository. It is a technical summary of the current implementation, not a chronological
development log.

## Scope

The port adds `CMAKE_CUDA_ARCHITECTURES=70` while retaining NInfer's upstream `120a` build. Volta
support is selected at compile time through `NINFER_VOLTA_BUILD`; the existing Blackwell kernels and
dispatch remain separate.

The implementation has been exercised on a Tesla V100-SXM2-32GB with CUDA 12.8.61. CUDA 12.8 is
required because CUDA 13 removed the `sm_70` target. The current model scope is:

| Target | Artifact profile | Text | Vision | MTP | DFlash | KV cache |
|---|---|:---:|:---:|:---:|:---:|---|
| Qwen3.6-27B | `groupwise-int` | yes | known issue | yes | — | BF16, INT8-G64 |
| Qwen3.8-27B | `groupwise-int` | yes | known issue | yes | — | BF16, INT8-G64 |
| Qwen3.6-35B-A3B | `groupwise-int` | yes | known issue | yes | yes | BF16, INT8-G64 |

NVFP4 execution is excluded because Volta has no FP4 support. The source still contains guarded
NVFP4 definitions so the same tree can build for `sm_120a`.

## Porting strategy

Volta lacks several instructions used by the upstream kernels: BF16 tensor-core MMA, `ldmatrix`,
`cp.async`, TMA, programmatic dependent launch, and FP4 operations. The port uses four approaches:

1. Compile-time dispatch keeps architecture-specific implementations isolated.
2. Volta FP16 tensor cores (`mma.sync.m8n8k4`) and CUTLASS Sm70 kernels handle wide matrix work.
3. SIMT, DP4A, or synchronous-copy fallbacks cover narrow or latency-bound operations.
4. Unsupported product profiles remain explicit rather than silently emulated.

The shared helpers provide safe pre-Ampere behavior for asynchronous-copy and dependent-launch
interfaces. Host-side dispatch uses `NINFER_VOLTA_BUILD`, because `__CUDA_ARCH__` is unavailable in
ordinary host compilation.

### Quantized linear operations

The V100 has FP16 tensor cores but no BF16 tensor-core arithmetic. Wide Q4/Q5/W8 operations cast
BF16 activations to FP16, dequantize weights into FP16 fragments, accumulate with CUTLASS Sm70
TensorOp kernels, and convert outputs back to the runtime type. Small-T decode retains tuned SIMT
and GEMV schedules where launch and conversion overhead dominate.

Measured seams select CUTLASS only where it wins. Important examples include Q4 LinearSwiGLU at
T=33 and Q5 LinearAdd at T=17, while narrower widths remain on their exact SIMT schedules. The
wide-T Q4 MTP verification kernel stages activations in shared memory so expert-independent weight
traffic dominates less of each tile.

### Attention and KV cache

Decode attention has separate BF16 and INT8-G64 Volta bodies. Prefill uses a Volta flash-attention
route based on the proven llama.cpp `m8n8k4` kernel structure, adapted to NInfer's paged cache and
workspace contracts. The vendored source and exact upstream revision are documented in
[`third_party/llama_cpp_fattn`](../third_party/llama_cpp_fattn/README.md).

The prefill launcher supports both BF16 and INT8-G64 cache pages. INT8 values are dequantized into
the shared FP16 staging path, so attention traverses the visible cache once per query block rather
than replaying a small-T decode kernel over the growing prefix. This changes measured prefill
scaling from strongly superlinear behavior to an approximately flat token rate over the tested
range.

The vision attention route uses the same Volta tensor-core family, but its current host/device tile
geometry disagrees when sizing the shared combine buffer. Compute Sanitizer reports an invalid
shared-memory write, so multimodal input is not currently a supported Volta path. Sliding-window
and bidirectional proposal attention used by DFlash have dedicated SIMT fallbacks.

### Gated Delta Network

The recurrent decode body was already SIMT-compatible. Volta-specific work was required around its
input projections, gating projections, workspace accounting, and wide-token routes. State is kept
across token chunks exactly as in the upstream contract; chunk boundaries do not reset recurrence.

### A3B sparse MoE

Replaying the original small-T sparse-MoE kernels over token slices was correct but repeatedly read
expert weights for individual assignments. The grouped Volta route instead sorts assignments by
expert, gathers token columns, reuses each decoded expert tile across a column tile, and scatters the
weighted result back to the token layout.

Selection stores tile-local rank separately from the final packed column. Keeping `local_rank` and
`packed_index` distinct makes gather idempotent and prevents graph replay from interpreting a
previous global packed index as a new local rank. Workspace capacity and launch selection use the
same predicate so the arena contract cannot drift from dispatch.

The qualified grouped path covers Q4 gate/up with Q5 or Q6 down. W8/W8 uses the existing exact
fallback.

### Speculative decoding

MTP is supported for the dense and A3B targets. Three draft tokens with the optimized proposal head
is the current general V100 starting point:

```text
--spec mtp --draft-tokens 3 --lm-head-draft
```

DFlash is supported for text-only Qwen3.6-35B-A3B. Its Volta path includes W8 feature projection,
sliding-window proposal attention, bidirectional verification attention, and the proposal MLP's
fused W8 SwiGLU. On the measured natural-language fixture, two drafts gave the best DFlash result:

```text
--spec dflash --draft-tokens 2 --lm-head-draft
```

Speculative acceptance is prompt-dependent. Exact long greedy output can also branch at a close
logit when a target is evaluated at a different verification width; this is the same batched-target
numerical behavior observed with the existing MTP verifier, not a licensing or cache-state defect.

## Performance

All measurements below use one Tesla V100-SXM2-32GB on CUDA device 0, after one warm-up. Rates are
single-request token throughput. They should be compared only within a table because model,
quantization, KV type, prompt, and speculative acceptance differ.

### Dense 27B regression profile

The shared 27B geometry was measured after the final A3B/DFlash changes to guard against regressions:

| KV cache | pp512 | pp4096 | pp12000 | ordinary tg256 |
|---|---:|---:|---:|---:|
| BF16 | 953.7 | 1,118.2 | 1,053.0 | 32.4 tok/s |
| INT8-G64 | 952.9 | 1,113.4 | 1,048.7 | 32.6 tok/s |

INT8-G64 cuts the 12K KV payload from 752.0 MiB to 387.8 MiB while retaining essentially all
prefill throughput. On a fixed greedy MTP3 review fixture, the optimized proposal head measured
43.9 tok/s. A persistent-server run with a 1,639-token prompt and 8,192 generated tokens measured
1,062.7 prefill tok/s and 39.0 decode tok/s at 2.36 accepted tokens per round.

### Dense 27B long-context decode

At C1 with INT8-G64 KV and MTP3, the current Qwen3.8-27B long-context curve is 51.8, 44.2, and
29.0 tok/s at approximately 1K, 82K, and 250K context. Four changes establish that result:

1. The decode split ceiling no longer binds before the model's 262,144-token limit.
2. Long-window SIMT attention uses measured register targets to hide global-load latency.
3. Verification widths route through a fused INT8-to-FP16 Volta tensor-core kernel while width-one
   proposal steps retain the faster SIMT path.
4. Both INT8 and BF16 tensor-core kernels pad their shared row stride, removing the near-worst-case
   bank serialization caused by the original 512-byte stride.

At width four, shared-tile padding changed the INT8 kernel from 9,277 to 3,630 microseconds at
262K and from 3,308 to 1,288 microseconds at 82K. End to end, 82K moved from 21.0 to 44.2 tok/s over
the complete long-context session. At 250K, padding moved the final route from 15.2 to 29.0 tok/s;
the combined tensor-core routing and padding work reduced round time from 290.3 to 124.1 ms.

### Qwen3.6-35B-A3B prefill

| Route | pp512 | pp4096 | pp12000 |
|---|---:|---:|---:|
| Previous chunked sparse MoE | — | — | 522.5 tok/s |
| Grouped sparse MoE, BF16 KV | 621.4 | 753.9 | 743.5 tok/s |
| Grouped sparse MoE, INT8-G64 KV | 625.9 | 749.3 | 743.1 tok/s |

At 12K, grouped sparse MoE is 42.3% faster than the chunked baseline. INT8-G64 prefill is within
0.1% of the BF16 control on this fixture.

### Qwen3.6-35B-A3B decode

| Path | Decode | Acceptance | Licensed tokens/round |
|---|---:|---:|---:|
| Ordinary CUDA Graph | 136.6 tok/s | — | 1.00 |
| MTP3, optimized head | 167.9 tok/s | 72.2% | prompt-dependent |
| DFlash, 2 drafts, optimized head | 146.3 tok/s | 69.2% | 2.38 |

MTP3 is the fastest result on these fixtures. DFlash remains useful for workloads where its proposal
acceptance differs, but it should not be selected from the synthetic complete-round ceiling alone.

## Memory and context

On a 32GB V100 with the dense 27B model, MTP, and the optimized proposal head, BF16 KV has been
exercised at 196,608 tokens with 1.22 GiB spare. INT8-G64 KV fits the model's full 262,144-token
native context with 5.21 GiB spare. These are allocation ceilings, not latency recommendations;
prefill time becomes the practical constraint at very long context.

`--kv-capacity auto` sizes the shared KV pool from memory remaining after weight and capability
residency. The process fixes model, vision, and speculative allocations at startup.

## Validation

Qualification combines public-operator numerical oracles with real-artifact execution:

- dense Qwen3.6/Qwen3.8 text generation, MTP, prefix reuse, and long-context continuation;
- text attention FP64-oracle coverage across BF16 and INT8-G64 cache routes;
- BF16 and INT8-G64 attention with identity, offset, and fragmented page mappings;
- A3B grouped sparse MoE at narrow, wide, and slice-boundary token counts;
- A3B real-model prefill followed by ordinary and MTP decode;
- DFlash ordinary-equivalence fixture, optimized/full proposal heads, graph replay, concurrent
  requests, prefix restore, partial terminal batches, and cyclic-cache boundary restore;
- workspace high-water checks, guard regions, and input-immutability checks for new launchers.

The full V100 CTest inventory after DFlash was 68 ordinary passes, 11 known architecture-specific
failures, and 5 artifact-dependent skips out of 84 tests. The four DFlash-related operator tests
that previously stopped on architecture guards became ordinary passes; no new failure was added.
The focused real-artifact gates require the matching `.ninfer` files and are not suitable for a
model-free CI runner.

## Known limitations

- Only `groupwise-int` artifacts are supported on V100; NVFP4 requires newer hardware.
- Vision attention currently fails Compute Sanitizer with an invalid shared-memory write; do not
  enable multimodal input on Volta until the combine-buffer geometry is corrected.
- Validation targets the 32GB V100. The published model footprints leave little or no usable KV
  capacity on a 16GB board.
- DFlash is A3B-only and text-only.
- The runtime remains single-GPU, with startup-bounded concurrency rather than large-scale
  preemptive continuous batching.
- The `sm_120a` source path is retained, but this fork's performance and regression campaign is
  centered on V100 hardware.
- The upstream performance document describes RTX 5090 results. Do not compare those values with
  this document without accounting for hardware and benchmark differences.

## Source provenance

The Volta flash-attention implementation incorporates MIT-licensed code derived from llama.cpp,
pinned and described under [`third_party/llama_cpp_fattn`](../third_party/llama_cpp_fattn/README.md).
CUTLASS is fetched header-only at its pinned tag by the Volta CMake configuration. All other
project licensing remains as described in the repository [LICENSE](../LICENSE) and third-party
license files.
