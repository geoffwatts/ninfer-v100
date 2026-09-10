# Qwen3.8-27B NVFP4 regression on Tesla V100 after 03fc0756

Hi Geoff, Thanks for very interesting work, below is what ChatGPT summarized for me. /Anders H.
---

I'm using `ninfer-v100` with Qwen3.8-27B NVFP4 on a Tesla V100 32 GB and believe I have reproduced a regression between `03fc0756` and current master.

Issues are disabled on this repository, so I'm submitting this as a draft PR to provide a reproducible report. Please feel free to close the PR after reviewing it.

## Working revision

The following revision works correctly:

```text
03fc0756742dcd2707c830dfe176b6e81dd5cd6e
```

Using the pre-DFlash2 Qwen3.8-27B NVFP4 artifact:

```text
SHA-256:
bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32
```

Example command:

```bash
./build-v100/apps/ninfer \
  models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Reply with exactly: MTP works." \
  --max-context 4096 \
  --max-new 64 \
  --kv-dtype int8 \
  --no-thinking \
  --greedy \
  --spec mtp \
  --draft-tokens 3 \
  --lm-head-draft \
  --print-token-ids
```

Result:

```text
MTP works.
generated token ids: 44 4100 4138 13 248046
finish reason: stop-token
generated tokens: 5
decode speed: 77.05 tok/s
mtp acceptance rate: 100%
```

A minimal test using BF16 KV, no MTP, no thinking and greedy decoding also produces the expected output on this revision.

## Current master fails

Current master tested:

```text
8fd0e2efdea77bab944991f2394309c07b8baffe
```

Using the same pre-DFlash2 artifact:

```bash
./build-v100/apps/ninfer \
  models-v100/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Reply with exactly: Qwen3.8 V100 works." \
  --max-context 32768 \
  --max-new 128 \
  --kv-dtype int8 \
  --spec mtp \
  --draft-tokens 3 \
  --lm-head-draft
```

Generation terminates immediately:

```text
finish reason             stop-token
generated tokens          1
decode                     0.000 s
mtp rounds                 0
mtp fallback steps         0
mtp drafted tokens         0
mtp accepted tokens        0
```

No visible response text is generated.

Because `mtp rounds` is zero, the bad first token is produced before speculative decoding runs.

## Minimal reproduction without MTP

I also tested current master with MTP disabled, BF16 KV, no thinking and greedy decoding:

```bash
./build-v100/apps/ninfer \
  models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Reply with exactly: NInfer server works." \
  --max-context 4096 \
  --max-new 64 \
  --kv-dtype bf16 \
  --no-thinking \
  --greedy \
  --print-token-ids
```

Current master generates only:

```text
248046
```

and immediately exits with:

```text
finish reason: stop-token
```

Running the exact same artifact and test after checking out `03fc0756` produces the correct answer.

This appears to place the regression after `03fc0756`. The first commit I would suspect is:

```text
2613471d4c496064528a8a09c6d80b7fd95bd7c8
perf(sm70): prepack quantized weights for Volta QPN kernels
```

because it substantially changes the Volta NVFP4/FP8 execution path.

## Environment

```text
GPU: NVIDIA Tesla V100 32 GB
CUDA Toolkit: 12.8
CUDA architecture: sm_70
Host compiler: GCC/G++ 14
Model: Qwen3.8-27B NVFP4
Artifact: pre-DFlash2 NInfer artifact
Artifact SHA-256:
bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32
Working revision:
03fc0756742dcd2707c830dfe176b6e81dd5cd6e
Failing revision:
8fd0e2efdea77bab944991f2394309c07b8baffe
```

## Separate artifact compatibility issue

The newest Hugging Face Qwen3.8-27B NVFP4 artifact now contains DFlash2 objects such as:

```text
dflash2/feature_projection
```

Current `ninfer-v100` rejects that artifact during `target-plan` with:

```text
artifact object was not consumed by the selected target:
dflash2/feature_projection
```

I therefore used the older pre-DFlash2 artifact for the regression comparison above.

I'm happy to run additional commands, test a patch, or provide full logs if useful.
