# Flint NVFP4 on two V100 PCIe 32 GB (TP2)

This document covers serving Recursive Flint (`FlintForCausalLM`,
`model_type="flint"`, package `vllm.models.flint`) on two Tesla V100-PCIE-32GB
GPUs with this fork. The target is a single coding agent with tool calls and
reasoning at 262,144 tokens of context, FP16 compute and FP16 KV cache. The
upstream implementation targets SM100 with BF16 and its own CuTe/Triton
kernels. This port keeps none of those kernels and uses the fork's SM70
routes instead.

Status: design and runbook. Every measured value below is **TBD** until
`tools/flint/sm70_validate.sh` has run on the target host. Fill each table from
that run's `summary.md` and record the source revision. The projections in
[Expected costs](#expected-costs) are arithmetic estimates, not measurements.
Do not quote them as results.

## Contract

### Hardware and software

| Item | Contract |
| --- | --- |
| GPUs | 2× Tesla V100-PCIE-32GB, SM70, 80 SMs, HBM2 at 900 GB/s, 250 W board power |
| Interconnect | PCIe only, no NVLink. Peer access goes through the PCIe switch or root complex |
| Driver and toolkit | CUDA 12.8 toolkit for the source build; driver 570.124.06 or newer |
| Python and Torch | Python 3.12, Torch 2.10.0+cu128 (`--torch-backend=cu128`, since CUDA 13 has no SM70 target) |
| Build | Editable source build with `TORCH_CUDA_ARCH_LIST=7.0`, including Flash-V100 (`flash-attention-v100/`) |
| Concurrency | One agent; `--max-num-seqs 2` leaves room for one overlapping request |

### Model

| Field | Value |
| --- | --- |
| Layers | 49: 14 full attention without RoPE (`no_rope_layer_indices`), 35 sliding-window attention (window 513) with RoPE (theta 10000) |
| Attention | 32 query heads, 4 KV heads (GQA 8:1), head dim 128, q/k RMSNorm, learned sinks (`softmax_type="learnable"`) |
| Dense MLP | Layers 0 and 1 (`mlp_only_layers`), intermediate size 6144 |
| MoE | Layers 2–48: 256 experts, top-6, sigmoid router with renormalized top-k, expert intermediate size 512, one ungated shared expert of 1024 |
| Embedding and head | hidden size 2048, vocabulary 150,144, untied |
| Parameters | 39.8B total, about 2.5B active per token |

### Checkpoint

The published NVFP4 GGUF is converted once, offline, into ModelOpt safetensors:

```bash
.venv/bin/python -m vllm.models.flint.convert_gguf SRC.gguf OUT_DIR \
  --hf-config RecursiveSI/flint
```

- Tensor names are the HF names from the upstream GGUF mapping (contract C3).
- Experts and the other quantized projections stay packed NVFP4 in ModelOpt
  layout: `weight` is uint8 holding two E2M1 values per byte, `weight_scale`
  holds one E4M3 value per 16 weights, and `weight_scale_2` is an FP32 global
  scale. There is no `input_scale`, so the scheme is W4A16.
- `config.json` declares `W4A16_NVFP4` with group size 16 and
  `exclude_modules = ["model.embed_tokens", "*.mlp.gate", "lm_head"]`.
- `lm_head.weight` is stored dequantized but normalized in FP16, with the
  scalar `lm_head.weight_scale_2`. In FP16 the model folds the scalar into the
  weight at load time, because the unscaled logits would overflow FP16. In
  other dtypes it applies the scalar to the logits, as upstream does.
- The sinks `self_attn.softmax_offset` are FP32 `[32]` per layer. The model
  shards them by query head across TP ranks.
- No tensor is BF16. Every BF16→FP16 cast checks `|x| <= 65504`.

The convert stage of the runbook checks all of this from the safetensors
headers: dtypes, sink count, expert count, scales, the absence of BF16 and the
quantization config.

### Serving flags

`scripts/serve_flint_nvfp4_v100_pcie.sh MODEL [--pp] [vllm serve options]`
serves the model name `flint` on port 8000 with:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_SKIP_P2P_CHECK=0
PYTORCH_ALLOC_CONF=expandable_segments:True
--tensor-parallel-size 2 --dtype half --kv-cache-dtype auto
--max-model-len 262144 --max-num-seqs 2 --max-num-batched-tokens 8192
--gpu-memory-utilization 0.90 --enable-prefix-caching
--enable-prompt-tokens-details --enable-auto-tool-choice
--tool-call-parser glm47 --reasoning-parser glm47
--exclude-tools-when-tool-choice-none --served-model-name flint
--override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20}'
```

`--pp` switches to pipeline parallelism over the two GPUs (PP2, TP1). Options
appended after it override these defaults. The reasoning parser `glm47` returns
the `<think>` block in the response's `reasoning` field. History may re-send it
as `reasoning` or as the deprecated `reasoning_content`. The chat template
renders it again as `<think>…</think>`.

## Routes

### Attention

| Route | Sinks | Prefill on SM70 | Decode on SM70 | Admission |
| --- | --- | --- | --- | --- |
| `TRITON_ATTN` | Native, same `sinks` semantics as gpt-oss | FP32 FMA on Volta, so long prefill is slow | Unified Triton kernel | Always available. The fallback when the Flash-V100 sink route does not apply |
| `FLASH_ATTN_V100` | Contract C4: optional `sinks` on `flash_attn_prefill_paged` and `flash_attn_decode_paged` | Tensor-core generic paged prefill for every request, D128 | Generic scalar paged decode, D128; the cross-partition reducer applies the sinks | See the selection rules below. D256, XQA, the small-query verifier, the decode bridges and the FP8 bridge have no sink term and are never used for sink layers |

With no `--attention-backend` given and every participating CUDA device at
SM 7.0 or 7.2, `FlintModelConfig` (`vllm/model_executor/models/config.py`)
picks `FLASH_ATTN_V100` only when all of these hold:

- Flash-V100 automatic selection is not disabled in the attention config.
- `flash_attn_v100.SUPPORTS_SINKS` is true. The package computes it from the
  built extension: true only when it exports both `prefill_paged_sinks_fwd`
  and `decode_paged_sinks_fwd`. It is false for an older build, and the
  reader reports false when the extension does not import.
- The model dtype is float16 (`--dtype half`).
- `head_dim` is 128, the only head size whose generic kernels take sinks.
- The KV cache is FP16 (`--kv-cache-dtype auto`).
- There is no speculative decoding.

Otherwise it picks `TRITON_ATTN` and logs the reason. With an FP8 E4M3 KV
cache it also warns, because Triton cannot compile E4M3 on SM70. An explicit
`--attention-backend TRITON_ATTN|FLASH_ATTN_V100` is kept as given, for A/B
runs. A deployment with any non-SM70 device keeps the platform's own choice.

Each Flash-V100 layer then admits its sinks only if it is decoder attention
with head size 128, an FP16 KV cache, contiguous FP32 sinks (the parameter
itself, so loaded weights stay visible), the paged prefill and scalar paged
decode kernels, and no speculative config. An admitted layer turns off XQA,
the small-query verifier, the decode bridges, prefill split-KV, BFLA, the
contiguous and gathered dense prefill and the FP8 bridge. It hands its sinks
to the decode and prefill executors through `DecodeConfig.sinks` and
`PrefillConfig.sinks`, which pass them to the two generic kernels:

- Decode takes the scalar paged decoder.
- Prefill runs every request on the generic paged prefill kernel. A request
  without a prefix reads its raw K/V as one page per request, and a chunked
  or prefix request reads the paged cache. The mixed decode-rows batch is
  bypassed, and the prefix-anchored window and tree attention raise.

A layer with sinks that is not admitted raises at its first forward, unless
`VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK=1`.

The Flash-V100 sink term adds mass to the softmax denominator only and adds
no value vector. Both kernels keep the running maximum `m` of the scaled
scores in natural-log units, as the sinks are. Before the final
normalization:

```text
prefill epilogue:   row_sum += exp(min(sink - row_max, 80))   (also in the LSE)
decode reducer:     m = max(global_max, sink); sum += exp(sink - m)
```

`sinks=None` keeps the previous kernels bit for bit. Full-attention layers
skip RoPE. Sliding layers use window 513 per layer. The model clears the
global `cache_config.sliding_window`, so the KV manager keeps separate full
and sliding groups and the whole model is not treated as SWA.

### MoE

| Route | Selection | Kernel | Contract |
| --- | --- | --- | --- |
| Skinny SM70 (default) | Automatic when `kernel_config.sm70_skinny_moe` is true (the default) and `Nvfp4SkinnySm70Experts.is_supported_config` admits the layer | `torch.ops._C.skinny_moe_qpn_sm70`: QPN NVFP4 grouped GEMM, FP16 activations | Hidden and per-partition intermediate sizes must tile in 16; no expert parallelism, LoRA or bias |
| TurboMind | `--kernel-config.sm70_skinny_moe false` routes ModelOpt NVFP4 MoE to `ModelOptNvFp4SM70MoEMethod` | `nvfp4_sm70_moe.py`: TurboMind repacked FP4 with FP32 accumulation | `(hidden 2048, intermediate 512, 256 experts, top-6)` in `_SUPPORTED_CONTRACTS` |

Both routes keep the experts packed and need `VLLM_SM70_NVFP4_TURBOMIND`
enabled, which is the default. The router is unquantized (`*.mlp.gate` is
excluded), scores with sigmoid and renormalizes the top 6. The shared expert
is a dense ungated MLP of width 1024 whose output is added to the routed sum.
The route a server actually used is visible in its startup log (`Skinny MoE`
lines) and in `/v1/sm70/acceleration`. The runbook saves both for every
benchmark configuration.

### Communication

| Option | Mechanism | Per decode step | Per 8192-token prefill chunk |
| --- | --- | --- | --- |
| TP2, custom all-reduce | `CustomAllreduce` over CUDA IPC and PCIe P2P. World size 2 over PCIe is admitted when the real IPC write test passes (`VLLM_SKIP_P2P_CHECK=0` runs `gpu_p2p_access_check`) | 98 hidden-state all-reduces of 4 KiB (two per layer), plus one for the vocabulary-parallel embedding; latency bound | 98 all-reduces of 32 MiB. Messages above the custom-AR buffer use NCCL |
| TP2, pynccl only | NCCL over PCIe, or through host memory when P2P is unavailable | Same count, higher latency per call | Bandwidth bound |
| PP2 | One activation send and receive per stage boundary | One 4 KiB transfer. The two stages run one after the other for a single request | One 32 MiB transfer per chunk |

TP2 halves the weight and KV bytes each GPU reads per step, but pays for two
collectives per layer. PP2 has almost no communication. For one request,
though, its stages run in sequence: each GPU reads half the weights, but one
after the other, and the full-layer KV is not split across GPUs within a
step. The topology stage measures the eager 4 KiB all-reduce latency and the
device-to-device copy bandwidth. These two numbers are the inputs to the
TP2-versus-PP2 decision below.

### Expected costs

These are back-of-the-envelope figures that set what the benchmarks should
confirm or refute. Measured values replace them.

| Quantity | Formula | Estimate |
| --- | --- | --- |
| Weight bytes read per decode token (batch 1, whole model) | attention 0.52 GB + six routed experts × 47 layers 0.50 GB + shared/dense 0.21 GB + FP16 LM head 0.62 GB | ≈ 1.85 GB, so ≈ 0.92 GB per GPU at TP2 |
| Full-layer KV bytes read per decode token at context L | 14 layers × 4 KV heads × 128 × 2 (K, V) × 2 B × L = 28 KiB × L | 3.5 GiB at 128K, 7 GiB at 256K, halved per GPU at TP2 |
| TP2 communication per decode token | 98 × (eager 4 KiB all-reduce latency from the topo stage) | ≈ 1–2 ms at 10–20 µs per call |
| Full-attention prefill FLOPs for a prompt of L tokens | 14 layers × 32 heads × 2 × 128 × L² | 1.2e14 at 32K, 2.0e15 at 128K, 7.9e15 at 256K |
| Sliding and linear prefill FLOPs | 35 × 32 × 2 × 128 × 513 × L + 2 × 2.5e9 × L | ≈ 7e14 at 128K |

At long context, full-attention prefill dominates. At 128K it is about three
times the linear and MoE work. That is the case for Flash-V100 sinks:
`TRITON_ATTN` runs this part in FP32 FMA on Volta, while Flash-V100 uses the
tensor cores. Long-context decode is bounded by KV bandwidth, which TP2 splits
across both GPUs and PP2 does not. Short-context decode is bounded by weights
plus TP2 collective latency, which can favor PP2.

## Memory budget

Per GPU at TP2. The usable budget at `--gpu-memory-utilization 0.90` is about
28.6 GiB of the 31.7 GiB on a V100-PCIE-32GB.

| Item | Per GPU | Basis |
| --- | --- | --- |
| Routed experts, NVFP4 | ≈ 10.6 GB | 47 × 256 × 3 × 2048 × 512 = 37.85B weights at 4.5 bits (E2M1 + one E4M3 scale per 16), split over 2 |
| Attention, shared experts, dense MLP | ≈ 0.4 GB | 0.92B + 0.30B + 0.08B weights at 4.5 bits, split over 2 |
| Embedding and LM head, FP16 | ≈ 0.6 GB | 2 × 150,144 × 2048 × 2 B, vocabulary-parallel |
| Router and norms | ≈ 0.1 GB | 47 × 256 × 2048, replicated |
| **Weights** | **≈ 12 GB (11.7 GB, 10.9 GiB)** | Before route-specific packed copies (TBD) |
| Full-attention KV, FP16 | 14 KiB/token → 3.5 GiB at 262,144 tokens | 14 layers × 2 local KV heads × 128 × 2 (K, V) × 2 B = 14,336 B |
| Sliding KV, FP16 | 35 KiB/token over (513 + scheduled tokens) per running request: ≈ 0.3 GiB with an 8192-token chunk, ≈ 18 MiB while decoding | 35 layers × 2 local KV heads × 128 × 2 × 2 B, bounded by the 513 window |
| Two 262K requests | ≈ 7.6 GiB of KV | max-num-seqs 2 |
| Activations, workspaces, CUDA graphs, collectives | TBD | Measure: free memory after load, and graph capture size |
| KV pool reported by the server | TBD | `GPU KV cache size: N tokens` in the server log |

FP16 KV fits with a wide margin, so FP8 KV is not used: E4M3 does not compile
in Triton on SM70, and FP16 avoids a second numerical change. PP2 holds about
the same total per GPU. Each stage keeps half the layers with all four KV
heads: 7 full layers × 4 heads × 128 × 2 × 2 B is also 14 KiB per token. The
embedding sits unsharded on stage 0 and the LM head on stage 1, at 0.6 GB each.

## Validation runbook

`tools/flint/sm70_validate.sh` runs on the V100 host from the repository root.
It writes `bench_results/flint_sm70/<timestamp>/` with one log per stage under
`logs/` and a `summary.md` that has a pass/fail table and per-stage tables.
`bench_results/` is git-ignored.

```bash
# Print the build commands; add --yes to run them (uv + .venv only).
tools/flint/sm70_validate.sh --stage build
tools/flint/sm70_validate.sh --stage build --yes
# Everything else, once the GGUF is downloaded.
tools/flint/sm70_validate.sh --stage all \
  --gguf /models/flint-nvfp4.gguf --model /models/flint-nvfp4-modelopt
# Re-run the client checks against a server that is already up.
tools/flint/sm70_validate.sh --stage smoke,quality,agent \
  --server-url http://127.0.0.1:8000
```

| Stage | What it runs | Pass criterion |
| --- | --- | --- |
| topo | `nvidia-smi -L`, `topo -m`, `topo -p2p r/w`, PCIe link, clocks and power; `can_device_access_peer`; vLLM's `run_p2p_access_test` and `gpu_p2p_access_check` with `VLLM_SKIP_P2P_CHECK=0`; a two-rank FP16 all-reduce through vLLM's TP group | Two SM 7.0 GPUs and a correct all-reduce. A failed IPC test or disabled custom all-reduce is a warning, because serving falls back to NCCL |
| build | `requirements/build/cuda.txt`, then `uv pip install --no-build-isolation -e .` with `TORCH_CUDA_ARCH_LIST=7.0` and `--torch-backend=cu128`, then the test requirements. `--flash-only` rebuilds just the Flash-V100 extension in place | `vllm._C`, `flash_attn_v100_cuda`, `paged_kv_utils` and `vllm.models.flint` import; sm_70 is in Torch's arch list. It reports `SUPPORTS_SINKS` and whether the `sinks=` kwargs exist |
| convert | `python -m vllm.models.flint.convert_gguf SRC OUT --hf-config RecursiveSI/flint`, then a header-level checkpoint check | Config fields as above, 49 FP32 sinks, 36,096 expert projections, NVFP4 scale dtypes, `lm_head.weight_scale_2`, no BF16 |
| kernels | `test_sm70_flash_v100_sinks_{prefill,decode}.py`, `test_sm70_nvfp4_contract_flint.py`, plus Flash-V100 paged-prefill determinism, split-KV, anchored SWA, any-page and varlen layout, plus the skinny MoE tests | No failures. A warning if every new test skipped |
| audit | `python -m tools.flint.fp16_range_audit --model DIR --tensor-parallel-size 2` from the repository root | Exit 0: no layer above the FP16 threshold |
| smoke | Start the serve script, wait for `/health`, check `/v1/models`, one chat completion, stop | Non-empty content without raw `<think>` or `<tool_call>` markup. A warning if 17 × 23 is not 391 |
| quality | Greedy coherence (twice), 10 inline arithmetic problems, 5 coding tasks executed against asserts, `tools/flint/needle.py` at 32,768, 131,072 and 250,000 tokens and depths 0.1/0.5/0.9 | Coherent greedy text, ≥ 8/10 arithmetic, ≥ 4/5 coding, every needle found |
| agent | `tools/flint/agent_e2e.py`: multi-turn agent with `list_dir`, `read_file` and `run_python` in a temporary workspace | Tool calls parse, reasoning re-sent in history reaches the prompt, `cached_tokens` > 0 on every later turn, `tool_choice="none"` returns text only, streamed tool call assembles, correct final answer |
| bench | `vllm bench serve` with random prompts, one request at a time, 1 cold + 3 measured requests, 256 output tokens with `--ignore-eos`, at 4K/32K/128K, for each configuration | Every configuration starts and reports TTFT and decode tok/s |

The benchmark drops the first request of each length. `vllm bench serve`
reuses that request's prompt for warmups and for its readiness check, which
would turn it into a prefix-cache hit. A distinct seed per length keeps the
prompts of different lengths from sharing a prefix. Decode tok/s is
`len(itl) / sum(itl)` per request, and the table reports the median.
Configurations are `<tp2|pp2>-<triton|flash>-<skinny|turbomind>`. The default
one-way matrix runs `tp2-flash-skinny` plus one change per factor, and
`--bench-matrix full` runs all eight.

## Results

All TBD. Record the source revision, driver, clocks and power limits from the
topology stage with every table.

### Topology and communication

| Measurement | GPU 0 → 1 | GPU 1 → 0 |
| --- | --- | --- |
| `nvidia-smi topo -m` link | TBD | TBD |
| `can_device_access_peer` | TBD | TBD |
| vLLM CUDA IPC P2P test | TBD | TBD |
| Device copy GB/s (256 MiB) | TBD | TBD |
| Custom all-reduce enabled | TBD | TBD |
| Eager 4 KiB all-reduce µs | TBD | TBD |

### Kernels and FP16 range

| Check | Result |
| --- | --- |
| Sink prefill vs torch reference (SWA 513, D128, GQA 8:1, FP16 KV, chunked prefill, one and many KV tiles; D128 has no split-KV prefill) | TBD |
| Sink decode vs torch reference, including split-KV | TBD |
| `sinks=None` unchanged | TBD |
| MoE contract `(2048, 512, 256, 6)` | TBD |
| Flash-V100 and skinny MoE regression subset | TBD |
| FP16 range audit: max residual, MoE output and attention logits | TBD |

### Quality

| Probe | `TRITON_ATTN` | `FLASH_ATTN_V100` |
| --- | --- | --- |
| Greedy coherence | TBD | TBD |
| Arithmetic (of 10) | TBD | TBD |
| Coding with asserts (of 5) | TBD | TBD |
| Needle at 32,768 tokens (3 depths) | TBD | TBD |
| Needle at 131,072 tokens (3 depths) | TBD | TBD |
| Needle at 250,000 tokens (3 depths) | TBD | TBD |

### Agent

| Check | Result |
| --- | --- |
| Tool calls parse, no raw markup | TBD |
| Reasoning returned and re-sent | TBD |
| `cached_tokens` > 0 on later turns | TBD |
| `tool_choice="none"` | TBD |
| Streaming tool call | TBD |

### Single-request latency

TTFT in ms and decode in tok/s, medians of three requests after one cold
request. Prompts are random and 256 output tokens are generated with
`--ignore-eos`.

| Configuration | TTFT 4K | Decode 4K | TTFT 32K | Decode 32K | TTFT 128K | Decode 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tp2-flash-skinny | TBD | TBD | TBD | TBD | TBD | TBD |
| tp2-triton-skinny | TBD | TBD | TBD | TBD | TBD | TBD |
| tp2-flash-turbomind | TBD | TBD | TBD | TBD | TBD | TBD |
| pp2-flash-skinny | TBD | TBD | TBD | TBD | TBD | TBD |

### Memory

| Measurement | TP2 | PP2 |
| --- | --- | --- |
| Weights per GPU after load | TBD | TBD |
| Memory per GPU after graph capture | TBD | TBD |
| `GPU KV cache size` (tokens) | TBD | TBD |
| Maximum concurrency at 262,144 tokens | TBD | TBD |

## Known limits

- Sinks on Flash-V100 cover only the D128 generic paged prefill and decode
  routes. D256, XQA and the FP8 KV bridge reject sinks and keep their previous
  admission. Without `SUPPORTS_SINKS`, Flint runs on `TRITON_ATTN`, whose
  long-context prefill on Volta is several times slower.
- On the sink route, prefill skips the prefill planner and the mixed
  decode-rows batch, and runs no-prefix prefill on the paged kernel instead of
  the dense one. Its cost against Triton is what the bench stage measures.
- KV cache is FP16 only. FP8 KV is out of scope on this path, because E4M3
  does not compile in Triton on SM70.
- No BF16 anywhere. Activations that leave the FP16 range overflow instead of
  rounding, and the FP16 audit is the gate for that. If it flags a layer, keep
  that layer's residual in FP32 before serving.
- The TurboMind MoE route admits only the exact `(2048, 512, 256, 6)`
  contract at TP 1, 2 or 4. The skinny route rejects expert parallelism, LoRA
  and expert bias.
- Custom all-reduce needs working PCIe P2P. ACS or IOMMU settings on some PCIe
  switches make peer writes fail even when the driver reports peer access.
  The IPC test catches that, and vLLM then uses NCCL only.
- PP2 does not split one request's work across GPUs within a step. It helps
  only where TP2 collective latency dominates.
- Upstream sampling defaults are temperature 1.0, top-p 0.95 and top-k 20.
  Greedy decoding is not validated upstream. The runbook uses greedy only for
  the coherence probe.
- No speculative decoding, no multimodal inputs and no Rust frontend. That
  frontend rejects `--exclude-tools-when-tool-choice-none`.
- The quality probes are small smoke tests, not benchmark scores. They detect
  broken numerics or parsers, not small accuracy regressions.
