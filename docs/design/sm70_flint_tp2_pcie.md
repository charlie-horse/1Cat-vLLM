# Flint NVFP4 on two V100 PCIe 32 GB (TP2)

This document covers serving Recursive Flint (`FlintForCausalLM`,
`model_type="flint"`, package `vllm.models.flint`) on two Tesla V100-PCIE-32GB
GPUs with this fork. The target is a single coding agent with tool calls and
reasoning at 262,144 tokens of context, FP16 compute and FP16 KV cache. The
upstream implementation targets SM100 with BF16 and its own CuTe/Triton
kernels. This port keeps none of those kernels and uses the fork's SM70
routes instead.

Status: validated on the target host on 2026-10-28; see [Results](#results).
Values not measured yet are marked **TBD**. The projections in
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

### FP16 headroom

Flint was trained in BF16. In FP16 one massive-activation token per sequence
overflows in layer 46 unless the model rescales two places at load time
(`vllm/models/flint/model.py`, FP16 only):

- **MoE intermediate.** Two routed experts of layer 46 reach
  `silu(gate) * up` of 9.1e4 and 9.4e4, above the FP16 maximum of 65504,
  while gate and up (≤ 2.4e3), the expert outputs (≤ 8.2e3) and the MoE
  output (7.3e3) fit. The FP16 intermediate between the W13 and W2 GEMMs
  became inf, the token's hidden state NaN, and generation degenerated. Every
  MoE up projection (routed and shared experts) is divided by 16 and its down
  projection multiplied by 16. In NVFP4 this moves up's E4M3 block scales, so
  gate and up keep one global scale and the skinny route stays admitted, and
  down's FP32 global scale. On this checkpoint 2.5e-4 of the shifted block
  scales round, with a scale-weighted relative error of 6e-9.
- **Residual stream.** From layer 46 on, the MoE output and the residual
  stream reach 3.1e4, 2.1x below the FP16 maximum. The stream from layer 46 on
  is carried divided by 8: layer 46 scales its residual after the attention
  add, layers 47 and 48 scale their attention output, and the MoE down scales
  of layers 46 to 48 are divided by 8. Every RMSNorm, the final one included,
  is invariant to the scale. The audit's logits are the same with and without
  it (max 87.7).

Both are powers of two, so they only move exponents. The FP16 range audit is
the gate: after the change the largest value anywhere is 3.9e3, 16.8x below
the FP16 maximum.

### Serving flags

`scripts/serve_flint_nvfp4_v100_pcie.sh MODEL [--pp] [vllm serve options]`
serves the model name `flint` on port 8000 with:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_SKIP_P2P_CHECK=0
PYTORCH_ALLOC_CONF=expandable_segments:True
--tensor-parallel-size 2 --dtype half --kv-cache-dtype auto
--max-model-len 262144 --max-num-seqs 2 --max-num-batched-tokens 8192
--gpu-memory-utilization 0.90 --kernel-config.sm70_skinny_moe false
--enable-prefix-caching --enable-prompt-tokens-details --enable-auto-tool-choice
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

- Decode takes the scalar paged decoder's entry point. With sinks, 8:1 GQA
  and aligned tensors (Flint at TP1 and TP2), the extension runs it on the
  grouped-query D128 kernel: one CTA per KV head and partition covers the
  eight query heads. Its window-aware reducer adds the sinks. Other shapes
  keep the per-head kernel and the generic reducer.
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
| Skinny SM70 | Automatic when `kernel_config.sm70_skinny_moe` is true (the engine default) and `Nvfp4SkinnySm70Experts.is_supported_config` admits the layer | `torch.ops._C.skinny_moe_qpn_sm70`: QPN NVFP4 grouped GEMM, FP16 activations | Hidden and per-partition intermediate sizes must tile in 16; no expert parallelism, LoRA or bias |
| TurboMind (serve-script default) | `--kernel-config.sm70_skinny_moe false` routes ModelOpt NVFP4 MoE to `ModelOptNvFp4SM70MoEMethod` | `nvfp4_sm70_moe.py`: TurboMind repacked FP4 with FP32 accumulation | `(hidden 2048, intermediate 512, 256 experts, top-6)` in `_SUPPORTED_CONTRACTS` |

The serve script selects TurboMind: it decodes about 40% faster than skinny
at 4K and prefills 20-30% faster (see Decode performance). Skinny's expert
GEMMs are faster per call, but its device-side routing adds about 25 small
kernels per MoE layer.

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
| Weights measured after load | 11.02 GiB | `Model loading took 11.02 GiB` (TP2, skinny MoE) |
| Activations, workspaces, CUDA graphs, collectives | ≈ 0.6 GB | Startup budget: activation peak 0.40 GB, warmup 0.11 GB, idle 0.06 GB, non-torch 0.05 GB; graphs for batch 1 and 2 only |
| KV pool reported by the server | 18.27 GB, 1,067,772 tokens | `GPU KV cache size` at `--gpu-memory-utilization 0.90`; 4.07 requests of 262,144 tokens. 29.6 GiB in use per GPU in `nvidia-smi` |

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

On a NixOS host without system Python or compilers (the reference host),
the build stage's `uv` commands do not apply. There the fork was built in
place against the host's Nix vLLM environment (Torch 2.10.0+cu128, CUDA
12.8, GCC 14 from the service wrapper's exports), with `python setup.py
build_ext --inplace` in `flash-attention-v100/` and then in the repository
root (`TORCH_CUDA_ARCH_LIST=7.0`, `CMAKE_CUDA_ARCHITECTURES=70`,
`CMAKE_ARGS=-DPython3_EXECUTABLE=<env python>`, `cmake`, `git` and `patchelf`
from `nix shell`). A `.venv/bin/python` shim sets `PYTHONPATH` to the
repository and `flash-attention-v100/` so they shadow the installed wheel,
and a `.venv/bin/vllm` shim runs `python -m vllm.entrypoints.cli.main`. The
runbook's other stages then run unchanged. The full build takes about 25
minutes with `MAX_JOBS=24`.

The benchmark drops the first request of each length. `vllm bench serve`
reuses that request's prompt for warmups and for its readiness check, which
would turn it into a prefix-cache hit. A distinct seed per length keeps the
prompts of different lengths from sharing a prefix. Decode tok/s is
`len(itl) / sum(itl)` per request, and the table reports the median.
Configurations are `<tp2|pp2>-<triton|flash>-<skinny|turbomind>`. The default
one-way matrix runs `tp2-flash-skinny` plus one change per factor, and
`--bench-matrix full` runs all eight.

## Results

Measured on 2026-10-28 on the reference host:

| Item | Value |
| --- | --- |
| Host | 2× Tesla V100-PCIE-32GB on one PCIe switch, driver 580.159.03, 250 W, max clocks 1380 MHz SM / 877 MHz memory; 48 CPUs, 125 GB RAM, NixOS 26.05 |
| Software | Torch 2.10.0+cu128, nvcc 12.8.93, GCC 14.4, CMake 4.1.6, Python 3.12.14; in-place source build |
| Source | `claude/flint-v100-pcie`; the fixes found on the GPU are in the commits after `8a77063` |
| Checkpoint | `RecursiveSI/flint-nvfp4-gguf` at `6ec43e19b565fc8d560e2d04985d7724f20673bf`, converted with `--hf-config` from `RecursiveSI/flint` at `2a247be0c0aec895ccb0ac7ca92229ec6df37288` |
| Serving | The serve script's profile (TP2, `FLASH_ATTN_V100` chosen automatically, skinny NVFP4 MoE, FP16 KV, 262,144 tokens, utilization 0.90, FULL decode CUDA graphs) |

### Topology and communication

| Measurement | GPU 0 → 1 | GPU 1 → 0 |
| --- | --- | --- |
| `nvidia-smi topo -m` link | PIX; P2P read and write OK; PCIe Gen3 x16 | PIX |
| `can_device_access_peer` | True | True |
| vLLM CUDA IPC P2P test | True | True |
| Device copy GB/s (256 MiB) | 13.14 | 13.16 |
| Custom all-reduce enabled | True, results correct | True, results correct |
| Eager 4 KiB all-reduce µs | 160.8 (eager Python loop with a clone per call, not a graph replay) | 160.7 |

### Build and conversion

| Check | Result |
| --- | --- |
| Flash-V100 extension | Builds without changes. `SUPPORTS_SINKS` is True; both native sink entries exist |
| Register use (`cuobjdump -res-usage`) | D128 generic paged prefill: 64 registers, 128 to 192 bytes of stack, identical to the 1.5.1 wheel, so the sink epilogue adds no spills. Decode reducer: 32 registers, no stack, also unchanged |
| Conversion | 1 min 58 s, peak RSS 24.7 GB, 21.71 GiB in 5 shards, 109,613 tensors. All fused q/k/v and gate/up groups already share `weight_scale_2`, so unification is a no-op. `lm_head.weight_scale_2` is 1.256e-3 |
| Checkpoint check | 49 FP32 sinks, 36,096 expert projections, no BF16, NVFP4 scale dtypes, ModelOpt `W4A16_NVFP4` group 16 |
| ModelOpt W4A16 NVFP4 on SM70 | Admitted: `get_min_capability` returns 70 with the default SM70 NVFP4 format policy |

### Kernels and FP16 range

| Check | Result |
| --- | --- |
| Sink prefill vs torch reference (SWA 513, D128, GQA 8:1, FP16 KV, chunked prefill, one and many KV tiles; D128 has no split-KV prefill) | Pass (`test_sm70_flash_v100_sinks_prefill.py`) |
| Sink decode vs torch reference, including split-KV | Pass: 77 GPU cases incl. 64K context, every partition size and CUDA graph replay |
| `sinks=None` unchanged | Pass: bit-identical decode, sink-free routes unchanged |
| MoE contract `(2048, 512, 256, 6)` | Pass, 110 tests, after the TurboMind split-K fix (11 to 13 tokens failed before) |
| Flash-V100 and skinny MoE regression subset | 45 of 45 Flash-V100; 64 skinny MoE pass, 13 skipped (need SM75) |
| In-engine attention vs FP32 reference (1,496-token prefill, layers 2, 5, 20, 21, both ranks) | Relative error 2e-4 with the sinks and window 513; dropping the sinks gives 0.1 to 0.8 |
| In-engine MoE vs FP32 dequantized reference (layers 2 to 46, every token) | Median relative error 8e-4, max 1.7e-3 |
| lm_head with the global scale folded into FP16 | Folded weights at most 3.375, none subnormal. Against FP32 logits: max error 0.04 on a ±70 range, top-1 agreement 100 %, mean KL 2e-5 |
| FP16 range audit (4 prompts up to 16,219 tokens, TP2) | Before the [FP16 headroom](#fp16-headroom) fix: NaN from layer 46 on and NaN logits. After: no inf or NaN; largest value 3.9e3 (16.8x headroom); final norm 12.5, logits 87.7, `qk_bound` at most 217 |

### Quality

| Probe | `TRITON_ATTN` | `FLASH_ATTN_V100` |
| --- | --- | --- |
| Greedy coherence | Pass; repeat identical | Pass; repeat differs (the second run hits the prefix cache) |
| Arithmetic (of 10) | 10 | 10 |
| Coding with asserts (of 5) | 5 | 5 |
| Needle at 32,768 tokens (3 depths) | TBD | 3 of 3 (14.5 s each) |
| Needle at 131,072 tokens (3 depths) | TBD | 3 of 3 (125 to 137 s each) |
| Needle at 250,000 tokens (3 depths) | TBD | 3 of 3 (about 448 s each) |

Backend agreement (`TRITON_ATTN` against `FLASH_ATTN_V100`, the same server
profile otherwise):

- Greedy chat continuations of 256 tokens match for 256, 141 and 256 tokens,
  with at most 0.12 difference in log-probability before they diverge.
- Teacher-forced prompt log-probabilities over 1.3K to 1.5K-token user turns
  agree on 75 % (code) and 97 % (prose) of top-1 tokens. The differences sit
  on low-confidence positions: inside a user turn the model often predicts a
  comment on the pasted text instead of the next token (both backends give
  repeated code a mean log-probability of about -3.3 to -5.5). The same text
  outside a chat turn is copied normally (-0.02 by the sixth repeat), so this
  is the model, not the port.

### Agent

| Check | Result |
| --- | --- |
| Tool calls parse, no raw markup | Pass: 6 calls over `list_dir`, `read_file` and `run_python`; final answer correct |
| Reasoning returned and re-sent | Pass: `message.reasoning` on 3 of 5 turns; re-sent `reasoning_content` reaches the prompt (1,105 against 1,096 tokens) |
| `cached_tokens` > 0 on later turns | Pass: 1,088/1,105, 1,120/1,181, 1,200/1,274, 1,344/1,369 |
| `tool_choice="none"` | Fail in the check, by the model: the server drops the tools and returns no `tool_calls`, but with a system prompt that says to use tools the model invents tools (`bash`, `shell_command`) in raw `<tool_call>` markup in 2 of 3 samples. With a neutral system prompt it answers in text. A client can forbid the markup with `logit_bias` `{"150010": -100}` (`<tool_call>`) |
| Streaming tool call | Pass |

### Single-request latency

TTFT in ms and decode in tok/s, medians of three requests after one cold
request. Prompts are random and 256 output tokens are generated with
`--ignore-eos`, with the server's default sampling. Only the 4K baseline
was run here; the configuration matrix and longer inputs are TBD.

| Configuration | TTFT 4K | Decode 4K | TTFT 32K | Decode 32K | TTFT 128K | Decode 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tp2-flash-skinny | 1,069 | 55.5 | TBD | TBD | TBD | TBD |
| tp2-triton-skinny | TBD | TBD | TBD | TBD | TBD | TBD |
| tp2-flash-turbomind | TBD | TBD | TBD | TBD | TBD | TBD |
| pp2-flash-skinny | TBD | TBD | TBD | TBD | TBD | TBD |

The needle runs give the long-prompt TTFT of `tp2-flash-skinny` without
prefix-cache hits and with no other load: about 14.5 s at 32K, 125 to 137 s
at 128K and 448 s at 250K.

### Memory

| Measurement | TP2 | PP2 |
| --- | --- | --- |
| Weights per GPU after load | 11.02 GiB | TBD |
| Memory per GPU after graph capture | 29.6 GiB in `nvidia-smi` at utilization 0.90 | TBD |
| `GPU KV cache size` (tokens) | 1,067,772 (`TRITON_ATTN`: 1,043,750) | TBD |
| Maximum concurrency at 262,144 tokens | 4.07x | TBD |

With the TurboMind MoE route (the serving default since the decode work
below), weights take 12.17 GiB per GPU and the KV cache holds 982,513
tokens, 3.75x a 262,144-token request.

## Decode performance

Single-stream decode on the reference host (TP2, FP16, FULL decode CUDA
graphs). Prompts are random token ids sent to `/v1/completions`, so prefix
caching never hits; 256 tokens are generated with `ignore_eos`. Decode
tok/s is (tokens - 1) over the time from the first to the last streamed
token, median of three requests after a warm-up. "Default" is the served
sampling (temperature 1.0, top-p 0.95, top-k 20); "greedy" is temperature 0.

### Baseline and result

| Context | Before: default / greedy | After: default / greedy | TTFT before / after |
| --- | ---: | ---: | ---: |
| 4K | 52.9 / 60.0 tok/s | **96.3 / 100.6 tok/s** | 1.03 / 0.69 s |
| 32K | 46.7 / 47.6 tok/s | **92.7 / 96.2 tok/s** | 13.07 / 10.36 s |
| 128K | 26.9 tok/s | **77.2 tok/s** | 123.8 / 112.9 s |
| 250K | 15.2 tok/s | **24.0 tok/s; 63.7 cool** | 422.7 / 397.9 s |

Before is the integration branch at `abcc6d9`: skinny NVFP4 MoE and the
per-head Flash-V100 paged decode kernel for sink layers. After is the
`claude/flint-perf` branch (see Changes): the grouped decode kernel, the
metadata fix, and routed experts on the TurboMind route. The 128K and 250K
rows use default sampling: one request at 128K, the median of three cold
requests at 250K. Decode right after a 250K prefill is limited by GPU 1's
temperature; "cool" is the same context served from the prefix cache after
a cool-down (see [Heat at 250K](#heat-at-250k)).

The same host serves Qwen3.6-35B-A3B at 66 tok/s at 4K. The "about 21 tok/s"
first seen on Flint came from long-context requests, where the 128K to 250K
decode rate was 15-27 tok/s and prefill was the bigger cost (TTFT 124 s at
128K, 423 s at 250K).

### Where the time went

The FULL decode graph was replayed on every decode step: 1 CUDA graph launch
per token and no piecewise or eager fallback for batch sizes 1 and 2
(`--cudagraph-metrics`). The workers block on the GPU only for the
previous step's sampled tokens. So the cost sat in three other places,
found with the torch profiler, `nvidia-smi` utilization sampling and a
Python stack sampler in the workers:

1. **Host: Flash-V100 metadata re-resolved the policy every step.** The
   attention metadata builder runs outside a forward context. There, the
   execution-policy and runtime-trace adapters build and resolve a fresh
   standalone Flash-V100 policy from the environment on every call. Flint
   has four KV-cache groups (14 full layers, 35 sliding ones padded to 42),
   and every decode step resolved the policy twice per group, once to check
   dynamic decode partitions and once to find that decode tracing is off.
   This made up most of the 46% of the rank-0 worker's step time spent in
   metadata building. With the TurboMind MoE route, rank 0's GPU sat idle
   30-36% of each step, and rank 1 spent that time spinning in the first
   all-reduce of the step. The skinny route hid it behind its own GPU time.
2. **Kernel count: the skinny MoE routing glue.** The skinny route computes
   its expert grouping with about 25 small PyTorch ops per MoE layer
   (argsort, scatter, cumsum, fills, casts, and the weighted sum as
   multiply + reduce + copies), about 115 us per layer. A decode step
   replayed 2,466 kernels, and the graph's inter-kernel gaps alone came to
   5.6 ms on rank 1. The TurboMind route's single-token path needs 1,197.
3. **Attention: the per-head decode kernel.** One CTA per query head and
   partition read each K/V row eight times (GQA 8:1). In PV, 128 threads
   each walked the 256-token partition alone. The reducer walked every
   partition up to the sequence length, so sliding-window layers got slower
   with context. A full-attention layer took 0.41 ms at 32K, 1.32 ms at
   128K and 2.38 ms at 250K per rank. At 250K the 14 full layers alone
   took 33 ms of the 67 ms step.

Hypotheses that the measurements ruled out:

- **All-reduce over PCIe.** The custom one-stage all-reduce takes 10-13 us
  per call, about 1.0-1.3 ms per token for 99 calls. The 66-101 us averages
  in the first profiles were rank skew absorbed by the first all-reduce of
  each step.
- **The sampler.** Top-k plus top-p over the 150,144-entry vocab costs
  about 0.35 ms of GPU time per token. Greedy is 0.3-0.5 ms faster per
  token.
- **The fp32 router.** The cast plus fp32 GEMV, split-K reduce and top-k
  cost 1.24 ms per token, which is real but secondary.
- **The shared-expert aux stream.** It overlaps the router and the experts;
  disabling it is slightly slower.
- **The forced output-token sync.** The reasoning parser makes sampling
  keep output token ids, so each step waits for the previous step's sampled
  token on the host. Removing that wait changed nothing measurable, so it
  was left in place.

### Per-step breakdown at 4K

Rank 1, torch profiler (which inflates each kernel by about 1 us), ms per
decoded token. After is the final build (grouped decode v3). The last row is
the median inter-token latency without the profiler; with about 1,240
kernels per step, the profiler's inflation is why the rows add up to more:

| Component | Before | After |
| --- | ---: | ---: |
| Kernels per step (graph + eager) | 2,466 + 47 | 1,197 + 47 |
| Dense NVFP4 GEMVs (TurboMind: qkv, o, shared expert) | 3.36 | 3.23 |
| Routed-expert GEMMs | 0.93 (skinny) | 1.51 (TurboMind) |
| MoE routing glue (sort, scatter, scan, casts, weighted sum) | ~4.1 | 0.20 |
| Router (fp32 cast, GEMV, split-K reduce, sigmoid top-6) | 1.24 | 1.24 |
| Attention (partition + reduce, 49 layers) | 3.10 | 1.56 |
| All-reduce transfer (99 calls) | 1.13 | 1.08 |
| RMSNorm, RoPE, KV write, activations, adds | ~1.4 | ~1.3 |
| Inter-kernel gaps inside the graph | 5.6 | 1.1 |
| lm_head, eager FP16 GEMM | 0.41 | 0.40 |
| Sampler and logits gather, eager | ~0.45 | ~0.44 |
| Unprofiled step (greedy / default) | 16.0 / 18.2 | 9.97 / 10.34 |

Per-layer attention microbenchmark (rank shapes 16q/2kv, D128, sinks,
CUDA-graph replay with the server's 262K workspace, including about 10 us of
replay overhead), us per layer:

| Layer, context | Per-head kernel | Grouped v1 + v2 | Grouped v3 (final) |
| --- | ---: | ---: | ---: |
| Full, 4K | 58.5 | 31.4 | 27.8 |
| Full, 32K | 410.7 | 94.4 | 80.9 |
| Full, 128K | 1,323.8 | 315.7 | 264.8 |
| Full, 192K | 1,887.0 | 451.8 | 381.0 |
| Full, 250K | 2,382.0 | 576.8 | 478.6 |
| Sliding 512, 32K | 65.0 | 29.8 | 24.2 |
| Sliding 512, 250K | 248.2 | 35.2 | 30.1 |

At 128K the v3 kernel moves the 134 MB of K and V a rank holds per full
layer at about 510 GB/s, 57% of the V100's 900 GB/s.

### A/B results

Decode tok/s (default / greedy) and TTFT, single stream:

| Configuration | 4K | 32K | TTFT 4K / 32K |
| --- | ---: | ---: | ---: |
| Before (skinny, per-head attention) | 52.9 / 60.0 | 46.7 / 47.6 | 1.03 / 13.07 s |
| + grouped decode kernel | 56.1 / 57.9 | 54.3 / 57.3 | 1.03 / 13.07 s |
| + TurboMind MoE, no metadata fix | 49.0 / 59.1 | 49.0 / 55.1 | 0.70 / 10.36 s |
| + metadata fix, skinny MoE | 62.3 / 63.8 | 57.2 / 58.4 | 1.03 / 13.07 s |
| + metadata fix, TurboMind MoE | 83.0 / 88.9 | | 0.83 s |
| + grouped decode v2 | 92.6 / 95.9 | 85.8 / 90.9 | 0.69 / 10.36 s |
| **+ grouped decode v3 (final)** | **96.3 / 100.6** | **92.7 / 96.2** | **0.69 / 10.36 s** |
| v2 build with PP2 (`--pp`) | 83.7 / 90.6 | 76.0 / 78.7 | 0.87 / 11.63 s |
| v2 build with `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` | 87.1 / 91.8 | | |

The rows "+ metadata fix, TurboMind MoE" and "shared-experts stream off"
come from single 1,024-token runs; the v2 build measured 91.8 / 93.9 the
same way. The skinny row also carried the reverted output-token change,
which measured neutral. At 128K the v2 build decoded 53.8 tok/s and v3
77.2 tok/s. At 250K both builds decoded about 24-25 tok/s after a cold
prefill (v2: 24.8 tok/s, 396 s TTFT), which the next section explains.

TP2 beats PP2 at every length, so the serve script keeps TP2.

### Heat at 250K

During the 128K decode both GPUs held the 1,380 MHz SM clock at 70-74 C and
about 160 W. A 250K request is different. Its prefill keeps both GPUs at
their 82-83 C operating limit for six and a half minutes, at 1,230-1,300 MHz.
In the decode that follows, GPU 1 (`07:00.0`) has its HBM at 84-85 C, its
85 C maximum. The driver then drops its SM clock to 250-700 MHz (thermal
slowdown), while GPU 0 stays at 1,380 MHz. TP2 runs at the slower GPU's
pace, so decode right after a 250K prefill measures 24-25 tok/s with a
35-40 ms median ITL and a 77 ms p90. GPU 1's HBM runs 4-8 C hotter than GPU
0's under the same load.

Sent again from the prefix cache after a four-minute cool-down (TTFT
1.3 s), the same 250K context decodes at 63.7 tok/s, with a 16.65 ms median
ITL and both GPUs at 1,380 MHz. That is what the microbenchmark predicts:
10.3 ms at 4K plus 0.45 ms more for each of the 14 full-attention layers.

This is a cooling limit of the host, not of the software: more airflow over
the second card would lift long-context decode on this box.

### Quality

Quality on the final configuration matches the baseline:

- **Probes:** arithmetic 10/10, executed coding probes 5/5, coherent greedy
  output, needles at 32K and 128K pass, on both the v2 and the v3 build.
  The probe stage's "two greedy runs differ" warning shows up as on the
  baseline; see the prefix-cache note below.
- **Teacher forcing:** prompt logprobs against the skinny route differ by
  0.014 nats on average (968/975 top-1 agreement). Two runs of one build
  are bit-identical.
- **Kernel tests:** 195 decode-sink and Flash-V100 decode tests pass on the
  grouped kernels, including a grouped-against-per-head comparison over
  windows, partition sizes and lengths up to 8K.

### Repeated greedy requests and the prefix cache

A greedy request sent twice can differ the second time, when it hits the
prefix cache. The perf runs capture nine greedy prompts twice (128 tokens,
top-2 logprobs) and compare the chosen token's logprob between the cold and
the cached run:

- Prompts shorter than one 16-token KV block never hit the cache and repeat
  bit-identically: every logprob at every position, on every build.
- Prompts of 21 and 23 tokens (one cached block) drift from the first
  token, by 1e-4 to 1e-3 nats, growing to 0.03-0.15 over 128 tokens. Their
  tokens still match.
- The 4,190-token prompt drifts 0.03-0.05 nats at its first token. It
  repeats on some builds and diverges at a near tie on others: at token 11
  with a 0.40 margin (baseline), token 30 with a 0.002 margin (grouped v1)
  and token 7 with a 0.18 margin (final). The grouped v2 build repeated it.

The drift depends on the cache hit, not on the MoE route (skinny and
TurboMind) or the attention kernel (per-head and grouped). A cached request
recomputes only its uncached tail: a few rows through the dense, MoE and
all-reduce paths instead of the whole prompt, and attention over the paged
prefix instead of the request's raw K/V. FP16 rounding then differs at
the 1e-3 level, which flips greedy only where the top two tokens are close.
Nothing in these runs points at corrupted cache contents. A check that would
settle it: compare the cached request's first-token logprobs with a cold
request whose prompt shares the prefix but differs in its last block.

### Changes

In commit order on `claude/flint-perf`:

1. **Grouped-query D128 paged decode for sink layers**
   (`8d5ded61058a30f4839a4ad70895923db058dfcb`). One CTA per KV head and
   partition covers the eight query heads. The kernel writes the same
   partials and statistics as the per-head kernel, so the reducer still
   applies the sinks. Only the sink entry point (fp16 KV, D128, 8:1 GQA)
   takes it.
2. **Flash-V100 metadata binds its per-step policy reads once**
   (`9f98f4cac9834b45707c4f7214591cdf9a01521b`). This is the host fix.
3. **Grouped decode v2** (`7ef09434473940f93d818fda9b8258f76083ae0c`).
   Half-warp tokens with 16-byte K/V loads, and a reducer that skips
   partitions before the sliding window and splits the rest across the
   warps.
4. **The serve script defaults to the TurboMind MoE route**
   (`a14a870ccfcaac7f5a4f4f037fa80e0ef168d04b`):
   `--kernel-config.sm70_skinny_moe false`.
5. **Grouped decode v3** (`778a6332eaf7012f359742aaf6650a23de9189f4`).
   Each of the eight warps runs PV for all eight heads over its eighth of
   the partition, and a shared-memory pass sums the slices. This is the
   long-context step: 128K decode goes from 53.8 to 77.2 tok/s.

### Single-token NVFP4 GEMV

Item 1 of the remaining ideas, on `claude/flint-m1-fp4`. At decode batch 1
and 2, the dense ModelOpt W4A16 projections and the routed experts leave
TurboMind for two dedicated paths. Prefill (M > 2) keeps TurboMind. Both
paths read the TurboMind-prepared weights in place (`tm_weight` codes and the
fp16 `tm_scales` that already fold in `weight_scale_2` and Flint's
FP16-overflow rescale), so they add no weight copies, and both accumulate in
FP32.

- **Dense GEMV** (`nvfp4_tm_gemv_sm70.cu`, op
  `nvfp4_tm_gemv_dispatch_sm70_out`). Admitted at load only for ModelOpt
  W4A16 on SM70 and Flint's six per-rank shapes. The M <= 2 / TurboMind
  branch lives inside the op, so the compiled graph sees one call per
  projection. Kill switch: `VLLM_SM70_MODELOPT_NVFP4_GEMV=0`.
- **Routed-expert decode** (`nvfp4_moe_decode_qpn_sm70.cu`). Two kernels
  per layer instead of TurboMind's 5 (M=1) or about 10 (M=2): W13 with a
  fused SwiGLU, reading the expert ids on the device, then W2 with the
  top-k weighted reduce. Reductions run in a fixed order (no float
  atomics). A 384 KiB workspace per layer is the only extra memory. A
  load-time power-of-two pre-shift guards the fp16 scale product against
  overflow; on the real checkpoint no layer needs it (largest W2 scale
  1.18). Kill switch: `VLLM_SM70_FLINT_MOE_QPN_DECODE=0`.

Both switches are read when the layers load. Qwen3.x routes are keyed on
their own schemes and shapes and do not change.

Kernel microbenchmark (`benchmarks/kernels/benchmark_sm70_nvfp4_m1.py`; CUDA
graph, cold L2, median of 200 replays, median of 3 runs), us at M=1 / M=2,
new-path GB/s at M=1 counting 4.5 bits per weight:

| Shape (K x N), per layer | TurboMind | QPN2 shared | New | GB/s |
| --- | ---: | ---: | ---: | ---: |
| qkv 2048 x 2560 | 23.10 / 17.98 | 7.96 / 7.96 | 7.17 / 7.51 | 411 |
| o 2048 x 2048 | 16.76 / 16.76 | 7.07 / 7.17 | 6.61 / 6.89 | 357 |
| Shared gate_up 2048 x 1024 | 15.81 / 15.87 | 6.53 / 6.56 | 4.61 / 4.80 | 256 |
| Shared down 512 x 2048 | 9.09 / 9.09 | 3.71 / 3.78 | 3.78 / 4.10 | 156 |
| Dense gate_up 2048 x 6144 | 23.04 / 23.30 | 16.64 / 16.64 | 13.82 / 14.34 | 512 |
| Dense down 3072 x 2048 | 20.86 / 20.99 | 9.34 / 9.34 | 8.32 / 8.70 | 425 |
| Routed experts, top-6 | 40.83 / 57.22 | | 17.28 / 27.14 | 307 |

Summed per token (qkv and o x49, shared expert x47, dense MLP x2, routed
experts x47), the kernels go from 5.13 to 1.93 ms at M=1 and from 5.65 to
2.45 ms at M=2.

The 70%-of-HBM target was not reached. On this V100 a pure streaming-read
kernel over the same byte counts reaches only 258-698 GB/s for 0.66-7.9 MB,
and each kernel in a graph costs about 1.15 us before it does any work. The
new GEMV sits 1.2-2.5 us above that read-only floor per shape; with the
weights resident in L2 it is barely faster, so per-CTA latency, not HBM,
is the limit.

**The host became the limit.** With the new kernels the decode graph takes
7.55 ms of GPU time, but each worker spent 8.0-9.5 ms of host time in
`execute_model` plus about 1.4 ms in `sample_tokens`. Step time then
followed the CPU clock (1.7-3.0 GHz on this host): runs landed anywhere
from 9.0 to 11.5 ms per token, the GPU sat idle 2-4 ms per step, and the
ranks drifted up to about 5.5 ms apart. The old path's 9.6 ms graph had
hidden this. Most of the host time is attention metadata, about 1.2 ms for
each of the four KV-cache groups. Two per-step calls ran without an engine
config in scope and rebuilt a policy from the environment on every call:
`configured_smallq_max_query_len()` (0.77 ms, once per group) and
`_should_record_sm70_sample_hidden_ready_event()` (0.27 ms). Both now
borrow the policy the engine already resolved from `vllm_config` and fall
back to the old lookup without one. Host work is now about 5-6 ms per step
against 7.5 ms of GPU work, so the next GPU saving will meet the host
again; the remaining metadata cost is the next target.

End to end (single stream, random-token prompts, 256 tokens, median of 3
unless noted). The before columns are this branch with both kill switches
at `0` in the same tree (old path) and the doc's earlier result:

| Context, sampling | Old path | New path | ITL new | TTFT |
| --- | ---: | ---: | ---: | ---: |
| 4K, default | 96.7 tok/s (10.34 ms) | **118.5 tok/s** | 8.46 ms | 0.69 s |
| 4K, greedy | 100.3 tok/s (9.97 ms) | **124.0 tok/s** | 8.09 ms | 0.69 s |
| 32K, default (1 run) | 92.7 tok/s (doc) | **110.4 tok/s** | 9.07 ms | 10.37 s |

That is +23% (default) and +24% (greedy) at 4K, and +19% at 32K, where
attention keeps the step GPU-bound. TTFT does not change: prefill keeps
TurboMind. The TTFT column comes from the run before the host fix, with the
same kernels.

Correctness:

- GPU unit tests compare every shape at M=1 and M=2 against an FP32
  dequantized reference and against TurboMind, including CUDA-graph
  replay; the 1-200-token MoE reference test passes.
- On real Flint weights (layers 0, 5 and 22, float64 reference), the new
  paths are exactly as accurate as TurboMind: median relative error
  2.1-3.3e-4 dense and 7.4-8.3e-4 for the MoE block, for both. They differ
  from TurboMind by about 1e-5 (dense) and 1e-4 (MoE). The MoE path rounds
  each expert's output to fp16 before the weighted sum, as TurboMind does.
- Greedy (9 prompts x 128 tokens, old vs new path): 4 of 9 identical,
  including the 4K prompt. Mean logprob drift is 0.011 (median 0.0003,
  max 0.69). Dense-only and MoE-only servers each drift about as much as
  both together, and all flips but one are near-ties (top-2 margin
  <= 0.31); the exception follows already degenerate text.
- TurboMind itself is not reproducible across server starts: it times
  tactics live at M <= 64, and five fresh processes gave 2-3 distinct
  outputs for qkv at M=1/2 and for the MoE at M=1. The coordinated tuning
  also logs "no LUT; rank 1 is falling back to local tuning". The new
  decode paths give one result every time.
- Quality probes: math 10/10, code 5/5, needle at 32K 3/3.

### GPU decode fusions

With the GEMVs in place, the step went back to the GPU. Rank 1 at 4K on
`5fc02fa` (torch profiler, ms per step): the graph is busy 6.96 ms of an
8.06 ms span, eager kernels take 0.85 ms and the GPU is idle 1.25 ms, for a
9.05 ms step. The largest pieces:

| Component | ms/step |
| --- | ---: |
| Router: fp32 cast 0.31, cuBLAS GEMV 0.41, split-K 0.14, topkGating 0.46 | 1.32 |
| Attention: partition kernel 1.43, reduce 0.21 | 1.64 |
| Dense GEMVs | 1.37 |
| All-reduce, 99 x 11 us | 1.09 |
| Routed experts | 0.96 |
| RMSNorm: q/k 98 x 3.5 us, fused add 98 x 3.5 us | 0.69 |
| lm_head, eager | 0.41 |

The eager sampler also runs two full-vocab fp32 softmaxes, 98 us each. The
changes below each have a `VLLM_SM70_FLINT_*` kill switch, read at load and
on by default unless noted.

1. **Fused router** (`VLLM_SM70_FLINT_FUSED_ROUTER`). For 1-2 tokens, one
   launch per MoE layer replaces the fp32 cast, the cuBLAS GEMV, the split-K
   reduce and topkGating. It reads fp16 activations and the FP32 gate as
   loaded, accumulates in FP32 and keeps the sigmoid top-6 tie-breaking.
   20.61 -> 8.32 us per layer at M=1 (25.60 -> 7.81 at M=2), 0.58 ms per
   step (0.84 at M=2). On real weights (layers 10-48, 2,000 vectors per
   distribution) there are no hard flips and the largest weight error is
   4.2e-7; one near-tie flips on layer 40.
2. **One-launch grouped sink decode** (`VLLM_SM70_FLINT_ATTN_FUSED_DECODE`).
   The partition grid was sized for the 262K capture envelope, so at 4K
   almost every CTA exited at once. The new kernel fixes about 80 CTAs per
   KV head by launch shape, chunks the window on the device (64-256 tokens)
   and lets the last CTA combine the partials in index order and apply the
   sinks. About 0.43 ms per step at 4K; 200 decode-sink tests pass.
3. **q/k RMSNorm, RoPE and KV-cache write in one Triton kernel**
   (`VLLM_SM70_FLINT_QK_ROPE_KV`). 2.25-2.5 us per layer against 9.3-10.3
   (sliding, with RoPE) and 6.2-6.9 (full): about 0.3 ms and 130 kernels
   fewer per step. It also runs in prefill. The cache layout is
   bit-identical; q and k are within 1 fp16 ulp.
4. **Vocab-parallel candidate sampler** (`VLLM_SM70_FLINT_SAMPLER`). Each
   rank takes its top 32 fp16 logits and the ranks all-gather 256 B per row.
   One kernel then applies temperature, top-k (ties at the k-th value kept),
   top-p in the reference order and the exponential-race draw with the same
   Philox offsets. GPU time per step drops from 581-759 to 50 us at batch 1
   (1,449-1,846 to 48 at batch 2). Greedy keeps the argmax path.
5. **Single-launch routed-expert decode** (`VLLM_SM70_FLINT_MOE_PERSISTENT`,
   off). Bitwise equal to the two kernels but not faster: 18.44 -> 19.07 us
   at M=1, 27.52 -> 27.39 at M=2. Likely cause: 80 registers x 384 threads
   allow 24 warps per SM in W13 instead of 32, and W2 blocks start only
   after every W13 block is dispatched.
6. **All-reduce fused with residual add and RMSNorm**
   (`VLLM_SM70_FLINT_AR_RMSNORM`, off in this release), through the
   all-reduce + RMSNorm compile pass. Bitwise equal. The microbenchmark
   saves 1.71 us per boundary at 1 row (3.47 on MoE boundaries, where it
   also absorbs the shared + routed expert add), 0.90 / 2.76 at 2 rows. The
   first version crashed the server at FULL-graph capture: it registered its
   expert-output operands for peer IPC, which expandable segments cannot
   export (`cudaIpcGetMemHandle: invalid argument`). It now stages its
   inputs into the communicator's peer-readable buffer and registers nothing
   at capture. End-to-end validation of this path is pending.

Per-layer attention, two launches -> fused, us:

| Layer | 4K | 32K | 128K | 250K |
| --- | ---: | ---: | ---: | ---: |
| Full | 25.3 -> 18.8 | 75.6 -> 70.1 | 249.2 -> 230.9 | 459.6 -> 421.4 |
| Sliding 512 | 22.8 -> 13.0 | | | 29.5 -> 13.1 |

The sampler is admitted for batch 1-2, all-random requests, top-k <= 20 and
no logprobs, penalties, logits processors, speculative decoding or
structured output; seeded requests only at batch 1. If more than 32 logits
on one rank tie at or above the k-th value, the extra ties cannot be
sampled. A device counter records that and the server logs a warning; no
serving run has hit it.

End to end (single stream, random-token prompts, 256 tokens, median of 3).
Before is `5fc02fa`; after is `claude/flint-gpu-opt` with `main` merged, including the
host-metadata fix `2f9170d`, with switches 5 and 6 off:

| Context, sampling | Before | After | ITL before / after |
| --- | ---: | ---: | ---: |
| 4K, default | 118.6 tok/s | **166.4-167.7 tok/s** | 8.45 / 6.00 ms |
| 4K, greedy | 124.1 tok/s | **167.3 tok/s** | 8.09 / 6.00 ms |
| 32K, default | 112.9 tok/s | **155.5 tok/s** | 9.06 / 6.64 ms |
| 32K, greedy | 118.2 tok/s | **155.1 tok/s** | 8.69 / 6.65 ms |

That is +40% (default) and +35% (greedy) at 4K, +38% and +31% at 32K.
Default and greedy now decode at the same rate. TTFT does not change
(0.68 s at 4K, 10.3 s at 32K). Runs still vary with the host CPU clock: one
greedy 4K session gave 167.3 / 155 / 146 tok/s, a rerun 162.4 / 167.5 /
167.3.

Quality:

- Probes: math 10/10, code 5/5, needle at 32K 3/3 (depths 0.1, 0.5, 0.9).
  The "two greedy runs differ" warning shows up as on the baseline.
- Greedy (9 prompts x 128 tokens) against the old paths in the same tree:
  6 of 9 identical; the flips are near-ties (top-2 margins 0.0098, 0.066,
  0.198).
- A flip at margin 2.23 against `5fc02fa` does not come from these paths:
  with all six switches off the new tree reproduces it at the same token
  and margin. It comes from server-start variance (TurboMind's live tactic
  timing at M <= 64, which moves the short prompts, and the `main` merge).
  Within one server start, greedy repeats.

### Remaining ideas, by expected gain

Done since the first version of this list: the TurboMind M=1 GEMVs (see
[Single-token NVFP4 GEMV](#single-token-nvfp4-gemv)), the host-side
attention metadata (`2f9170d`), the fused router and the vocab-parallel
sampler (see [GPU decode fusions](#gpu-decode-fusions)). The persistent
routed-expert kernel was built and is no faster.

1. **All-reduce + RMSNorm.** About 0.25 ms per step. The path is built
   (switch 6 above); it needs end-to-end validation, then default on.
2. **Prefill.** The largest cost of a long request: TTFT is 10.4 s at 32K,
   113 s at 128K and 398 s at 250K, against 3-11 s of decode for 256
   tokens. At 250K it also heats GPU 1 into thermal slowdown for the decode
   that follows. Sink layers bypass the tuned Flash-V100 prefill routes
   (split-KV, dense, small-query). The D128 paged prefill with sinks could
   get a grouped-query, split-KV path, as decode now has.
3. **Long-context attention.** About 10% at 128K and 15% at 250K. The v3
   kernel reaches about 510 GB/s at 128K; the 14 full layers took 3.7 ms
   per token there and 6.7 ms at 250K, and the one-launch kernel trims only
   7-8% of that. More loads in flight or a two-stage K/V prefetch could
   bring it closer to 800 GB/s.
4. **Host time.** The step ITL is now 6.0 ms at 4K. Whether host work is
   again close to the GPU time has not been measured since the fusions;
   profile both before the next GPU change.
5. **Reproducible TurboMind tactic selection.** Now the main source of
   greedy drift across server starts: a shared LUT across ranks and warm-up
   for prefill lengths 3-15.
6. **Single-launch routed-expert decode.** Fit three blocks per SM
   (<= 56 registers) so W13 keeps 32 warps, or drop the kernel.
7. **Dense GEMVs.** Close to the floor measured above; little is left.
8. **Speculative decoding.** n-gram or suffix drafts for agent workloads.
   The sink path rejects speculative configs today. The D128 verify path
   goes through the same scalar sink-aware decode kernel and would need the
   grouped kernel and a GPU check of verify correctness with sinks first.

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
  rounding, and the FP16 audit is the gate for that. The model's
  [FP16 headroom](#fp16-headroom) rescaling covers the layer-46 massive
  activations measured on the audit prompts (16.8x headroom left). Rerun the
  audit with representative prompts after any checkpoint change.
- Long prompts are slow to prefill on Volta: about 1,000 tokens/s at 128K and
  560 tokens/s at 250K on the sink route.
- `tool_choice="none"` drops the tools from the prompt and never returns
  `tool_calls`, but the model can still write raw `<tool_call>` markup into
  `content` when the system prompt tells it to use tools. Clients that need
  plain text can send `logit_bias` `{"150010": -100}`.
- With prefix caching, a repeated greedy request may take a different prefill
  route than the first one and diverge after some tokens. `TRITON_ATTN`
  repeated identically in the probes.
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
