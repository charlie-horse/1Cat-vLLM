# V100 model playbook: bring-up and decode optimization

A repeatable procedure for taking a new LLM onto NVIDIA V100 (SM70) with this
fork, then making single-stream decode fast. It is distilled from the Flint
port (39.8B MoE, NVFP4, 2× V100-PCIe-32GB, TP2), which went from a first
validated run to roughly 3× its baseline decode rate in two days. The full
record, with every measurement, is
[`docs/design/sm70_flint_tp2_pcie.md`](../design/sm70_flint_tp2_pcie.md).

Read the [checklist](#one-page-checklist) first if you have done this before.

Conventions:

- Paths are relative to the repository root.
- Commands use `.venv/bin/python` per `AGENTS.md`. On the NixOS reference
  host the same commands run through a `.venv/bin/python` shim (see
  [Phase 5](#phase-5-deploy)).
- Anything marked **(example: c4130)** is specific to the reference host:
  2× V100-PCIE-32GB on one PCIe switch, 48 CPUs, 125 GB RAM, no swap,
  NixOS. Substitute your own values.

## The Flint arc in one table

The order of the wins is the main lesson. Each row is single stream, 4K
context, default sampling, TP2.

| Step | 4K decode | What moved it |
| --- | ---: | --- |
| First validated run (skinny MoE, per-head attention) | 52.9 tok/s | |
| Grouped-query decode kernel, metadata host fix, TurboMind MoE | 96.3 tok/s | Kernel count 2,466 → 1,197 per step; host policy rebuild removed |
| Single-token NVFP4 GEMV and two-kernel routed experts | 96.7 tok/s | Kernel time 5.13 → 1.93 ms/token, **but no e2e change: host-bound** |
| Borrow engine policies on the decode host path | 118.5 tok/s | Host step time cut; GPU idle gaps closed |
| Metadata trim (host 1.1–1.7 → 0.75 ms/step) | 118.5 tok/s | No change: GPU-bound again; buys slack |
| Profile-driven GPU round (router, attention grid, AR+norm, fusion, sampler) | ~167.7 tok/s | Per-kernel profile of the current tip |

Two things cost the most time. A GPU optimization landed on a host-bound
step and showed nothing, and hand estimates of where GPU time went were
wrong until a fresh per-kernel profile replaced them.

## Phase 0: qualify the model

Answer these before writing code. Each takes minutes; a wrong answer costs
hours later.

### Hardware and toolchain facts

| Fact | Consequence |
| --- | --- |
| SM70 has no BF16 | Everything runs FP16 (`--dtype half`). BF16-trained models can overflow; see [FP16 range](#fp16-range) |
| CUDA 13 cannot target SM70 | Build with CUDA 12.8, Torch 2.10.0+cu128 (`--torch-backend=cu128`), `TORCH_CUDA_ARCH_LIST=7.0` |
| E4M3 does not compile in Triton on SM70 | Keep the KV cache FP16 (`--kv-cache-dtype auto`) unless a non-Triton FP8 route covers every layer |
| 900 GB/s HBM2, 6 MB L2, 80 SMs | Batch-1 decode is bandwidth and launch-latency bound. Each kernel in a graph costs about 1.15 µs before it does work |
| PCIe-only pairs | TP2 beat PP2 at every length for Flint. Use the real peer-write check `VLLM_SKIP_P2P_CHECK=0`; the driver's P2P report is not enough |
| Host CPU clock (1.7–3.0 GHz on the example host) | Batch-1 step time can follow the CPU clock. Check host vs GPU time ([Phase 2](#phase-2-measure)) |

### Weight format

Pick a format the fork already runs on SM70. Proven so far:

- **ModelOpt W4A16 NVFP4** (Flint): dense layers and MoE through TurboMind
  (`ModelOptNvFp4SM70MoEMethod`, `nvfp4_sm70_moe.py`) or the skinny QPN
  route; single-token GEMV and routed-expert decode at M ≤ 2.
- **FP8** (Qwen3.6-35B-A3B-FP8), with a DFlash2 speculative draft.
- AWQ, GDN and DeepSeek-V4 paths exist (see `tests/kernels/test_sm70_*`);
  check their admission before relying on them.

If the published checkpoint is in another container (Flint shipped as an
NVFP4 GGUF), plan a one-time offline converter to a format above rather than
a new loader. Check the MoE contract too: TurboMind admits only shapes listed
in `_SUPPORTED_CONTRACTS` (Flint added `(2048, 512, 256, 6)`).

### FP16 range

A BF16-trained model may carry massive activations above 65504. Flint did,
in two places of layer 46. You cannot know until you run the audit
([Phase 1](#fp16-range-audit)), but budget time for a rescale if:

- the model card or config mentions BF16-only training, or
- it has very deep residual streams or large MoE intermediates.

The fix is always a power-of-two shift (exponent-only, exact) at load time.

### Memory budget

Do the arithmetic per GPU before any download. Flint at TP2:

| Item | Formula | Flint |
| --- | --- | --- |
| Quantized weights | params × bits / 8 / TP (NVFP4: 4.5 bits with the E4M3 scale) | ≈ 11 GB |
| FP16 embedding + LM head | 2 × vocab × hidden × 2 B / TP | ≈ 0.6 GB |
| Usable GPU memory | 31.7 GiB × `--gpu-memory-utilization` (0.90) | 28.6 GiB |
| Full-attention KV per token | layers × local KV heads × head dim × 2 × 2 B | 14 KiB |
| Sliding-window KV | bounded by window + scheduled tokens | ≈ 0.3 GiB per request |
| Activations, graphs, collectives | measure at startup | ≈ 0.6 GB |
| **Host RAM at load** | converter or loader peak, plus everything else on the box | 24.7 GB convert, ~27 GB vLLM load |

The host side is easy to forget. On the example host the first production
switch hung the machine: LMCache's 96 GB pinned L1 plus vLLM's ~27 GB load
peak exceeded 125 GB with no swap. Budget host RAM as
`LMCache L1 + vLLM load peak + converter (if run on the box) + OS ≤ RAM`.

## Phase 1: bring-up

### Conversion

Write the converter as a module under `vllm/models/<model>/` so it is
versioned with the loader that reads its output. Flint's:

```bash
.venv/bin/python -m vllm.models.flint.convert_gguf SRC.gguf OUT_DIR \
  --hf-config RecursiveSI/flint
```

Make it:

- deterministic (byte-identical output, so a Nix build step can cache it),
- strict (assert no BF16 survives; every cast checks `|x| <= 65504`),
- checkable from headers alone: dtypes, expert count, scale dtypes, the
  quantization config. `tools/flint/sm70_validate.sh --stage convert` does
  this for Flint; copy its `cmd_checkpoint` check.

Keep load-time transforms (FP16 rescale, LM-head scale folding) in the model,
not the converter, so a fix does not force a reconversion of tens of GB.

### Model and routes

1. Implement the model on stock vLLM layers first (`139e38b` for Flint).
2. Route every special feature to a backend that implements it. Flint's
   learned attention sinks needed sink support in Flash-V100 prefill and
   decode; without it the model falls back to `TRITON_ATTN`, whose prefill
   runs FP32 FMA on Volta.
3. Log the chosen route at startup and expose it (`/v1/sm70/acceleration`).
   Save it with every benchmark: a silent fallback looks like a slow kernel.

### Serve profile

Put the whole profile in one script, `scripts/serve_<model>_v100_pcie.sh`,
modelled on `scripts/serve_flint_nvfp4_v100_pcie.sh`:

- `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `VLLM_SKIP_P2P_CHECK=0`
- `--dtype half --kv-cache-dtype auto --gpu-memory-utilization 0.90`
- the MoE route as an explicit flag (`--kernel-config.sm70_skinny_moe false`)
- parsers, served name and generation defaults
- a `--pp` switch for the TP-vs-PP comparison

### Correctness gates

Run these in order; each gate is cheaper than the next.

1. **Topology:** `tools/flint/sm70_validate.sh --stage topo`. Two SM 7.0
   GPUs, IPC P2P test passes, custom all-reduce correct.
2. **Kernel tests** against a torch reference for every new route
   (`--stage kernels`). For a new contract, add a test like
   `tests/kernels/moe/test_sm70_nvfp4_contract_flint.py`.
3. <a id="fp16-range-audit"></a>**FP16 range audit**, with your real serve
   flags:

    ```bash
    .venv/bin/python -m tools.flint.fp16_range_audit --model DIR \
      --tensor-parallel-size 2 --dtype half --max-model-len 32768 \
      --output fp16_range_audit.json
    ```

    Exit 0 means no layer above 30000. It is model-agnostic (`--layers-path`
    for unusual layouts). On overflow, find the op (Flint: `silu(gate) * up`
    in two experts, then the residual stream from layer 46), shift by a power
    of two at load time (`_rebalance_moe_intermediate`,
    `_rescale_fp16_stream` in `vllm/models/flint/model.py`), and re-run until
    headroom is ≥ 10×.
4. **In-engine layer checks** against an FP32 dequantized reference: attention
   and MoE outputs of a few layers on both ranks. Flint: attention 2e-4
   relative, MoE median 8e-4.
5. **Smoke, quality and agent:** `--stage smoke,quality,agent` against a
   running server. Thresholds: arithmetic ≥ 8/10, coding ≥ 4/5, every needle
   found at 32K/128K/~250K, tool calls parse, `cached_tokens` > 0 on later
   turns.

The runbook pattern generalizes: one script, staged, each stage writing a log
and a row in `summary.md` under `bench_results/<model>_sm70/<timestamp>/`.
Copy `tools/flint/sm70_validate.sh` and swap the model-specific stages.

## Phase 2: measure

Never optimize from estimates. Flint's hand estimates of GPU time were wrong
by enough to reorder priorities.

### Baseline end to end

Single stream, random-token prompts to `/v1/completions` (no prefix-cache
hits), 256 tokens with `ignore_eos`, one warm-up then the median of three.
Decode tok/s is `(tokens - 1) / (last token time - first token time)`.
Measure 4K and 32K for the short loop; 128K and 250K only at milestones,
because a 250K prefill takes minutes and heats the GPUs into thermal
slowdown.

Report default sampling and greedy; greedy is 0.3–0.5 ms/token faster, so
never compare across them.

### Host vs GPU per step

Do this **before and after every GPU change**. It is the step that would have
saved the most time on Flint.

1. Capture a short decode window with the torch profiler
   (`--profiler-config`, see [profiling](profiling.md)).
2. Per rank, compare GPU busy time per step (sum of the decode graph's
   kernels) with the step span (start of one graph launch to the next).
3. Look for idle gaps between graph replays and for one rank spinning in the
   first all-reduce of the step: that is rank skew from host work, not
   communication cost.
4. Sample the worker's Python stacks (e.g. `py-spy dump` on the worker PIDs)
   to name the host hot spots.

If `step span − GPU busy` is more than about 10% of the step, you are
host-bound, and a GPU optimization will not show end to end. A tell-tale
symptom: step time that drifts with the CPU clock (Flint: 9.0–11.5 ms/token
on the same build).

Check that the decode really replays one FULL CUDA graph per step with
`--cudagraph-metrics`: no piecewise or eager fallback at batch 1 and 2.

### Per-kernel GPU profile

From the same trace, group the decode graph's kernels into classes and fill a
table like Flint's [per-step breakdown](../design/sm70_flint_tp2_pcie.md#per-step-breakdown-at-4k):
dense GEMVs, routed experts, routing glue, router, attention, all-reduce,
norms and elementwise, inter-kernel gaps, LM head, sampler. Also record
kernels per step: at ~1.15 µs launch cost each, 1,000 kernels is over 1 ms.

The torch profiler adds about 1 µs per kernel, so the rows sum to more than
the unprofiled step. Use it for ranking, and the unprofiled ITL for totals.

Redo the profile on the current tip after each landed optimization. The
ranking changes.

### Microbenchmark harness

For a kernel class worth attacking, write a microbenchmark on the real
per-rank shapes, modelled on `benchmarks/kernels/benchmark_sm70_nvfp4_m1.py`:

- capture each call in a CUDA graph,
- rotate several weight copies so consecutive calls miss the 6 MB L2,
- queue replays behind a GPU sleep so CUDA events measure device time,
- report the median of many replays, median of three runs, and GB/s against
  the bytes touched,
- run old and new paths by flipping the kill switches at layer load, in one
  process.

```bash
.venv/bin/python benchmarks/kernels/benchmark_sm70_nvfp4_m1.py \
  --backends turbomind,qpn2_shared,new --json out.json --markdown out.md
```

Also measure a pure streaming-read kernel over the same byte counts. On the
example V100 it reaches only 258–698 GB/s for 0.66–7.9 MB, so 70% of HBM peak
is not a realistic floor for small decode GEMVs.

## Phase 3: optimize

### Decision tree

```text
Is step span - GPU busy > ~10% of the step?
├─ yes → HOST-BOUND. Fix host first.
│        Stack-sample the workers. Usual suspects on this fork:
│        - policies/configs rebuilt from env vars per step or per KV group
│        - attention metadata builds per KV-cache group
│        - per-step Python allocations in the sampler/output path
│        Re-measure: does tok/s move? If not, you are GPU-bound now.
└─ no  → GPU-BOUND. Rank kernel classes from the fresh profile:
         ├─ many tiny kernels / gaps   → fuse (routing glue, elementwise,
         │                               all-reduce + RMSNorm, router)
         ├─ attention                  → check the grid matches the live
         │                               context, not the capture envelope;
         │                               GQA grouping; window-aware reducer
         ├─ quantized GEMV at M ≤ 2    → dedicated GEMV reading the loaded
         │                               weights in place
         ├─ MoE experts                → fewer kernels per layer (W13+SwiGLU,
         │                               W2+top-k reduce)
         └─ sampler                    → top-k before top-p; avoid full-vocab
                                          sort
```

Before building a kernel, bound its gain: `(current µs - streaming-read
floor µs) × calls per token`. If that is under about 2% of the step, skip it.
Flint's MoE persistent kernel was estimated at 3–5 µs per layer and gave no
measurable gain; it ships switched off.

### What moved Flint, by kernel class

| Class | Change | Saving |
| --- | --- | --- |
| Routing glue | TurboMind route instead of skinny (25 small ops/layer) | ~4 ms/token; kernels 2,466 → 1,197 |
| Attention, long context | Grouped-query D128 decode, half-warp tokens, window-aware reducer, PV split over warps | 128K: 26.9 → 77.2 tok/s |
| Dense NVFP4 at M ≤ 2 | Single-token GEMV (`nvfp4_tm_gemv_sm70.cu`) | Part of 5.13 → 1.93 ms kernel time |
| Routed experts at M ≤ 2 | Two-kernel decode (`nvfp4_moe_decode_qpn_sm70.cu`) | Part of the same |
| Host | Borrow engine policies (`0b731f4`) | 96.7 → 118.5 tok/s |
| Router | One fused launch for cast, GEMV, split-K and top-k | ~0.58 ms/token |
| Attention, short context | Grid sized for the live context, not the 262K capture envelope | ~0.43 ms/token |
| All-reduce + RMSNorm, elementwise, sampler | Fusions | Smaller; see the design doc |

### Rules for every change

1. **Kill switch.** A `VLLM_SM70_*` flag registered in `vllm/envs.py` that
   restores the previous route, read at layer load (e.g.
   `VLLM_SM70_MODELOPT_NVFP4_GEMV`, `VLLM_SM70_FLINT_MOE_QPN_DECODE`).
   Admission is narrow: exact scheme and shapes, so other models' routes do
   not change.
2. **Reference test.** GPU unit tests compare against an FP32 (or FP64)
   dequantized reference **and** against the old path, at every shape and M,
   including CUDA-graph replay. Flint: new paths as accurate as TurboMind
   (median relative error 2–3e-4 dense, 7–8e-4 MoE). Use fixed-order
   reductions, no float atomics, so the new path is reproducible.
3. **No extra weight copies.** Read the weights the existing route already
   prepared, in place.
4. **Same-tree A/B.** One build, two servers: kill switch off, then on.
   Never compare against a number from another tree or another day; the CPU
   clock and GPU temperature move results by 10–20%.
5. **Re-check host vs GPU** after the change lands.

## Phase 4: validate

### Greedy ablation

1. Run nine greedy prompts × 128 tokens with top-2 logprobs on the old path
   (all new switches off) and the new path.
2. Run the old path twice in fresh processes. TurboMind times tactics live at
   M ≤ 64, so the old path itself gives 2–3 distinct outputs across starts.
   That run-to-run spread is the noise floor; judge drift against it, not
   against zero.
3. For each flip, check the top-2 margin at the flip point. Near-ties
   (Flint: ≤ 0.31) are FP16 rounding, not bugs.
4. Attribute drift per switch: one switch off per run. If each switch alone
   drifts about as much as all together, nothing is wrong in one of them.
5. Keep prompts shorter than one 16-token KV block for exact repeats; longer
   prompts that hit the prefix cache drift by 1e-3 nats even on one build.

### Quality probes

`tools/flint/sm70_validate.sh --stage quality,agent --server-url URL` on the
final configuration: Flint scored arithmetic 10/10, coding 5/5, needle at
32K 3/3 (and 128K/250K at milestones). These catch broken numerics and
parsers, not small accuracy regressions.

## Phase 5: deploy

The reference host runs NixOS **(example: c4130)**.

- **Build the fork from source in Nix.** No system Python or compilers:
  build in place (`python setup.py build_ext --inplace` in
  `flash-attention-v100/` and the root, `TORCH_CUDA_ARCH_LIST=7.0`,
  `CMAKE_CUDA_ARCHITECTURES=70`), about 25 minutes with `MAX_JOBS=24`. A
  `.venv/bin/python` shim puts the tree first on `PYTHONPATH`.
- **Make conversion a Nix build step.** The converter output is
  byte-identical, so the derivation caches it. Pin the converter's source
  separately from the fork, so bumping the fork does not reconvert 23 GB.
- **RAM budget.** Size LMCache's pinned L1 from the Phase 0 host formula.
  Flint: L1 cut from 96 GB to 64 GB.
- **RAM guard.** Wrap risky starts (first boot of a new model, conversion)
  in a guard unit that refuses to start without enough available memory.
  With no swap, an overcommit hangs the box before the OOM killer acts.
- **LMCache and expandable segments.** Unset
  `PYTORCH_ALLOC_CONF=expandable_segments:True` when LMCache attaches. With
  it, CUDA IPC uses `pidfd_getfd`, which `kernel.yama.ptrace_scope=1` denies;
  vLLM's KV registration then times out.
- **Cooling.** A 250K prefill holds both GPUs at 82–83 °C for minutes, and
  the hotter card's HBM then throttles the decode that follows (24 vs
  63.7 tok/s cooled). TP runs at the slower card's pace.

## Phase 6: coordinate agents and sessions

Several agents shared one GPU box. What worked and what did not:

- **GPU arbitration through one coordinator.** A `flock` convention failed
  because the production systemd service ignored it. An explicit protocol
  worked: an agent sends `REQUEST` with what it will run and for how long;
  the coordinator stops the production service and replies `GRANT`; the agent
  replies `RELEASE` when its GPUs are free; the coordinator restores the
  service. Nobody touches the GPUs without a grant.
- **One remote work tree per agent.** Concurrent builds in one tree collide.
  Do not copy build trees: CMake caches keep pointing at the old path.
  Clone fresh, or delete `build/` and `CMakeCache.txt` in the copy.
- **Commit WIP early to local branches.** A long-running agent was lost to an
  accidental stop and its uncommitted work had to be rescued by hand.
- **Batch GPU questions.** An agent in a GPU window should get one batched
  message, not a stream of interruptions.
- **A/B is the unit of evidence.** A rule of "exactly two e2e measurements"
  was too strict once a result surprised; allow a same-tree A/B with kill
  switches whenever a number needs explaining.

## What we would do differently

Ranked by time it would have saved on Flint:

1. **Check host vs GPU time before and after every GPU change.** The
   single-token kernels cut kernel time 62% and moved e2e by 0%; the host fix
   then gave +23%. A five-minute profile up front would have reordered the
   work.
2. **Profile the current tip per kernel before choosing work.** Hand
   estimates misranked the router, the attention grid and the MoE persistent
   kernel. The fresh profile found ~1 ms/token in two cheap fixes.
3. **Bound a kernel's gain against a streaming-read floor before building
   it.** The persistent MoE kernel cost a work session for no gain.
4. **Set up GPU arbitration, per-agent trees and WIP commits on day one.**
   Lock collisions, CMake cache confusion and a lost agent each cost time.
5. **Budget host RAM before the production switch.** The OOM hang and the
   LMCache attach failure were both predictable from the Phase 0 table.

## One-page checklist

Phase 0, qualify:

- [ ] Format has a proven SM70 route; MoE shape in `_SUPPORTED_CONTRACTS`
- [ ] No BF16-only op on the path; plan for an FP16 rescale
- [ ] Per-GPU memory: weights + KV at target context + 0.6 GB ≤ 28.6 GiB
- [ ] Host RAM: LMCache L1 + load peak + converter + OS ≤ RAM

Phase 1, bring-up:

- [ ] Deterministic converter under `vllm/models/<model>/`, header check
- [ ] Serve script with the full profile and a `--pp` switch
- [ ] `topo`, kernel tests, FP16 audit (≥ 10× headroom), smoke, quality, agent

Phase 2, measure:

- [ ] Baseline 4K/32K, default and greedy, median of 3, same tree
- [ ] Host vs GPU per step on both ranks; one FULL graph per step
- [ ] Per-kernel profile table and kernels per step
- [ ] Microbench of the top class with a streaming-read floor

Phase 3, optimize (per change):

- [ ] Gain bound ≥ 2% of the step
- [ ] `VLLM_SM70_*` kill switch in `vllm/envs.py`, narrow admission
- [ ] Unit test vs FP32/FP64 reference and the old path, with graph replay
- [ ] Same-tree A/B; host vs GPU re-checked

Phase 4, validate:

- [ ] Greedy old vs new, against the old path's own run-to-run noise
- [ ] Per-switch ablation of any flips; flips at near-ties only
- [ ] Quality probes pass

Phase 5–6, deploy and coordinate:

- [ ] Nix build, conversion as a cached build step, converter pinned
- [ ] RAM guard on risky starts; LMCache without expandable segments
- [ ] GPU grants through the coordinator; one tree per agent; WIP committed
