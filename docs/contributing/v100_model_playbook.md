# V100 model playbook: bring-up and decode optimization

A repeatable procedure for taking a new LLM onto NVIDIA V100 (SM70) with this
fork, then making single-stream decode fast. It is distilled from the Flint
port (39.8B MoE, NVFP4, 2× V100-PCIe-32GB, TP2). The full record, with every
measurement, is
[`docs/design/sm70_flint_tp2_pcie.md`](../design/sm70_flint_tp2_pcie.md).

If you have done this before, skip to the [checklist](#one-page-checklist).

Conventions:

- Paths are relative to the repository root.
- Commands use `.venv/bin/python` per `AGENTS.md`. On the NixOS reference
  host the same commands run through a `.venv/bin/python` shim (see
  [Phase 5](#phase-5-deploy)).
- Anything marked **(example: c4130)** is specific to the reference host:
  2× V100-PCIE-32GB on one PCIe switch, 48 CPUs, 125 GB RAM, no swap,
  NixOS. Substitute your own paths and numbers.

## The Flint arc in one table

The order of the wins is the main lesson. All rows are single stream, 4K
context, default sampling, TP2.

| Step | 4K decode | What moved it |
| --- | ---: | --- |
| First validated run (skinny MoE, per-head attention) | 52.9 tok/s | |
| Grouped-query decode kernel, metadata fix, TurboMind MoE | 96.3 tok/s | 2,466 → 1,197 kernels per step; one host policy rebuild removed |
| Start of day two, production build | 89.5 tok/s | Same code; CPU clock and run-to-run spread |
| Single-token NVFP4 GEMV and two-kernel routed experts | ~90 tok/s | Kernel time 5.13 → 1.93 ms/token, **but no e2e change: host-bound** |
| Borrow engine policies on the decode host path (`0b731f4`) | 118.5 tok/s | Host step time cut; GPU idle gaps closed |
| Metadata trim (host 1.1–1.7 → 0.75 ms/step, `2f9170d`) | 118.6 tok/s | No change: GPU-bound again; buys slack |
| Profile-driven GPU round, five of six paths on | 167.7 tok/s | Router, attention grid, q/k-norm+RoPE+KV, sampler. 32K: 153.7 tok/s |

The last row is interim. It is on `claude/flint-gpu-opt`, not on `main`, and
waits on a greedy ablation and the quality gate.

Two things cost the most time:

- A GPU optimization landed on a host-bound step and showed nothing. That
  cost about 1.5 hours and an extra build cycle.
- Hand estimates of where GPU time went were wrong until a fresh per-kernel
  profile of the current tip replaced them.

## Phase 0: qualify the model

Answer these before writing code. Each takes minutes; a wrong answer costs
hours later.

### Hardware and toolchain facts

| Fact | Consequence |
| --- | --- |
| SM70 has no BF16 | Everything runs FP16 (`--dtype half`). BF16-trained models can overflow; see [FP16 range](#fp16-range) |
| CUDA 13 cannot target SM70 | Build with CUDA 12.8, Torch 2.10.0+cu128 (`--torch-backend=cu128`), `TORCH_CUDA_ARCH_LIST=7.0`. Third-party CUDA packages may only ship CUDA 13 wheels (LMCache 0.5.6: build from sdist) |
| E4M3 does not compile in Triton on SM70 | Keep the KV cache FP16 (`--kv-cache-dtype auto`) unless a non-Triton FP8 route covers every layer |
| 900 GB/s HBM2, 6 MB L2, 80 SMs | Batch-1 decode is bandwidth and latency bound. Each kernel in a graph costs about 1.15 µs before it does any work |
| PCIe-only pairs | TP2 beat PP2 at every length for Flint. Use the real peer-write check `VLLM_SKIP_P2P_CHECK=0`; the driver's P2P report is not enough |
| Host CPU clock (1.7–3.0 GHz on the example host) | Batch-1 step time can follow the CPU clock and look bimodal. Check host vs GPU time ([Phase 2](#phase-2-measure)) |

### Weight format

Pick a format the fork already runs on SM70. Proven so far:

- **ModelOpt W4A16 NVFP4** (Flint): dense layers and MoE through TurboMind
  (`ModelOptNvFp4SM70MoEMethod`, `nvfp4_sm70_moe.py`) or the skinny QPN
  route, plus the single-token GEMV and routed-expert decode at M ≤ 2.
- **FP8** (Qwen3.6-35B-A3B-FP8), with a DFlash2 speculative draft.
- AWQ, GDN and DeepSeek-V4 paths exist (see `tests/kernels/test_sm70_*`);
  check their admission before relying on them.

If the published checkpoint is in another container (Flint shipped as an
NVFP4 GGUF), write a one-time offline converter to a format above rather
than a new loader. Check the MoE contract too: TurboMind admits only shapes
listed in `_SUPPORTED_CONTRACTS` (Flint added `(2048, 512, 256, 6)`).

### FP16 range

A BF16-trained model may carry massive activations above 65504. Flint did,
in two places of layer 46. You cannot know until you run the
[audit](#fp16-range-audit), but budget time for a rescale if the model was
trained in BF16 only, or has a deep residual stream or large MoE
intermediates. The fix is a power-of-two shift at load time: it moves
exponents only, so it is exact.

### Memory budget

Do the arithmetic per GPU and for the host before any download. Flint at
TP2:

| Item | Formula | Flint |
| --- | --- | --- |
| Quantized weights | params × bits / 8 / TP (NVFP4: 4.5 bits with the E4M3 scale) | ≈ 11 GB |
| FP16 embedding and LM head | 2 × vocab × hidden × 2 B / TP | ≈ 0.6 GB |
| Usable GPU memory | 31.7 GiB × `--gpu-memory-utilization` (0.90) | 28.6 GiB |
| Full-attention KV per token | layers × local KV heads × head dim × 2 × 2 B | 14 KiB |
| Sliding-window KV | bounded by window + scheduled tokens | ≈ 0.3 GiB per request |
| Activations, graphs, collectives | measure at startup | ≈ 0.6 GB |
| LMCache server, outside vLLM's budget | measure | ~2.2 GiB per GPU, so utilization ≤ 0.91 |
| **Host RAM** | LMCache pinned L1 + vLLM load peak RSS + ~30 GB < total RAM | 64 + 26.7 + 30 < 125 GB |

The host side is easy to forget. LMCache allocates its pinned L1 in full at
startup. At 96 GB L1 plus Flint's 27 GB load peak on a 125 GB host with no
swap, the page cache starved, startup crawled for 14 minutes, SSH and ping
died, and the OOM killer hit LMCache. Also budget the converter (Flint:
24.7 GB peak) if it runs on the same box.

## Phase 1: bring-up

### Conversion

Write the converter as a module under `vllm/models/<model>/`, so it is
versioned with the loader that reads its output. Flint's:

```bash
.venv/bin/python -m vllm.models.flint.convert_gguf SRC.gguf OUT_DIR \
  --hf-config RecursiveSI/flint
```

Make it:

- **deterministic:** byte-identical output, so a Nix build step can cache it.
  Keep reports with timestamps out of the output directory.
- **strict:** assert that no BF16 survives; every cast checks
  `|x| <= 65504`.
- **checkable from headers alone:** dtypes, expert count, scale dtypes and
  the quantization config. `tools/flint/sm70_validate.sh --stage convert`
  does this for Flint; copy its `cmd_checkpoint`.

Keep load-time transforms (the FP16 rescale, LM-head scale folding) in the
model, not the converter. A fix then needs a restart, not a reconversion of
tens of GB.

### Model and routes

1. Implement the model on stock vLLM layers first (`139e38b` for Flint).
2. Route every special feature to a backend that implements it. Flint's
   learned attention sinks needed sink support in Flash-V100 prefill and
   decode. Without it the model falls back to `TRITON_ATTN`, whose prefill
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
   GPUs, the IPC P2P test passes, custom all-reduce is correct.
2. **Kernel tests** against a torch reference for every new route
   (`--stage kernels`). For a new MoE contract, add a test like
   `tests/kernels/moe/test_sm70_nvfp4_contract_flint.py`.
3. <a id="fp16-range-audit"></a>**FP16 range audit**, with your real serve
   flags:

    ```bash
    .venv/bin/python -m tools.flint.fp16_range_audit --model DIR \
      --tensor-parallel-size 2 --dtype half --max-model-len 32768 \
      --output fp16_range_audit.json
    ```

    Exit 0 means no layer above 30000. The tool is model-agnostic (use
    `--layers-path` for unusual layouts). On overflow, find the op (Flint:
    `silu(gate) * up` in two experts, then the residual stream from layer
    46), shift it by a power of two at load time (`_rebalance_moe_intermediate`
    and `_rescale_fp16_stream` in `vllm/models/flint/model.py`), and re-run
    until the headroom is at least 10×.
4. **In-engine layer checks** against an FP32 dequantized reference:
   attention and MoE outputs of a few layers on both ranks. Flint: attention
   2e-4 relative, MoE median 8e-4.
5. **Smoke, quality and agent:** `--stage smoke,quality,agent` against a
   running server. Thresholds: arithmetic ≥ 8/10, coding ≥ 4/5, every needle
   found at 32K/128K/~250K, tool calls parse, `cached_tokens` > 0 on later
   turns.

The runbook pattern generalizes: one script, staged, each stage writing a log
and a row in `summary.md` under `bench_results/<model>_sm70/<timestamp>/`.
Copy `tools/flint/sm70_validate.sh` and swap out the model-specific stages.

## Phase 2: measure

Never optimize from estimates. Before each GPU session, check what is
already on the GPUs (`nvidia-smi --query-gpu=index,memory.used`). On Flint,
14 GB per GPU at the start of a session meant a server was still loading.

### Baseline end to end

- Single stream, random-token prompts to `/v1/completions`, so prefix
  caching never hits.
- 256 tokens with `ignore_eos`, one warm-up, then the median of three.
- Decode tok/s is `(tokens - 1) / (last token time - first token time)`.
- Measure 4K and 32K for the short loop. Run 128K and 250K only at
  milestones: a 250K prefill takes minutes and heats the GPUs into thermal
  slowdown.
- Report default sampling and greedy separately. Greedy is 0.3–0.5 ms/token
  faster, so never compare across them.

### Host vs GPU per step

Do this **at baseline and after every GPU change**. It is the check that
would have saved the most time on Flint.

1. Start the server with the torch profiler and graph metrics:

    ```bash
    scripts/serve_flint_nvfp4_v100_pcie.sh DIR --cudagraph-metrics \
      --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/prof","ignore_frontend":true}'
    ```

2. Send a 4K request and bracket about 1 s of decode with `POST
   /start_profile` and `POST /stop_profile`.
3. In the trace, group kernels by the correlation id of their
   `cudaGraphLaunch`: one replay is one decode step. Per rank, compute graph
   busy time (sum of kernel times), step span (first kernel of one replay to
   the first of the next), eager busy time, and idle time per step.
4. Sample the workers' Python stacks (for example `py-spy dump` on the worker
   PIDs) to name the host hot spots.

If idle time is more than about 10% of the step, you are host-bound, and a
GPU optimization will not show end to end. Tell-tale signs:

- step time drifts with the CPU clock (Flint: 9.0–11.5 ms/token on one
  build);
- one rank spends the gap spinning in the first all-reduce of the step. That
  is rank skew from host work, not communication cost.

`--cudagraph-metrics` also confirms that the decode replays one FULL graph
per step, with no piecewise or eager fallback at batch 1 and 2.

### Per-kernel GPU profile

From the same trace, sort the decode graph's kernels into classes and fill a
table like Flint's
[per-step breakdown](../design/sm70_flint_tp2_pcie.md#per-step-breakdown-at-4k):
dense GEMVs, routed experts, routing glue, router, attention, all-reduce,
norms and elementwise, inter-kernel gaps, LM head, sampler. Record kernels per
step too: at about 1.15 µs of launch cost each, 1,000 kernels is over 1 ms.

Pitfalls seen on Flint:

- The torch profiler adds about 1 µs per kernel. Use the profile to rank,
  and the unprofiled ITL for totals.
- Class regexes misfile kernels. The MoE `w2_reduce_kernel` was first counted
  as attention reduce. Print the unmatched and the top-20 kernel names.
- Per-call averages hide grid problems. The attention partition grid was
  sized for the 262K capture envelope, so at 4K almost every CTA exited at
  once. Compare each kernel's per-call µs in the profile with its
  microbenchmark at the live shape.

Redo the profile on the current tip after each landed change. The ranking
moves.

### Microbenchmark harness

For a kernel class worth attacking, microbenchmark the real per-rank shapes.
Model it on `benchmarks/kernels/benchmark_sm70_nvfp4_m1.py`:

- Capture each call in a CUDA graph.
- Make L2 cold: rotate several weight copies, or fill a buffer of 64 MB or
  more before each sample (V100 L2 is 6 MB).
- Record CUDA events outside the graph, around `replay()`. Events recorded
  inside a capture break. Queue replays behind a GPU sleep so the events
  measure device time, not launch gaps.
- Report the median of 200 replays, then the median of 3 runs, and GB/s
  against the bytes touched.
- Kill switches are read at layer load, so **build fresh layers per backend**
  under that backend's switch values. Flint's first harness measured the new
  path twice before this was fixed.

```bash
.venv/bin/python benchmarks/kernels/benchmark_sm70_nvfp4_m1.py \
  --backends turbomind,qpn2_shared,new --json out.json --markdown out.md
```

The shapes are hard-coded to Flint (`DENSE_SHAPES`, `MOE_*`); swap them for
the new model. Also measure a pure streaming-read kernel over the same byte
counts. On the example V100 it reaches only 258–698 GB/s for 0.66–7.9 MB, so
70% of HBM peak is not a realistic target for small decode GEMVs.

## Phase 3: optimize

### Decision tree

```text
Is idle time per step > ~10% of the step?
├─ yes → HOST-BOUND. Fix the host first.
│        Stack-sample the workers. Usual suspects in this fork:
│        - policies or configs rebuilt from env vars per step or per KV group
│        - attention metadata builds, once per KV-cache group
│        - per-step Python work in the sampler or output path
│        Re-measure. If tok/s did not move, you are GPU-bound now.
└─ no  → GPU-BOUND. Rank kernel classes from a fresh profile:
         ├─ many tiny kernels, gaps → fuse (routing glue, elementwise,
         │                            q/k norm + RoPE + KV write, router)
         ├─ attention               → grid sized to the live context, not
         │                            the capture envelope; GQA grouping;
         │                            window-aware reducer
         ├─ quantized GEMV, M ≤ 2   → dedicated GEMV reading the loaded
         │                            weights in place
         ├─ MoE experts             → fewer kernels per layer (W13 + SwiGLU,
         │                            then W2 + top-k reduce)
         └─ sampler                 → top-k candidates per vocab shard; no
                                      full-vocab sort
```

### Bound the gain before you build

1. Gain ceiling: `(profiled µs per call - streaming-read floor µs) × calls
   per token`. Skip anything under about 2% of the step.
2. Check registers and occupancy against the kernel you are replacing
   (`cuobjdump -res-usage`, or ptxas `-v`). Fusing two kernels can lower
   occupancy enough to cancel the overlap you wanted.
3. Check whether the work is latency-bound: about 5 MB read at about
   290 GB/s already is.

Flint's persistent MoE kernel failed checks 2 and 3. Fusing W13 and W2 cut
occupancy from 32 to 24 warps per SM (80 registers × 384 threads). W13 is
latency-bound, so the lost occupancy cancelled the overlap, and the W2
blocks only started after every W13 block was dispatched. It measured
18.44 → 19.07 µs per layer and ships switched off.

### What moved Flint, by kernel class

On `main`:

| Class | Change | Kill switch | Saving |
| --- | --- | --- | --- |
| Routing glue | TurboMind route instead of skinny (25 small ops per layer) | `--kernel-config.sm70_skinny_moe` | ~4 ms/token |
| Attention, long context | Grouped-query D128 decode, window-aware reducer, PV split over warps | | 128K: 26.9 → 77.2 tok/s |
| Dense NVFP4, M ≤ 2 | Single-token GEMV (`nvfp4_tm_gemv_sm70.cu`) | `VLLM_SM70_MODELOPT_NVFP4_GEMV` | Kernels 5.13 → 1.93 ms/token, with the next row |
| Routed experts, M ≤ 2 | Two-kernel decode (`nvfp4_moe_decode_qpn_sm70.cu`) | `VLLM_SM70_FLINT_MOE_QPN_DECODE` | (see above) |
| Host | Borrow engine policies (`0b731f4`) | | 4K: ~90 → 118.5 tok/s |

On `claude/flint-gpu-opt` (interim; per-step savings from the microbenchmark
at M=1):

| Class | Change | Kill switch | Saving |
| --- | --- | --- | --- |
| Router | One fused launch for cast, GEMV, split-K and top-k (20.6 → 8.3 µs/layer) | `VLLM_SM70_FLINT_FUSED_ROUTER` | 0.58 ms |
| Sampler | Vocab-parallel top-32 candidates (759 → 50 µs) | `VLLM_SM70_FLINT_SAMPLER` | ~0.7 ms (default sampling) |
| Attention, short context | One launch per layer; grid sized to the live context | `VLLM_SM70_FLINT_ATTN_FUSED_DECODE` | 0.43 ms |
| q/k norm + RoPE + KV write | One Triton kernel (9.3 → 2.25 µs sliding) | `VLLM_SM70_FLINT_QK_ROPE_KV` | 0.30 ms |
| All-reduce + add + RMSNorm | Fused into the custom all-reduce | `VLLM_SM70_FLINT_AR_RMSNORM` | 0.25 ms; off: crashed under expandable segments |
| MoE persistent | W13 and W2 in one kernel | `VLLM_SM70_FLINT_MOE_PERSISTENT` | None; default off |

### Rules for every change

1. **Kill switch.** A `VLLM_SM70_*` flag registered in `vllm/envs.py` that
   restores the previous route, read at layer load. Admit narrowly (exact
   scheme and shapes), so other models' routes do not change.
2. **Reference test.** GPU unit tests compare against an FP32 or FP64
   dequantized reference **and** against the old path, at every shape and M,
   including CUDA-graph replay. Tolerances used on Flint:
    - MoE decode and all-reduce: bitwise against the old path.
    - q/k norm + RoPE: within 1 FP16 ulp; V and the cache layout bit-exact.
    - Attention: about 2e-3 against the old path, plus an FP32 reference.
    - Router: ids exact except near-ties, weights within 2e-6.
    - Sampler: top-k equal to `torch.topk`; same token for the same seed in
      at least 63 of 64 cases; a chi-square test of the distribution.
3. **Test under real serve conditions.** Run the unit tests with the serve
   script's allocator (`PYTORCH_ALLOC_CONF=expandable_segments:True`),
   buffers allocated before capture, and both PIECEWISE and FULL graph
   capture. The fused all-reduce passed its tests but crashed the server:
   `cudaIpcGetMemHandle` fails on expandable-segment memory.
4. **No extra weight copies.** Read the weights the existing route already
   prepared, in place. Use fixed-order reductions, not float atomics, so the
   new path is reproducible.
5. **Same-tree A/B.** One build, two servers: kill switch off, then on. Never
   compare against a number from another tree or another day; the CPU clock
   and GPU temperature move results by 10–20%. Pause other CPU work (for
   example `kill -STOP` a running build) during a host-sensitive A/B.
6. **Re-check host vs GPU** after the change lands.

## Phase 4: validate

### Greedy ablation

Make the first end-to-end run of a change set an ablation, so drift is
attributed in one GPU window:

1. Capture nine greedy prompts × 128 tokens with top-2 logprobs on the old
   path (every new switch off) and on the new path.
2. Capture the old path twice, in fresh processes. TurboMind times its
   tactics live at M ≤ 64, so the old path alone gives 2–3 distinct outputs
   across process starts. That spread is the noise floor; judge drift
   against it, not against zero.
3. Capture once per switch with only that switch off.
4. At each flip, check the top-2 margin. Near-ties (Flint: margin ≤ 0.31)
   are FP16 rounding. A flip at a wide margin (Flint had one at 2.23) blocks
   the merge until a single switch is shown to cause or clear it.
5. Use prompts shorter than one 16-token KV block when you need exact
   repeats. Longer prompts that hit the prefix cache drift by about 1e-3 nats
   even on one build.

### Quality probes

Run `tools/flint/sm70_validate.sh --stage quality,agent --server-url URL` on
the final configuration. Flint scored arithmetic 10/10, coding 5/5, and
needle at 32K 3/3 (plus 128K and 250K at milestones). These probes catch
broken numerics and parsers, not small accuracy regressions.

## Phase 5: deploy

The reference host runs NixOS; the flake lives at `/etc/nixos`, outside git
**(example: c4130)**.

### Nix build and conversion

- **Fork build.** `/etc/nixos/vllm/flint.nix` pins the fork
  (`fork = { rev; hash; }` into `fetchFromGitHub`) and runs
  `setup.py build_ext --inplace` in the sandbox: `sm_70`, `MAX_JOBS=16`,
  about 33 minutes. Prefetch the CMake dependencies (cutlass, triton,
  flash-attention-v100). Apply Flash-V100's SM70 patches in Nix, because
  CMake skips them when `VLLM_FLASH_ATTN_SRC_DIR` is set. Add
  `-DCMAKE_SKIP_RPATH=ON`.
- **Package.** `package.nix` returns the stock `vllm` env, `vllmFlint` (the
  same venv with the fork tree first on `PYTHONPATH`), `flintCheckpoint` and
  `flintConverter`.
- **Conversion as a build step.** Fetch the GGUF and HF metadata as
  fixed-output derivations (pinned revision, full sha256). Run the converter
  on CPU in a derivation (`flint.mkCheckpoint`, about 3.5 minutes, 25 GB
  peak) and drop the non-deterministic `conversion_report.json`. The output
  is byte-identical to a hand conversion.
- **Pin the converter separately** (`converterFork`), so bumping the serving
  fork does not reconvert 23 GB.
- **Build, then switch.** As your user: `nixos-rebuild build --flake
  /etc/nixos#c4130 --cores 16 --max-jobs 1`. Then `sudo nixos-rebuild
  switch`, after the RAM guard is running.

### RAM guard

Start a watchdog before any risky switch or first start:

```bash
sudo systemd-run --unit=flint-ramguard1 --collect ~/flint-nix/ramguard.sh
```

Every 5 s for about 25 minutes, the script reads `MemAvailable` from
`/proc/meminfo`. Below 6 GB it runs `systemctl kill --signal=SIGKILL vllm;
systemctl stop vllm vllm-lmcache`. Write it as a script file with an
explicit `PATH`. An inline `sh -c` died at once: `seq` was not on the
transient unit's `PATH`, and systemd expanded `$var` itself.

### LMCache

- **Allocator.** Unset `PYTORCH_ALLOC_CONF=expandable_segments:True` for
  models served with LMCache. With it, CUDA IPC goes through `pidfd_getfd`,
  which `kernel.yama.ptrace_scope=1` denies. The symptom is
  `pidfd_getfd: Operation not permitted` in the LMCache server, then
  `register_kv_caches` times out after 300 s and the unit restart-loops.
  Leave `ptrace_scope` alone; the cost is about 0.25% of KV.
- **Connector.** Use LMCache's own MP connector (`kv_connector`
  `LMCacheMPConnector`, `kv_connector_module_path`
  `lmcache.integration.vllm.lmcache_mp_connector`, `kv_load_failure_policy`
  `recompute`), not the one bundled in 1Cat.
- **Multi-group KV.** Pass `--separate-object-groups` for models with
  several KV-cache groups (sliding and full attention).
- **Chunk size.** Make it a multiple of the vLLM block size (Flint: 256,
  block 16).
- **Memory.** Size L1 from the [host RAM rule](#memory-budget). Restarting
  `vllm-lmcache` empties L1.
- **Retrieval check.** Send a fixed 8K prompt, restart only `vllm`, and send
  it again. Flint: 0.31 s against 1.56 s cold, with identical output.

### Heat

A 250K prefill holds both GPUs at 82–83 °C for minutes. The hotter card's HBM
then throttles the decode that follows: 24 tok/s, against 63.7 tok/s after a
cool-down. TP runs at the slower card's pace.

## Phase 6: coordinate agents and sessions

Several agent sessions shared one GPU box with a production service.

### GPU arbitration

A `flock` convention on a `gpu.lock` file failed, because the production
systemd service never took the lock. What worked was one coordinator and
three messages:

```text
GPU REQUEST: <purpose>, ~<minutes>
GPU GRANTED: <purpose>, ~<minutes>  (plus what the coordinator verified)
GPU RELEASED                        (after the holder stopped its servers
                                     and checked nvidia-smi)
```

Before each grant, the coordinator checks:

- `nvidia-smi --query-gpu=index,memory.used` and `--query-compute-apps=pid`;
- the test ports are closed (`ss -ltn`);
- `systemctl is-active vllm`.

Ask the owner before stopping production for the first time. To stop it,
kill first, because `Restart=on-failure` would respawn it:

```bash
sudo sh -c 'systemctl kill --signal=SIGKILL vllm; systemctl stop vllm vllm-lmcache'
```

To restore it, run `systemctl reset-failed vllm; systemctl start
vllm-lmcache vllm`, then wait about 3 minutes for `/v1/models`.

### Work trees and builds

- **One remote work tree per agent.** Concurrent builds in one tree collide.
- **Do not copy build trees.** A copied tree's CMake cache points at the old
  path. Clone fresh, or delete `build/` and `CMakeCache.txt` in the copy.
- **CPU-only agents work in worktrees.** They write and unit-test kernels
  off the GPU, then share one batched GPU window.
- **Commit WIP early to local branches.** A long-running agent was lost to
  an accidental stop, and its uncommitted work had to be rescued by hand.

### Session hygiene

- Keep remote-access daemons alive across container restarts (for example,
  start `tailscaled` with `setsid`).
- Batch questions to an agent that is in a GPU window into one message.
- For user status updates, chain exact one-off reminders at :00 and :30.
  Recurring schedules drifted 7–9 minutes.
- Cloud sessions cannot delete remote branches through the git proxy. Ask the
  user to delete them in the GitHub UI.
- A rule of "exactly two e2e measurements" was too strict once a result
  surprised. Allow a same-tree A/B with kill switches whenever a number
  needs explaining.
- After deleting upstream CI or docs from the fork, re-run the architecture
  hooks and markdownlint, and check Docker `COPY` lines and `setup.py`
  requirement reads. All of them broke silently.

## What we would do differently

Ranked by time it would have saved on Flint:

1. **Check host vs GPU time at baseline and after every GPU change** (about
   1.5 hours and a build cycle). The single-token kernels cut kernel time by
   62% and moved e2e by 0%; the host fix then gave +31%.
2. **Budget host RAM and start a RAM guard before the production switch**
   (about 40 minutes, including a hard hang). The OOM and the LMCache attach
   failure were both predictable from the Phase 0 table.
3. **Unit-test under real serve conditions** (one GPU window). Expandable
   segments, pre-capture buffers, and PIECEWISE plus FULL capture would have
   caught the fused all-reduce crash off the GPU.
4. **Profile the current tip per kernel, and bound each gain, before
   building** (one work session). Comparing profiled per-call µs with the
   microbenchmark found the attention grid; a register and occupancy check
   would have skipped the persistent MoE kernel.
5. **Set up the infrastructure on day one:** GPU arbitration, one tree per
   agent, WIP commits, a persistent `tailscaled`. Each gap cost 15–30
   minutes.

Also: make the first e2e of any change set a per-switch ablation, so drift
is attributed in one GPU window instead of two.

## One-page checklist

Phase 0, qualify:

- [ ] The format has a proven SM70 route; the MoE shape is in
      `_SUPPORTED_CONTRACTS`
- [ ] FP16 only; plan for a power-of-two rescale if trained in BF16
- [ ] Per GPU: weights + KV at the target context + 0.6 GB ≤ 28.6 GiB
- [ ] Host: LMCache L1 + vLLM load peak + ~30 GB < RAM

Phase 1, bring-up:

- [ ] Deterministic converter under `vllm/models/<model>/`, with a header
      check
- [ ] Serve script with the full profile and a `--pp` switch
- [ ] `topo`, kernel tests, FP16 audit (≥ 10× headroom), smoke, quality,
      agent

Phase 2, measure:

- [ ] GPUs idle before starting; baseline at 4K and 32K, default and greedy,
      median of 3
- [ ] Host vs GPU per step on both ranks; one FULL graph per step
- [ ] Per-kernel profile table, kernels per step, unmatched names checked
- [ ] Microbenchmark of the top class: graph, cold L2, fresh layers per
      backend, a streaming-read floor

Phase 3, optimize (per change):

- [ ] Gain bound ≥ 2% of the step; registers and occupancy checked
- [ ] `VLLM_SM70_*` kill switch in `vllm/envs.py`, narrow admission
- [ ] Unit tests against FP32/FP64 and the old path, with graph replay, under
      serve conditions
- [ ] Same-tree A/B; host vs GPU re-checked

Phase 4, validate:

- [ ] Greedy: old path twice, new path, one run per switch off
- [ ] Every flip at a near-tie, or attributed to one switch
- [ ] Quality probes pass

Phases 5 and 6, deploy and coordinate:

- [ ] Nix build; conversion as a cached build step; converter pinned
      separately
- [ ] RAM guard running before the switch; LMCache without expandable
      segments
- [ ] GPU grants through the coordinator; one tree per agent; WIP committed
