# charlie-horse's 1Cat-LLM fork

A personal fork of [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM), the
[vLLM](https://github.com/vllm-project/vllm) fork that treats NVIDIA Volta
(Tesla V100, SM70) as a first-class target. This fork is tuned for one box:
two Tesla V100-PCIe-32GB cards behind a PCIe switch, no NVLink, serving one
model at a time with tensor parallelism 2.

### What's different

- Optimized for Qwen3.6 35B A3B + DFlash2
- Support for Recursive's Flint added + optimized

## Flint

[Flint](https://huggingface.co/RecursiveSI/flint) is a 39.8B-parameter
mixture-of-experts model with 2.5B active parameters: 49 layers (2 dense, 47
MoE with 256 experts, top-6 plus a shared expert), sliding-window and
full-attention layers with attention sinks. This fork runs it in NVFP4 on
V100:

- **Model and checkpoint.** `vllm/models/flint/` holds the model and a
  converter from the published NVFP4 GGUF to a ModelOpt W4A16 NVFP4
  checkpoint (FP16 activations; V100 has no BF16). The converter applies an
  FP16 overflow rescale so the MoE and the late-layer residual stay in range.
- **Attention.** A grouped-query paged decode kernel for the sink layers in
  the Flash-V100 backend.
- **Single-token NVFP4 kernels.** At decode batch 1–2 the dense projections
  run a dedicated GEMV, and the routed experts run a two-kernel decode path
  (W13 with fused SwiGLU, then W2 with the top-k reduce). Both read the
  loaded TurboMind weights in place, with no extra copies.
- **Host path.** The decode step reuses the engine's resolved policies
  instead of rebuilding them from the environment on every step.

### Results

Single stream, 2× V100-PCIe-32GB, TP2, FP16, 256 generated tokens, median of
3 runs, random-token prompts:

| Context | Decode, default sampling | Decode, greedy | Time to first token |
| --- | ---: | ---: | ---: |
| 4K | 118.5 tok/s | 124.0 tok/s | 0.69 s |
| 32K | 110.4 tok/s | 118.2 tok/s | 10.4 s |

At 4K that is about +23% over the same build with the single-token kernels
switched off (96.7 / 100.3 tok/s). Measurements, profiles, correctness and
quality checks are in
[`docs/design/sm70_flint_tp2_pcie.md`](docs/design/sm70_flint_tp2_pcie.md).

### Converting and serving

```bash
# 1. Convert the GGUF (CPU only, ~2-4 min, ~25 GB peak RAM).
python -m vllm.models.flint.convert_gguf RecursiveSI/flint-nvfp4-gguf \
    /models/flint-nvfp4 --hf-config RecursiveSI/flint

# 2. Serve on two V100s (TP2, 262K context, up to 2 sequences).
scripts/serve_flint_nvfp4_v100_pcie.sh /models/flint-nvfp4 --port 8000
```

The serve script documents its defaults, including the TurboMind NVFP4 MoE
route and the GLM-4.7 tool-call and reasoning parsers. If you put
[LMCache](https://github.com/LMCache/LMCache) in front of it, unset
`PYTORCH_ALLOC_CONF=expandable_segments:True`. With expandable segments,
CUDA IPC goes through `pidfd_getfd`, which a `kernel.yama.ptrace_scope` of 1
denies to the LMCache server, and vLLM's KV registration times out.

### Kill switches

Both are read when the layers load, and set to `0` to fall back to the
TurboMind GEMMs:

| Variable | Turns off |
| --- | --- |
| `VLLM_SM70_MODELOPT_NVFP4_GEMV` | The single-token dense NVFP4 GEMV |
| `VLLM_SM70_FLINT_MOE_QPN_DECODE` | The two-kernel routed-expert decode |

## Qwen3.6 35B A3B

Served FP8 (`Qwen/Qwen3.6-35B-A3B-FP8`) with the
[`incoai/Qwen3.6-35B-A3B-DFlash2`](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2)
draft for speculative decoding: 7 speculative tokens, probabilistic draft
sampling, `FLASH_ATTN_V100` for the draft. On the same two V100s it decodes
about 111 tok/s at 4K, against 66 tok/s without the draft. Model-specific
flags: `--language-model-only --mamba-cache-mode align`, `qwen3_coder` tool
parser and `qwen3` reasoning parser.

## Building from source

Requirements: an SM70 GPU, CUDA 12.x (CUDA 13 can't target SM70), and
PyTorch built with SM70 support (2.10 + cu128 here). Use FP16 everywhere
(`--dtype half`).

The verified build compiles the extensions in place, using a Python
environment that already has PyTorch and the runtime requirements
(`requirements/common.txt`, `requirements/cuda.txt`), and puts the tree on
`PYTHONPATH`:

```bash
export TORCH_CUDA_ARCH_LIST=7.0 CMAKE_CUDA_ARCHITECTURES=70
export VLLM_TARGET_DEVICE=cuda CMAKE_BUILD_TYPE=Release MAX_JOBS=16
python setup.py build_ext --inplace
export PYTHONPATH=$PWD:$PWD/flash-attention-v100:$PYTHONPATH
```

A full build of the CUDA extensions takes about 25 minutes. The Flash-V100
attention kernels in `flash-attention-v100/` build separately, with their own
`setup.py`.

## Upstream and license

This fork builds on [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) and
[vLLM](https://github.com/vllm-project/vllm). Both are Apache-2.0, as is
this fork ([LICENSE](LICENSE)). Much of the work in this fork was done with
AI assistance (Claude Code).
