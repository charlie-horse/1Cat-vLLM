#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Flint NVFP4 (W4A16) coding-agent recipe, 2x V100-PCIe-32GB.
# Installed beside vllm by the wheel, so parameters follow the package version.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: serve_flint_nvfp4_v100_pcie.sh MODEL [--pp] [vllm serve options...]

MODEL is the ModelOpt NVFP4 directory converted from the Flint GGUF:

  python -m vllm.models.flint.convert_gguf RecursiveSI/flint-nvfp4-gguf \
    /models/flint-nvfp4 --hf-config RecursiveSI/flint

Profile: TP2, FP16 compute and KV, 262144-token context, 8192-token prefill,
up to 2 sequences, GPU memory utilization 0.90, prefix caching with cached
prompt tokens reported in usage, GLM-4.7 tool-call and reasoning parsers,
tools dropped from the prompt when tool_choice is "none", served as "flint".
The model config selects the SM70 attention backend (TRITON_ATTN or
FLASH_ATTN_V100); --attention-backend is left unset. Routed experts run on
the TurboMind NVFP4 MoE route (--kernel-config.sm70_skinny_moe false): on
2x V100 PCIe it decodes about 40% faster than the skinny route at 4K and
prefills 20-30% faster. Pass --kernel-config.sm70_skinny_moe true to
compare; give other kernel options in the same dotted form, since a JSON
--kernel-config would be replaced by this default.

Hardware: two V100 PCIe 32GB GPUs, ideally behind one PCIe switch. Check
`nvidia-smi topo -m`: PIX or PXB between the pair keeps peer traffic on the
switch; PHB, NODE or SYS routes it through the CPU, where --pp may be faster.

Environment defaults, each kept if already set:
  CUDA_DEVICE_ORDER=PCI_BUS_ID      GPU indices follow nvidia-smi.
  VLLM_SKIP_P2P_CHECK=0             Run the real CUDA IPC peer-write test
                                    instead of trusting the driver; without
                                    it, PCIe pairs that report P2P but drop
                                    peer stores hang at the first decode.
  PYTORCH_ALLOC_CONF=expandable_segments:True
                                    Only when PYTORCH_CUDA_ALLOC_CONF is
                                    also unset, since PyTorch prefers it.

Options:
  --pp        Pipeline parallelism over the two GPUs instead of TP2
              (--tensor-parallel-size 1 --pipeline-parallel-size 2), for the
              PCIe TP-vs-PP comparison.
  -h, --help  Show this help.

Additional vllm serve options override the profile, for example:

  serve_flint_nvfp4_v100_pcie.sh /models/flint-nvfp4 --port 8001
  serve_flint_nvfp4_v100_pcie.sh /models/flint-nvfp4 --max-model-len 131072
  serve_flint_nvfp4_v100_pcie.sh /models/flint-nvfp4 --pp

Client settings (OpenAI-compatible chat completions):
  base URL    http://<host>:8000/v1
  model       flint
  max_tokens  32768 or more; the thinking counts toward it.
  reasoning   Responses carry the thinking in message.reasoning. Send it back
              on assistant turns in the history as reasoning_content (the
              server also accepts reasoning) so the chat template keeps it.
  sampling    temperature 1.0, top_p 0.95, top_k 20 (the server defaults;
              greedy decoding is not validated).
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  usage
  exit 0
fi
if [[ $# -eq 0 || $1 == -* ]]; then
  usage >&2
  exit 2
fi
model=$1
shift
parallel_args=(--tensor-parallel-size 2)
serve_options=()
for arg in "$@"; do
  if [[ $arg == --pp ]]; then
    parallel_args=(--tensor-parallel-size 1 --pipeline-parallel-size 2)
  else
    serve_options+=("$arg")
  fi
done

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
# The driver's P2P report is not enough on PCIe; see the help text.
export VLLM_SKIP_P2P_CHECK=${VLLM_SKIP_P2P_CHECK:-0}
# PyTorch reads the legacy variable whenever it is present, even if empty.
if [[ -z ${PYTORCH_ALLOC_CONF+set} && -z ${PYTORCH_CUDA_ALLOC_CONF+set} ]]; then
  export PYTORCH_ALLOC_CONF=expandable_segments:True
fi

# Prefer the CLI installed alongside this script, including when the caller
# invokes it by absolute path without activating that Python environment.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
vllm_cli=vllm
if [[ -x "$script_dir/vllm" ]]; then
  vllm_cli="$script_dir/vllm"
fi
if ! vllm_cli=$(command -v "$vllm_cli"); then
  echo "error: vllm CLI not found; install vllm or activate its environment" >&2
  exit 127
fi
exec "$vllm_cli" serve "$model" \
  "${parallel_args[@]}" --dtype half --kv-cache-dtype auto \
  --max-model-len 262144 --max-num-seqs 2 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --kernel-config.sm70_skinny_moe false \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47 \
  --exclude-tools-when-tool-choice-none \
  --served-model-name flint \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20}' \
  "${serve_options[@]}"
