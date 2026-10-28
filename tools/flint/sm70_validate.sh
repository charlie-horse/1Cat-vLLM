#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Validation runbook: Flint NVFP4 on 2x V100 PCIe 32 GB (TP2, FP16, 262K).
# Stages run as "stage_$name" from run_stage, which shellcheck cannot follow.
# shellcheck disable=SC2329
set -uo pipefail

usage() {
  cat <<'EOF'
Usage: tools/flint/sm70_validate.sh [--stage STAGE[,STAGE...]] [options]

Runs the Flint SM70 validation on a 2x V100 PCIe 32 GB host and writes a
timestamped results directory with one log per stage and summary.md.

Stages (default: all, in this order):
  topo     nvidia-smi topology/clocks/power, peer access, the vLLM CUDA IPC
           P2P test (VLLM_SKIP_P2P_CHECK=0) and a TP2 custom all-reduce probe
  build    source build for SM70 (TORCH_CUDA_ARCH_LIST=7.0) incl. Flash-V100;
           prints the commands, runs them only with --yes; then checks imports
  convert  NVFP4 GGUF -> ModelOpt W4A16_NVFP4 safetensors, then checks them
  kernels  sink prefill/decode tests, MoE contract test, Flash-V100 subset
  audit    FP16 range audit (tools/flint/fp16_range_audit.py)
  smoke    start the serve script, wait for /health, one chat request, stop
  quality  greedy coherence, needles at 32K/128K/~250K, arithmetic and
           coding probes checked by executing asserts
  agent    tools/flint/agent_e2e.py (tool calls, reasoning, prefix cache)
  bench    vllm bench serve: TTFT and decode tok/s at 4K/32K/128K over an
           A/B matrix (attention backend, MoE route, TP2 vs PP2)

Options:
  --stage LIST         stages to run (repeatable, comma separated, or "all")
  --model DIR          converted checkpoint directory (env MODEL_DIR)
  --gguf FILE          NVFP4 GGUF source for the convert stage (env GGUF)
  --hf-config ID|DIR   HF config source for the converter (RecursiveSI/flint)
  --tokenizer ID|DIR   tokenizer for vllm bench serve (default: --model if it
                       holds tokenizer files, else --hf-config)
  --serve-script FILE  default scripts/serve_flint_nvfp4_v100_pcie.sh
  --port N             server port (default 8000)
  --gpus LIST          CUDA_VISIBLE_DEVICES for the run (default 0,1)
  --server-url URL     use an already running server for smoke/quality/agent
                       (and a single "external" bench row); never stopped
  --results-dir DIR    results directory (default
                       bench_results/flint_sm70/<timestamp>)
  --bench-matrix M     oneway (default: base config plus one change per
                       factor) or full (all 8 combinations)
  --bench-configs L    explicit space-separated configs, e.g.
                       "tp2-flash-skinny pp2-triton-turbomind"
  --yes, -y            run the build commands (otherwise they are printed)
  --flash-only         build: rebuild only the Flash-V100 extension in place
  --force-convert      convert even when --model already has config.json
  -h, --help           show this help

Bench configs are <tp2|pp2>-<triton|flash|auto>-<skinny|turbomind|auto>:
  triton/flash -> --attention-backend TRITON_ATTN / FLASH_ATTN_V100
  skinny/turbomind -> --kernel-config.sm70_skinny_moe true / false
  pp2 -> the serve script's --pp switch

Environment knobs: BENCH_LENS ("4096 32768 131072"), BENCH_OUTPUT_LEN (256),
BENCH_REPS (3 measured requests after one cold request), BENCH_BASE
(tp2-flash-skinny), NEEDLE_LENGTHS (32768,131072,250000), NEEDLE_DEPTHS
(0.1,0.5,0.9), MIN_MATH (8 of 10), MIN_CODE (4 of 5), SERVE_ARGS (extra
vllm serve flags for every launch), AUDIT_ARGS, AGENT_ARGS, CUDA_HOME
(/usr/local/cuda-12.8), MAX_JOBS, SERVER_START_TIMEOUT (3600 s),
PYTHON_BIN (.venv/bin/python).

Exit status: 0 when no stage failed, 1 otherwise.
EOF
}

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
ALL_STAGES=(topo build convert kernels audit smoke quality agent bench)

PY=${PYTHON_BIN:-$REPO/.venv/bin/python}
MODEL_DIR=${MODEL_DIR:-}
GGUF=${GGUF:-}
HF_CONFIG=${HF_CONFIG:-RecursiveSI/flint}
TOKENIZER=${TOKENIZER:-}
SERVE_SCRIPT=${SERVE_SCRIPT:-$REPO/scripts/serve_flint_nvfp4_v100_pcie.sh}
SERVED_MODEL=${SERVED_MODEL:-flint}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
GPU_IDS=${GPU_IDS:-0,1}
SERVER_URL=${SERVER_URL:-}
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-3600}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
BENCH_LENS=${BENCH_LENS:-4096 32768 131072}
BENCH_OUTPUT_LEN=${BENCH_OUTPUT_LEN:-256}
BENCH_REPS=${BENCH_REPS:-3}
BENCH_MATRIX=${BENCH_MATRIX:-oneway}
BENCH_CONFIGS=${BENCH_CONFIGS:-}
BENCH_BASE=${BENCH_BASE:-tp2-flash-skinny}
NEEDLE_LENGTHS=${NEEDLE_LENGTHS:-32768,131072,250000}
NEEDLE_DEPTHS=${NEEDLE_DEPTHS:-0.1,0.5,0.9}
MIN_MATH=${MIN_MATH:-8}
MIN_CODE=${MIN_CODE:-4}
SERVE_ARGS=${SERVE_ARGS:-}
AUDIT_ARGS=${AUDIT_ARGS:-}
AGENT_ARGS=${AGENT_ARGS:-}
RESULTS_DIR=${RESULTS_DIR:-}
ASSUME_YES=0
FLASH_ONLY=0
FORCE_CONVERT=0
REQUESTED=()
COMMAND_LINE="$0 $*"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

while (($#)); do
  case $1 in
    --stage) REQUESTED+=("${2:?--stage needs a value}"); shift ;;
    --stage=*) REQUESTED+=("${1#*=}") ;;
    --model) MODEL_DIR=${2:?}; shift ;;
    --gguf) GGUF=${2:?}; shift ;;
    --hf-config) HF_CONFIG=${2:?}; shift ;;
    --tokenizer) TOKENIZER=${2:?}; shift ;;
    --serve-script) SERVE_SCRIPT=${2:?}; shift ;;
    --port) PORT=${2:?}; shift ;;
    --gpus) GPU_IDS=${2:?}; shift ;;
    --server-url) SERVER_URL=${2:?}; shift ;;
    --results-dir) RESULTS_DIR=${2:?}; shift ;;
    --bench-matrix) BENCH_MATRIX=${2:?}; shift ;;
    --bench-configs) BENCH_CONFIGS=${2:?}; shift ;;
    --yes | -y) ASSUME_YES=1 ;;
    --flash-only) FLASH_ONLY=1 ;;
    --force-convert) FORCE_CONVERT=1 ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
  shift
done

STAGES=()
for item in "${REQUESTED[@]:-all}"; do
  for name in ${item//,/ }; do
    if [[ $name == all ]]; then
      STAGES+=("${ALL_STAGES[@]}")
    elif [[ " ${ALL_STAGES[*]} " == *" $name "* ]]; then
      STAGES+=("$name")
    else
      die "unknown stage: $name (choose from: ${ALL_STAGES[*]} all)"
    fi
  done
done
[[ $BENCH_MATRIX == oneway || $BENCH_MATRIX == full ]] ||
  die "--bench-matrix must be oneway or full"

if [[ -n $MODEL_DIR ]]; then MODEL_DIR=$(realpath -m -- "$MODEL_DIR"); fi
read -r -a SERVE_EXTRA <<<"$SERVE_ARGS"
BASE_URL=${SERVER_URL:-http://$HOST:$PORT}
BASE_URL=${BASE_URL%/}
BASE_URL=${BASE_URL%/v1}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export VLLM_SKIP_P2P_CHECK=0
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
[[ -d $CUDA_HOME/bin ]] && export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
[[ -d $REPO/.venv/bin ]] && export PATH="$REPO/.venv/bin:$PATH"
# An in-place build (setup.py build_ext --inplace) runs from the source tree.
if compgen -G "$REPO/vllm/_C*.so" >/dev/null; then
  export PYTHONPATH="$REPO:$REPO/flash-attention-v100${PYTHONPATH:+:$PYTHONPATH}"
fi

STAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR=$(realpath -m -- "${RESULTS_DIR:-$REPO/bench_results/flint_sm70/$STAMP}")
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/server" || die "cannot create $RUN_DIR"
: >"$RUN_DIR/stages.tsv"
HELPER=$RUN_DIR/flint_probes.py
STARTED=$(date -Is)

# ---------------------------------------------------------------- helpers
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }
show() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}
note() { STAGE_NOTE+="${STAGE_NOTE:+; }$*"; }
fail() {
  STAGE_FAILS=$((STAGE_FAILS + 1))
  note "$*"
  log "FAIL: $*"
}
warn() {
  STAGE_WARNS=$((STAGE_WARNS + 1))
  note "$*"
  log "WARN: $*"
}
stage_result() {
  ((STAGE_FAILS)) && return 1
  ((STAGE_WARNS)) && return 2
  return 0
}
need_python() {
  [[ -x $PY ]] && return 0
  fail "missing $PY (run --stage build --yes)"
  return 1
}
need_model() {
  [[ -n $MODEL_DIR && -f $MODEL_DIR/config.json ]] && return 0
  fail "no converted checkpoint (pass --model DIR; run --stage convert)"
  return 1
}

# run_check LABEL [--warn2] CMD...: exit 0 pass, 2 warn (with --warn2),
# anything else fail; the first FAIL:/WARN: line of the output is the note.
run_check() {
  local label=$1 warn2=0 out rc line
  shift
  if [[ ${1:-} == --warn2 ]]; then warn2=1; shift; fi
  out=$STAGE_DIR/$label.out
  "$@" 2>&1 | tee "$out"
  rc=${PIPESTATUS[0]}
  if ((rc == 0)); then
    log "ok: $label"
  elif ((rc == 2 && warn2)); then
    line=$(grep -m1 '^WARN: ' "$out" | cut -c7-)
    warn "$label: ${line:-see logs}"
  else
    line=$(grep -m1 '^FAIL: ' "$out" | cut -c7-)
    fail "$label (exit $rc)${line:+: $line}"
  fi
  return 0
}

write_summary() {
  local summary=$RUN_DIR/summary.md name status secs notes gpus git_rev
  gpus=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null |
    paste -sd ';' -)
  git_rev=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
  {
    echo "# Flint SM70 validation summary"
    echo
    echo "| Field | Value |"
    echo "| --- | --- |"
    echo "| Started | $STARTED |"
    echo "| Host | $(hostname 2>/dev/null) |"
    echo "| Source | \`$git_rev\` ($(git -C "$REPO" branch --show-current 2>/dev/null)) |"
    echo "| GPUs | ${gpus:-unknown} (CUDA_VISIBLE_DEVICES=$GPU_IDS) |"
    echo "| Model | ${MODEL_DIR:-unset} |"
    echo "| Serve script | ${SERVE_SCRIPT#"$REPO"/} |"
    echo "| Command | \`${COMMAND_LINE}\` |"
    echo
    echo "## Stages"
    echo
    echo "| Stage | Status | Seconds | Notes | Log |"
    echo "| --- | --- | ---: | --- | --- |"
    while IFS=$'\t' read -r name status secs notes; do
      echo "| $name | $status | $secs | ${notes:-} | \`logs/$name.log\` |"
    done <"$RUN_DIR/stages.tsv"
    for name in "${ALL_STAGES[@]}"; do
      [[ -s $RUN_DIR/$name/summary.md ]] || continue
      echo
      echo "## $name"
      echo
      cat "$RUN_DIR/$name/summary.md"
    done
  } >"$summary"
}

run_stage() {
  local name=$1 start rc status
  STAGE_DIR=$RUN_DIR/$name
  STAGE_NOTE=""
  STAGE_FAILS=0
  STAGE_WARNS=0
  mkdir -p "$STAGE_DIR"
  start=$SECONDS
  log "=== $name: start (log: $RUN_DIR/logs/$name.log)"
  "stage_$name" > >(tee -a "$RUN_DIR/logs/$name.log") 2>&1
  rc=$?
  case $rc in
    0) status=PASS ;;
    2) status=WARN ;;
    3) status=SKIP ;;
    *) status=FAIL ;;
  esac
  STAGE_NOTE=${STAGE_NOTE//$'\t'/ }
  STAGE_NOTE=${STAGE_NOTE//$'\n'/ }
  STAGE_NOTE=${STAGE_NOTE//|//}
  printf '%s\t%s\t%s\t%s\n' "$name" "$status" "$((SECONDS - start))" \
    "${STAGE_NOTE:0:400}" >>"$RUN_DIR/stages.tsv"
  log "=== $name: $status${STAGE_NOTE:+ ($STAGE_NOTE)}"
  write_summary
}

# ----------------------------------------------------------------- server
SERVER_PID=""
SERVER_LABEL=""
SERVER_ERROR=""

health() { curl -sf -m 5 -o /dev/null "$BASE_URL/health"; }

server_down() {
  [[ -n $SERVER_PID ]] || return 0
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    log "stopping server '$SERVER_LABEL' (process group $SERVER_PID)"
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null
    local i
    for ((i = 0; i < 60; i++)); do
      kill -0 -- "-$SERVER_PID" 2>/dev/null || break
      sleep 2
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null
  fi
  wait "$SERVER_PID" 2>/dev/null
  SERVER_PID=""
  SERVER_LABEL=""
  sleep 3
}

# server_up LABEL [--pp] [vllm serve args...]: reuse a running server with
# the same label, otherwise restart. Sets SERVER_ERROR and returns 1 on error.
server_up() {
  local label=$1 log waited=0
  shift
  SERVER_ERROR=""
  if [[ -n $SERVER_URL ]]; then
    health && return 0
    SERVER_ERROR="external server $BASE_URL is not healthy"
    return 1
  fi
  if [[ -n $SERVER_PID && $SERVER_LABEL == "$label" ]] &&
    kill -0 "$SERVER_PID" 2>/dev/null && health; then
    return 0
  fi
  server_down
  if health; then
    SERVER_ERROR="another server already answers on $BASE_URL (stop it or pass --server-url)"
    return 1
  fi
  if [[ ! -f $SERVE_SCRIPT ]]; then
    SERVER_ERROR="missing serve script $SERVE_SCRIPT"
    return 1
  fi
  if [[ -z $MODEL_DIR || ! -f $MODEL_DIR/config.json ]]; then
    SERVER_ERROR="no converted checkpoint (pass --model DIR)"
    return 1
  fi
  local pp=()
  if [[ ${1:-} == --pp ]]; then
    pp=(--pp)
    shift
  fi
  log=$RUN_DIR/server/$label.log
  local cmd=(bash "$SERVE_SCRIPT" "$MODEL_DIR" "${pp[@]}" --host "$HOST" --port "$PORT"
    "$@" "${SERVE_EXTRA[@]}")
  log "starting server '$label' (log: $log)"
  printf '+'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  # A new session makes the serve script, API server and workers one process
  # group, so server_down can stop all of them.
  if have setsid; then
    setsid "${cmd[@]}" >"$log" 2>&1 </dev/null &
  else
    "${cmd[@]}" >"$log" 2>&1 </dev/null &
  fi
  SERVER_PID=$!
  SERVER_LABEL=$label
  until health; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -n 40 "$log"
      SERVER_PID=""
      SERVER_ERROR="server '$label' exited during startup (see server/$label.log)"
      return 1
    fi
    if ((waited >= SERVER_START_TIMEOUT)); then
      tail -n 40 "$log"
      server_down
      SERVER_ERROR="server '$label' not healthy after ${waited} s"
      return 1
    fi
    sleep 10
    waited=$((waited + 10))
  done
  log "server '$label' healthy after ~${waited} s"
  curl -sf -m 30 "$BASE_URL/v1/models" >"$RUN_DIR/server/$label.models.json"
  curl -sf -m 30 "$BASE_URL/v1/sm70/acceleration" \
    >"$RUN_DIR/server/$label.acceleration.json" 2>/dev/null
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv \
    >"$RUN_DIR/server/$label.memory.csv" 2>/dev/null
  grep -E -i "KV cache size|maximum concurrency|attention backend|skinny moe|turbomind|custom allreduce|cuda graph" \
    "$log" | tail -n 40 >"$RUN_DIR/server/$label.routes.txt"
  return 0
}

trap server_down EXIT
trap 'exit 130' INT TERM

# ------------------------------------------------------------------ stages
stage_topo() {
  if ! have nvidia-smi; then
    fail "nvidia-smi not found"
    return 1
  fi
  show nvidia-smi -L
  show nvidia-smi topo -m
  show nvidia-smi topo -p2p r
  show nvidia-smi topo -p2p w
  show nvidia-smi --format=csv \
    --query-gpu=index,name,pci.bus_id,driver_version,memory.total,persistence_mode,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max ||
    warn "nvidia-smi device query failed"
  show nvidia-smi --format=csv \
    --query-gpu=index,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,power.draw,power.limit,power.max_limit,temperature.gpu ||
    warn "nvidia-smi clock/power query failed"
  show nvidia-smi -q -d CLOCK,POWER,PERFORMANCE
  show nvidia-smi nvlink -s
  show uname -r
  show nproc
  show free -g
  need_python || return 1
  # The cached P2P check writes its JSON under VLLM_CACHE_ROOT: use a fresh
  # one so the result reflects this run, not an older cache file.
  run_check p2p_allreduce --warn2 env VLLM_CACHE_ROOT="$STAGE_DIR/vllm_cache" \
    "$PY" "$HELPER" topo --out "$STAGE_DIR/topo.json" --markdown "$STAGE_DIR/summary.md"
  stage_result
}

write_build_script() {
  local script=$1 njobs=$2
  {
    cat <<EOF
#!/usr/bin/env bash
# Generated by tools/flint/sm70_validate.sh: SM70 source build of this fork.
set -euo pipefail
cd $(printf '%q' "$REPO")
export CUDA_HOME=$(printf '%q' "$CUDA_HOME")
export MAX_JOBS=$njobs
EOF
    cat <<'EOF'
export PATH="$CUDA_HOME/bin:$PATH"
# SM70 targets for vLLM, Flash-V100 and FlashQLA (README: Build From Source).
export TORCH_CUDA_ARCH_LIST=7.0 CMAKE_CUDA_ARCHITECTURES=70
export FLASH_ATTN_V100_CUDA_ARCH_LIST=7.0
export NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release
export VLLM_TARGET_DEVICE=cuda CCACHE_NOHASHDIR=true
unset VLLM_USE_PRECOMPILED
EOF
    if ((FLASH_ONLY)); then
      cat <<'EOF'
# Rebuild only flash_attn_v100_cuda and paged_kv_utils in place, then copy
# them into the package directory (as setup.py bundles them).
cd flash-attention-v100
../.venv/bin/python setup.py build_ext --inplace
cp -f flash_attn_v100_cuda*.so paged_kv_utils*.so flash_attn_v100/
EOF
    else
      cat <<'EOF'
# AGENTS.md: uv and .venv only. The torch backend is pinned to cu128:
# --torch-backend=auto may choose a CUDA 13 wheel on new drivers, and CUDA 13
# has no SM70 target.
[[ -x .venv/bin/python ]] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements/build/cuda.txt \
  --torch-backend=cu128
# Editable build: vllm/_C*, _moe_C, Flash-V100 (flash-attention-v100/) and
# FlashQLA SM70; setup.py builds and bundles Flash-V100 itself.
uv pip install --python .venv/bin/python --no-build-isolation -e . \
  --torch-backend=cu128
# Test dependencies: cuda.txt is pinned for x86_64, cuda.in elsewhere.
test_reqs=requirements/test/cuda.in
[[ $(uname -m) == x86_64 ]] && test_reqs=requirements/test/cuda.txt
uv pip install --python .venv/bin/python -r "$test_reqs" --torch-backend=cu128
EOF
    fi
  } >"$script"
  chmod +x "$script"
}

stage_build() {
  local script=$STAGE_DIR/build.sh njobs mem_gib
  mem_gib=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo 2>/dev/null)
  # About 3 GiB of host RAM per nvcc job.
  njobs=${MAX_JOBS:-$(($(nproc) < ${mem_gib:-8} / 3 ? $(nproc) : ${mem_gib:-8} / 3))}
  ((njobs >= 1)) || njobs=1
  have uv || warn "uv not found (curl -LsSf https://astral.sh/uv/install.sh | sh)"
  if [[ -x $CUDA_HOME/bin/nvcc ]]; then
    show "$CUDA_HOME/bin/nvcc" --version
    "$CUDA_HOME/bin/nvcc" --version | grep -q 'release 12\.8' ||
      warn "nvcc is not CUDA 12.8 (the SM70 release uses 12.8)"
  else
    warn "no nvcc at $CUDA_HOME/bin (set CUDA_HOME to a CUDA 12.8 toolkit)"
  fi
  have gcc && show gcc --version
  log "host: $(nproc) CPUs, ${mem_gib:-?} GiB available, MAX_JOBS=$njobs"
  write_build_script "$script" "$njobs"
  log "build commands (saved as $script):"
  cat "$script"
  if ((ASSUME_YES)); then
    show bash "$script" || {
      fail "build failed"
      return 1
    }
  else
    note "commands printed only; rerun with --yes to build"
  fi
  if [[ ! -x $PY ]]; then
    if ((ASSUME_YES)); then fail "no $PY after the build"; else warn "not built yet"; fi
    stage_result
    return
  fi
  run_check imports --warn2 "$PY" "$HELPER" verify-build --markdown "$STAGE_DIR/summary.md"
  if ((!ASSUME_YES && STAGE_FAILS)); then
    # Without --yes a failing import check means "not built yet", not a
    # broken build.
    STAGE_WARNS=$((STAGE_WARNS + STAGE_FAILS))
    STAGE_FAILS=0
  fi
  stage_result
}

stage_convert() {
  need_python || return 1
  if [[ -z $MODEL_DIR ]]; then
    fail "pass --model OUT_DIR for the converted checkpoint"
    return 1
  fi
  if [[ -f $MODEL_DIR/config.json ]] && ((!FORCE_CONVERT)); then
    log "$MODEL_DIR/config.json exists; checking it without converting (--force-convert)"
    note "existing checkpoint checked"
  else
    if [[ -z $GGUF || ! -f $GGUF ]]; then
      fail "pass --gguf with the NVFP4 GGUF file"
      return 1
    fi
    mkdir -p "$(dirname "$MODEL_DIR")"
    show df -h "$(dirname "$MODEL_DIR")"
    show free -g
    local timer=()
    [[ -x /usr/bin/time ]] && timer=(/usr/bin/time -v)
    show "${timer[@]}" "$PY" -m vllm.models.flint.convert_gguf "$GGUF" "$MODEL_DIR" \
      --hf-config "$HF_CONFIG" || {
      fail "converter exited non-zero"
      return 1
    }
  fi
  show du -sh "$MODEL_DIR"
  run_check checkpoint --warn2 "$PY" "$HELPER" checkpoint "$MODEL_DIR" \
    --markdown "$STAGE_DIR/summary.md"
  stage_result
}

# run_pytest LABEL REQUIRED TEST...: REQUIRED=1 fails on missing files and
# warns when every test skipped.
run_pytest() {
  local label=$1 required=$2 xml rc tests failures errors skipped present=() t
  shift 2
  for t in "$@"; do
    if [[ -f $REPO/${t%%::*} ]]; then
      present+=("$t")
    elif ((required)); then
      fail "missing ${t%%::*}"
    else
      warn "missing ${t%%::*}"
    fi
  done
  ((${#present[@]})) || return 0
  xml=$STAGE_DIR/$label.xml
  (cd "$REPO" && show env CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" "$PY" -m pytest -q -rs \
    -p no:cacheprovider --junitxml="$xml" "${present[@]}")
  rc=$?
  read -r tests failures errors skipped < <("$PY" "$HELPER" junit "$xml" 2>/dev/null ||
    echo "0 0 0 0")
  printf '| %s | %s | %s | %s | %s | %s |\n' "$label" "$tests" "$failures" "$errors" \
    "$skipped" "$rc" >>"$STAGE_DIR/summary.md"
  if ((rc != 0 && rc != 5)) || ((failures + errors)); then
    fail "$label: $failures failed, $errors errors (pytest exit $rc)"
  elif ((tests == 0 || tests == skipped)); then
    if ((required)); then warn "$label: every test skipped"; fi
  else
    log "ok: $label: $tests tests, $skipped skipped"
  fi
}

stage_kernels() {
  need_python || return 1
  server_down
  printf '| Group | Tests | Failed | Errors | Skipped | Exit |\n' >"$STAGE_DIR/summary.md"
  printf '| --- | ---: | ---: | ---: | ---: | ---: |\n' >>"$STAGE_DIR/summary.md"
  run_pytest flint_new 1 \
    tests/kernels/attention/test_sm70_flash_v100_sinks_prefill.py \
    tests/kernels/attention/test_sm70_flash_v100_sinks_decode.py \
    tests/kernels/moe/test_sm70_nvfp4_contract_flint.py
  run_pytest flash_v100_regression 0 \
    tests/kernels/attention/test_sm70_flash_v100_paged_prefill_determinism.py \
    tests/kernels/attention/test_sm70_flash_v100_splitkv_n.py \
    tests/kernels/attention/test_sm70_flash_v100_anchored_swa.py \
    tests/kernels/attention/test_sm70_flash_v100_paged_prefill_any_page.py \
    tests/kernels/attention/test_sm70_flash_v100_varlen_layout.py::test_sm70_flash_v100_varlen_type_a_layout_exact
  run_pytest moe_regression 0 tests/kernels/moe/test_skinny_sm70_moe.py
  stage_result
}

stage_audit() {
  need_python || return 1
  need_model || return 1
  local tool=$REPO/tools/flint/fp16_range_audit.py extra=() rc
  if [[ ! -f $tool ]]; then
    fail "missing tools/flint/fp16_range_audit.py"
    return 1
  fi
  server_down
  read -r -a extra <<<"$AUDIT_ARGS"
  show "$PY" "$tool" --model "$MODEL_DIR" --tensor-parallel-size 2 "${extra[@]}"
  rc=$?
  case $rc in
    0) log "ok: no layer above the FP16 audit threshold" ;;
    1) fail "FP16 range audit flagged layers (see logs/audit.log)" ;;
    *) fail "audit tool error (exit $rc)" ;;
  esac
  stage_result
}

stage_smoke() {
  need_python || return 1
  if [[ -z $SERVER_URL ]]; then server_down; fi
  if ! server_up smoke; then
    fail "$SERVER_ERROR"
    return 1
  fi
  show curl -sf "$BASE_URL/v1/models" | tee "$STAGE_DIR/models.json"
  echo
  grep -q "\"$SERVED_MODEL\"" "$STAGE_DIR/models.json" ||
    fail "/v1/models does not list '$SERVED_MODEL'"
  run_check chat --warn2 "$PY" "$HELPER" smoke --base-url "$BASE_URL" \
    --model "$SERVED_MODEL" --out "$STAGE_DIR/chat.json" --markdown "$STAGE_DIR/summary.md"
  if [[ -z $SERVER_URL ]]; then
    local log=$RUN_DIR/server/smoke.log
    server_down
    grep -q Traceback "$log" && warn "Traceback in server/smoke.log"
    {
      echo
      echo "Server log excerpts (KV capacity, routes):"
      echo
      echo '```text'
      cat "$RUN_DIR/server/smoke.routes.txt" 2>/dev/null
      echo '```'
    } >>"$STAGE_DIR/summary.md"
  fi
  stage_result
}

stage_quality() {
  need_python || return 1
  if ! server_up default; then
    fail "$SERVER_ERROR"
    return 1
  fi
  run_check probes --warn2 "$PY" "$HELPER" quality --base-url "$BASE_URL" --model "$SERVED_MODEL" \
    --min-math "$MIN_MATH" --min-code "$MIN_CODE" --out "$STAGE_DIR/quality.json" \
    --markdown "$STAGE_DIR/quality.md"
  run_check needle "$PY" "$REPO/tools/flint/needle.py" --base-url "$BASE_URL" \
    --model "$SERVED_MODEL" --lengths "$NEEDLE_LENGTHS" --depths "$NEEDLE_DEPTHS" \
    --report "$STAGE_DIR/needle.json" --markdown "$STAGE_DIR/needle.md"
  {
    cat "$STAGE_DIR/quality.md"
    echo
    cat "$STAGE_DIR/needle.md"
  } >"$STAGE_DIR/summary.md" 2>/dev/null
  stage_result
}

stage_agent() {
  need_python || return 1
  if ! server_up default; then
    fail "$SERVER_ERROR"
    return 1
  fi
  local extra=()
  read -r -a extra <<<"$AGENT_ARGS"
  run_check agent_e2e "$PY" "$REPO/tools/flint/agent_e2e.py" --base-url "$BASE_URL" \
    --model "$SERVED_MODEL" --report "$STAGE_DIR/agent.json" \
    --markdown "$STAGE_DIR/summary.md" "${extra[@]}"
  stage_result
}

# bench_server_args LABEL: fills BENCH_SERVER_ARGS for <par>-<attn>-<moe>.
bench_server_args() {
  local par attn moe rest
  IFS=- read -r par attn moe rest <<<"$1"
  BENCH_SERVER_ARGS=()
  [[ -z $rest ]] || return 1
  case $par in
    tp2) ;;
    pp2) BENCH_SERVER_ARGS+=(--pp) ;;
    *) return 1 ;;
  esac
  case $attn in
    triton) BENCH_SERVER_ARGS+=(--attention-backend TRITON_ATTN) ;;
    flash) BENCH_SERVER_ARGS+=(--attention-backend FLASH_ATTN_V100) ;;
    auto) ;;
    *) return 1 ;;
  esac
  case $moe in
    skinny) BENCH_SERVER_ARGS+=(--kernel-config.sm70_skinny_moe true) ;;
    turbomind) BENCH_SERVER_ARGS+=(--kernel-config.sm70_skinny_moe false) ;;
    auto) ;;
    *) return 1 ;;
  esac
}

bench_configs() {
  local par attn moe
  if [[ -n $BENCH_CONFIGS ]]; then
    echo "$BENCH_CONFIGS"
  elif [[ $BENCH_MATRIX == full ]]; then
    for par in tp2 pp2; do
      for attn in triton flash; do
        for moe in skinny turbomind; do echo "$par-$attn-$moe"; done
      done
    done
  else
    IFS=- read -r par attn moe <<<"$BENCH_BASE"
    echo "$BENCH_BASE"
    case $attn in triton) echo "$par-flash-$moe" ;; flash) echo "$par-triton-$moe" ;; esac
    case $moe in skinny) echo "$par-$attn-turbomind" ;; turbomind) echo "$par-$attn-skinny" ;; esac
    case $par in tp2) echo "pp2-$attn-$moe" ;; pp2) echo "tp2-$attn-$moe" ;; esac
  fi
}

stage_bench() {
  need_python || return 1
  local tokenizer=$TOKENIZER label len configs=() ok_configs=0
  if [[ -z $tokenizer ]]; then
    tokenizer=$HF_CONFIG
    [[ -n $MODEL_DIR && -f $MODEL_DIR/tokenizer_config.json ]] && tokenizer=$MODEL_DIR
  fi
  if [[ -n $SERVER_URL ]]; then
    configs=(external)
  else
    read -r -a configs <<<"$(bench_configs | tr '\n' ' ')"
  fi
  mkdir -p "$STAGE_DIR/raw"
  log "bench configs: ${configs[*]}; input lengths: $BENCH_LENS; tokenizer: $tokenizer"
  for label in "${configs[@]}"; do
    BENCH_SERVER_ARGS=()
    if [[ $label != external ]] && ! bench_server_args "$label"; then
      warn "bad bench config '$label'"
      continue
    fi
    if ! server_up "$label" "${BENCH_SERVER_ARGS[@]}"; then
      warn "$label: $SERVER_ERROR"
      continue
    fi
    local produced=0
    for len in $BENCH_LENS; do
      # One cold request plus BENCH_REPS measured ones, one at a time. The
      # random prompts differ per request and the seed differs per length,
      # so no request hits another one's prefix cache. The summary drops the
      # first (cold) request: warmups and the ready check would reuse its
      # prompt and turn it into a prefix-cache hit.
      if show "$PY" -m vllm.entrypoints.cli.main bench serve --backend openai \
        --base-url "$BASE_URL" --endpoint /v1/completions --model "$SERVED_MODEL" \
        --tokenizer "$tokenizer" --dataset-name random --random-input-len "$len" \
        --random-output-len "$BENCH_OUTPUT_LEN" --random-range-ratio 0 \
        --num-prompts $((BENCH_REPS + 1)) --max-concurrency 1 --ignore-eos \
        --seed "$len" --percentile-metrics ttft,tpot,itl,e2el --disable-tqdm \
        --save-result --save-detailed --result-dir "$STAGE_DIR/raw" \
        --result-filename "$label-$len.json"; then
        produced=$((produced + 1))
      else
        warn "$label at $len tokens: bench serve failed"
      fi
    done
    ((produced)) && ok_configs=$((ok_configs + 1))
    [[ -z $SERVER_URL ]] && server_down
  done
  run_check summary --warn2 "$PY" "$HELPER" bench-summary --raw "$STAGE_DIR/raw" \
    --markdown "$STAGE_DIR/summary.md"
  ((ok_configs)) || fail "no bench configuration produced results"
  stage_result
}

# ------------------------------------------------------- embedded helper
write_helper() {
  cat >"$HELPER" <<'PY'
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probes for tools/flint/sm70_validate.sh (written into the results dir).

Subcommands: topo, verify-build, checkpoint, junit, smoke, quality and
bench-summary. Each prints FAIL:/WARN: lines and exits 0 (pass), 1 (fail)
or 2 (warn only).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

RAW_MARKUP = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>")


def finish(problems: list[str], warnings: list[str]) -> int:
    for problem in problems:
        print(f"FAIL: {problem}")
    for warning in warnings:
        print(f"WARN: {warning}")
    return 1 if problems else 2 if warnings else 0


def cell(value: Any) -> str:
    return str(value).replace("|", "/").replace("\n", " ")


def post(url: str, payload: dict[str, Any], timeout: float = 1800.0) -> Any:
    key = os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


# ------------------------------------------------------------------- topo
def _custom_ar_worker(rank: int, world: int, port: int, out_dir: str) -> None:
    import torch

    report: dict[str, Any] = {"rank": rank}
    try:
        torch.cuda.set_device(rank)
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
            ensure_model_parallel_initialized,
            get_tp_group,
            init_distributed_environment,
        )

        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(
                world_size=world,
                rank=rank,
                distributed_init_method=f"tcp://127.0.0.1:{port}",
                local_rank=rank,
            )
            ensure_model_parallel_initialized(world, 1)
            ca = getattr(get_tp_group().device_communicator, "ca_comm", None)
            enabled = ca is not None and not ca.disabled
            report["custom_ar_enabled"] = enabled
            device = torch.device("cuda", rank)
            correct, used = True, []
            # 1 and 8 decode tokens and a 1 MiB message at hidden size 2048.
            for numel in (2048, 8 * 2048, 256 * 2048):
                x = torch.full((numel,), rank + 1.0, dtype=torch.float16, device=device)
                if enabled:
                    y = ca.custom_all_reduce(x)
                    used.append(y is not None)
                    if y is not None:
                        correct = correct and bool(torch.all(y == 3.0))
                z = tensor_model_parallel_all_reduce(x.clone())
                correct = correct and bool(torch.all(z == 3.0))
            report["custom_ar_used"] = used
            report["allreduce_correct"] = correct
            x = torch.ones(2048, dtype=torch.float16, device=device)
            for _ in range(20):
                tensor_model_parallel_all_reduce(x.clone())
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(200):
                tensor_model_parallel_all_reduce(x.clone())
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / 200
            report["eager_allreduce_4KiB_us"] = round(elapsed * 1e6, 1)
            destroy_model_parallel()
            destroy_distributed_environment()
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    Path(out_dir, f"custom_ar_rank{rank}.json").write_text(json.dumps(report))


def cmd_topo(args: argparse.Namespace) -> int:
    import torch
    import torch.multiprocessing as mp

    problems: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {"torch": torch.__version__, "cuda": torch.version.cuda}
    count = torch.cuda.device_count()
    report["devices"] = [
        {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "capability": "{}.{}".format(*torch.cuda.get_device_capability(i)),
            "memory_gib": round(
                torch.cuda.get_device_properties(i).total_memory / 2**30, 2
            ),
        }
        for i in range(count)
    ]
    if count < 2:
        problems.append(f"{count} visible GPU(s); the TP2 contract needs 2")
    caps = sorted({d["capability"] for d in report["devices"]})
    if caps != ["7.0"]:
        problems.append(f"compute capability {caps}, expected 7.0 (V100)")
    pairs = [(i, j) for i in range(count) for j in range(count) if i != j]
    peer = {f"{i}->{j}": torch.cuda.can_device_access_peer(i, j) for i, j in pairs}
    report["can_device_access_peer"] = peer
    if not all(peer.values()):
        warnings.append(f"driver reports no peer access: {peer}")
    try:
        # The real CUDA IPC + P2P write test custom all-reduce relies on,
        # run in a fresh interpreter, and the cached check built on it.
        from vllm.distributed.device_communicators.all_reduce_utils import (
            gpu_p2p_access_check,
            run_p2p_access_test,
        )

        if pairs:
            results = run_p2p_access_test([i for i, _ in pairs], [j for _, j in pairs])
            ipc = {f"{i}->{j}": bool(r) for (i, j), r in zip(pairs, results)}
            report["cuda_ipc_p2p_test"] = ipc
            report["gpu_p2p_access_check"] = {
                f"{i}->{j}": bool(gpu_p2p_access_check(i, j)) for i, j in pairs
            }
            if not all(ipc.values()):
                warnings.append(
                    f"CUDA IPC P2P test failed {ipc}: custom all-reduce stays "
                    "off and TP2 uses NCCL only"
                )
    except Exception as exc:
        problems.append(f"vLLM P2P test did not run: {type(exc).__name__}: {exc}")
    try:
        from vllm.platforms import current_platform

        physical = [
            current_platform.device_id_to_physical_device_id(i) for i in range(count)
        ]
        report["nvlink_fully_connected"] = bool(
            current_platform.is_fully_connected(physical)
        )
    except Exception as exc:
        report["nvlink_fully_connected"] = f"unknown ({exc})"
    bandwidth = {}
    for i, j in pairs:
        src = torch.empty(256 << 20, dtype=torch.uint8, device=f"cuda:{i}")
        dst = torch.empty(256 << 20, dtype=torch.uint8, device=f"cuda:{j}")
        dst.copy_(src)
        torch.cuda.synchronize(i)
        torch.cuda.synchronize(j)
        start = time.perf_counter()
        for _ in range(10):
            dst.copy_(src)
        torch.cuda.synchronize(i)
        torch.cuda.synchronize(j)
        rate = 10 * src.numel() / (time.perf_counter() - start) / 1e9
        bandwidth[f"{i}->{j}"] = round(rate, 2)
        del src, dst
    torch.cuda.empty_cache()
    report["device_copy_GB_per_s"] = bandwidth
    if count >= 2:
        out_dir = str(Path(args.out).resolve().parent)
        for rank in range(2):
            Path(out_dir, f"custom_ar_rank{rank}.json").unlink(missing_ok=True)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        try:
            mp.spawn(_custom_ar_worker, args=(2, port, out_dir), nprocs=2, join=True)
        except Exception as exc:
            problems.append(f"custom all-reduce probe crashed: {exc}")
        ranks = [
            json.loads(Path(out_dir, f"custom_ar_rank{r}.json").read_text())
            for r in range(2)
            if Path(out_dir, f"custom_ar_rank{r}.json").exists()
        ]
        report["custom_all_reduce"] = ranks
        errors = [r["error"] for r in ranks if "error" in r]
        if errors:
            problems.append(f"custom all-reduce probe: {errors[0]}")
        elif len(ranks) < 2 or not all(r.get("allreduce_correct") for r in ranks):
            problems.append("TP2 all-reduce returned wrong values or no result")
        elif not all(r.get("custom_ar_enabled") for r in ranks):
            warnings.append("vLLM custom all-reduce is disabled for TP2 (pynccl only)")
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    lines = ["| Probe | Result |", "| --- | --- |"]
    for d in report["devices"]:
        lines.append(
            f"| GPU {d['index']} | {d['name']}, SM {d['capability']}, "
            f"{d['memory_gib']} GiB |"
        )
    for key in (
        "can_device_access_peer",
        "cuda_ipc_p2p_test",
        "nvlink_fully_connected",
        "device_copy_GB_per_s",
    ):
        lines.append(f"| {key} | {cell(report.get(key))} |")
    for r in report.get("custom_all_reduce", []):
        detail = {k: v for k, v in r.items() if k != "rank"}
        lines.append(f"| custom all-reduce rank {r['rank']} | {cell(detail)} |")
    Path(args.markdown).write_text("\n".join(lines) + "\n")
    return finish(problems, warnings)


# ---------------------------------------------------------- verify-build
def cmd_verify_build(args: argparse.Namespace) -> int:
    import importlib
    import inspect

    problems: list[str] = []
    warnings: list[str] = []
    facts: dict[str, Any] = {}
    try:
        import torch

        facts["torch"] = torch.__version__
        facts["torch CUDA"] = torch.version.cuda
        if not str(torch.version.cuda).startswith("12.8"):
            warnings.append(
                f"torch CUDA {torch.version.cuda}; the SM70 release uses 12.8"
            )
        if "sm_70" not in torch.cuda.get_arch_list():
            problems.append(f"torch has no sm_70 kernels: {torch.cuda.get_arch_list()}")
        facts["GPUs"] = [
            "{}.{}".format(*torch.cuda.get_device_capability(i))
            for i in range(torch.cuda.device_count())
        ]
    except Exception as exc:
        problems.append(f"torch: {exc}")
    try:
        import vllm

        importlib.import_module("vllm._C")
        facts["vllm"] = vllm.__version__
        import torch

        facts["skinny MoE op"] = hasattr(torch.ops._C, "skinny_moe_qpn_sm70")
    except Exception as exc:
        problems.append(f"vllm native extension: {exc}")
    try:
        import flash_attn_v100
        from flash_attn_v100 import flash_attn_v100_cuda, paged_kv_utils  # noqa: F401

        facts["flash_attn_v100"] = flash_attn_v100.__version__
        facts["SUPPORTS_SINKS"] = bool(
            getattr(flash_attn_v100, "SUPPORTS_SINKS", False)
        )
        for name in ("flash_attn_prefill_paged", "flash_attn_decode_paged"):
            params = inspect.signature(getattr(flash_attn_v100, name)).parameters
            facts[f"{name}(sinks=)"] = "sinks" in params
        if not facts["SUPPORTS_SINKS"]:
            warnings.append(
                "flash_attn_v100.SUPPORTS_SINKS is False: FLASH_ATTN_V100 rejects "
                "sinks, so Flint runs on TRITON_ATTN"
            )
    except Exception as exc:
        problems.append(f"flash_attn_v100: {exc}")
    for module in ("vllm.models.flint", "vllm.models.flint.convert_gguf"):
        try:
            importlib.import_module(module)
            facts[module] = "imports"
        except Exception as exc:
            problems.append(f"{module}: {type(exc).__name__}: {exc}")
    print(json.dumps(facts, indent=2))
    lines = ["| Check | Value |", "| --- | --- |"]
    lines += [f"| {cell(k)} | {cell(v)} |" for k, v in facts.items()]
    Path(args.markdown).write_text("\n".join(lines) + "\n")
    return finish(problems, warnings)


# ------------------------------------------------------------ checkpoint
EXPECTED_CONFIG = {
    "model_type": "flint",
    "num_hidden_layers": 49,
    "hidden_size": 2048,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "sliding_window": 513,
    "num_experts": 256,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 512,
    "vocab_size": 150144,
}


def cmd_checkpoint(args: argparse.Namespace) -> int:
    problems: list[str] = []
    warnings: list[str] = []
    model = Path(args.model_dir)
    cfg = json.loads((model / "config.json").read_text())
    for key, value in EXPECTED_CONFIG.items():
        if cfg.get(key) != value:
            problems.append(f"config.{key}={cfg.get(key)!r}, expected {value!r}")
    if "FlintForCausalLM" not in cfg.get("architectures", []):
        problems.append(f"architectures={cfg.get('architectures')}")
    quant = dict(cfg.get("quantization_config") or {})
    if (model / "hf_quant_config.json").exists():
        quant["hf_quant_config"] = json.loads(
            (model / "hf_quant_config.json").read_text()
        )
    quant_text = json.dumps(quant)
    if "NVFP4" not in quant_text:
        problems.append("no NVFP4 quantization config")
    if '"group_size": 16' not in quant_text:
        warnings.append("quantization config does not state group_size 16")
    for module in ("lm_head", "model.embed_tokens", "*.mlp.gate"):
        if module not in quant_text:
            warnings.append(f"exclude_modules lacks {module}")
    dtypes: dict[str, str] = {}
    total_bytes = 0
    for path in sorted(glob.glob(str(model / "*.safetensors"))):
        total_bytes += os.path.getsize(path)
        with open(path, "rb") as handle:
            (size,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(size))
        for name, meta in header.items():
            if name != "__metadata__":
                dtypes[name] = meta["dtype"]
    counts: dict[str, int] = {}
    for dtype in dtypes.values():
        counts[dtype] = counts.get(dtype, 0) + 1
    if not dtypes:
        problems.append("no safetensors tensors")
    if counts.get("BF16"):
        problems.append(f"{counts['BF16']} BF16 tensors (FP16 policy allows none)")
    sinks = [n for n in dtypes if n.endswith("self_attn.softmax_offset")]
    if len(sinks) != 49:
        problems.append(f"{len(sinks)} softmax_offset (sink) tensors, expected 49")
    if any(dtypes[n] != "F32" for n in sinks):
        problems.append("sink tensors are not F32")
    expected = {
        "lm_head.weight": "F16",
        "lm_head.weight_scale_2": "F32",
        "model.embed_tokens.weight": "F16",
        "model.layers.2.mlp.experts.0.gate_proj.weight": "U8",
        "model.layers.2.mlp.experts.0.gate_proj.weight_scale": "F8_E4M3",
        "model.layers.2.mlp.experts.0.gate_proj.weight_scale_2": "F32",
    }
    for name, dtype in expected.items():
        if name not in dtypes:
            problems.append(f"missing {name}")
        elif dtypes[name] != dtype:
            warnings.append(f"{name} is {dtypes[name]}, expected {dtype}")
    experts = sum(
        1 for n in dtypes if ".mlp.experts." in n and n.endswith("_proj.weight")
    )
    if experts != 47 * 256 * 3:
        problems.append(f"{experts} expert projection weights, expected {47 * 256 * 3}")
    if any(n.endswith("input_scale") for n in dtypes):
        warnings.append("input_scale tensors present (W4A16 expects none)")
    if not any(
        (model / f).exists() for f in ("tokenizer.json", "tokenizer_config.json")
    ):
        warnings.append("no tokenizer files; pass --tokenizer to the server and bench")
    facts = {
        "safetensors GiB": round(total_bytes / 2**30, 2),
        "tensors": len(dtypes),
        "dtypes": counts,
        "sinks": len(sinks),
        "expert projection weights": experts,
        "quantization": quant,
    }
    print(json.dumps(facts, indent=2))
    lines = ["| Check | Value |", "| --- | --- |"]
    lines += [f"| {cell(k)} | {cell(v)} |" for k, v in facts.items()]
    Path(args.markdown).write_text("\n".join(lines) + "\n")
    return finish(problems, warnings)


# ----------------------------------------------------------------- junit
def cmd_junit(args: argparse.Namespace) -> int:
    import xml.etree.ElementTree as ElementTree

    root = ElementTree.parse(args.xml).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    totals = [
        sum(int(s.get(k, 0)) for s in suites)
        for k in ("tests", "failures", "errors", "skipped")
    ]
    print(*totals)
    return 0


# ----------------------------------------------------------------- smoke
def chat(
    base: str,
    model: str,
    content: str,
    max_tokens: int,
    temperature: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    start = time.monotonic()
    response = post(f"{base}/v1/chat/completions", payload)
    choice = response["choices"][0]
    message = choice["message"]
    return {
        "content": message.get("content") or "",
        "reasoning": message.get("reasoning") or message.get("reasoning_content") or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage") or {},
        "seconds": round(time.monotonic() - start, 2),
    }


def cmd_smoke(args: argparse.Namespace) -> int:
    problems: list[str] = []
    warnings: list[str] = []
    try:
        result = chat(args.base_url, args.model, "What is 17*23? Answer briefly.", 4096)
    except (urllib.error.URLError, OSError, KeyError) as exc:
        return finish([f"chat completion failed: {exc}"], [])
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    content = result["content"]
    if not content.strip():
        problems.append(f"empty content (finish_reason={result['finish_reason']})")
    raw = [tag for tag in RAW_MARKUP if tag in content]
    if raw:
        problems.append(f"raw markup {raw} in content: are the parsers enabled?")
    if "391" not in content.replace(",", ""):
        warnings.append("the answer 391 is not in the content")
    lines = [
        "| Field | Value |",
        "| --- | --- |",
        f"| finish_reason | {result['finish_reason']} |",
        f"| seconds | {result['seconds']} |",
        f"| usage | {cell(result['usage'])} |",
        f"| reasoning chars | {len(result['reasoning'])} |",
        f"| content | {cell(content[:200])} |",
    ]
    Path(args.markdown).write_text("\n".join(lines) + "\n")
    return finish(problems, warnings)


# --------------------------------------------------------------- quality
MATH = [
    ("A bakery bakes 7 trays of 12 muffins and sells all but 9 of them. "
     "How many muffins does it sell?", 75),
    ("Tom has $250. He buys 3 books at $18 each and a lamp for $47. "
     "How many dollars does he have left?", 149),
    ("A train travels at 84 km per hour for 3.5 hours. "
     "How many kilometres does it travel?", 294),
    ("A rectangle is 23 cm long and 17 cm wide. "
     "What is its area in square centimetres?", 391),
    ("Five boxes hold 48 pencils each. The pencils are shared equally among "
     "16 students. How many pencils does each student get?", 15),
    ("A $640 bicycle is discounted by 15%, then $12 is added for assembly. "
     "What is the final price in dollars?", 556),
    ("Maria reads 35 pages a day for 2 weeks, then 50 pages a day for 4 days. "
     "How many pages does she read in total?", 690),
    ("A tank holds 1200 litres. It loses 45 litres per hour for 8 hours and is "
     "then refilled with 130 litres. How many litres does it hold now?", 970),
    ("The sum of three consecutive integers is 141. "
     "What is the largest of them?", 48),
    ("A farm has three times as many chickens as cows, and the animals have "
     "150 legs in total. How many cows are there?", 15),
]  # fmt: skip
CODE = [
    ("is_palindrome",
     "`is_palindrome(s: str) -> bool` that returns True when s reads the same "
     "forwards and backwards, ignoring case and every character that is not "
     "a letter or digit",
     "assert is_palindrome('A man, a plan, a canal: Panama')\n"
     "assert not is_palindrome('race a car')\n"
     "assert is_palindrome('')\n"
     "assert is_palindrome('No lemon, no melon')\n"),
    ("fizzbuzz",
     "`fizzbuzz(n: int) -> list[str]` that returns the FizzBuzz sequence for "
     "1..n: 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' "
     "for multiples of both, otherwise the number as a string",
     "r = fizzbuzz(15)\n"
     "assert len(r) == 15 and r[0] == '1' and r[2] == 'Fizz'\n"
     "assert r[4] == 'Buzz' and r[14] == 'FizzBuzz'\n"
     "assert fizzbuzz(0) == []\n"),
    ("merge_intervals",
     "`merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that "
     "merges overlapping or touching closed intervals and returns them "
     "sorted by start",
     "assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == "
     "[[1, 6], [8, 10], [15, 18]]\n"
     "assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]\n"
     "assert merge_intervals([]) == []\n"
     "assert merge_intervals([[5, 7], [1, 2]]) == [[1, 2], [5, 7]]\n"),
    ("word_counts",
     "`word_counts(text: str) -> dict[str, int]` that counts words "
     "case-insensitively, where a word is a maximal run of ASCII letters, and "
     "maps each lowercase word to its count",
     "assert word_counts('The cat and the hat.') == "
     "{'the': 2, 'cat': 1, 'and': 1, 'hat': 1}\n"
     "assert word_counts('') == {}\n"
     "assert word_counts('Go, go, GO!') == {'go': 3}\n"),
    ("nth_prime",
     "`nth_prime(n: int) -> int` that returns the n-th prime number, with "
     "nth_prime(1) == 2",
     "assert nth_prime(1) == 2 and nth_prime(6) == 13\n"
     "assert nth_prime(100) == 541 and nth_prime(1000) == 7919\n"),
]  # fmt: skip
COHERENCE_PROMPT = (
    "In four or five sentences, explain to a new programmer why version "
    "control is useful."
)


def coherence_issues(result: dict[str, Any]) -> list[str]:
    content = result["content"]
    words = re.findall(r"[A-Za-z']+", content)
    issues = []
    if result["finish_reason"] != "stop":
        issues.append(f"finish_reason={result['finish_reason']}")
    if len(words) < 30:
        issues.append(f"only {len(words)} words")
    issues += [f"raw {tag}" for tag in RAW_MARKUP if tag in content]
    if "�" in content:
        issues.append("replacement characters")
    if re.search(r"(.)\1{19,}", content):
        issues.append("a character repeated 20+ times")
    grams = [tuple(w.lower() for w in words[i : i + 3]) for i in range(len(words) - 2)]
    if grams and len(set(grams)) / len(grams) < 0.6:
        issues.append("repetitive text (distinct 3-grams < 60%)")
    if content and sum(c.isalpha() for c in content) / len(content) < 0.6:
        issues.append("mostly non-letter characters")
    return issues


def parse_answer(text: str) -> tuple[float | None, str]:
    tagged = re.findall(r"ANSWER:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if tagged:
        return float(tagged[-1].replace(",", "")), "tagged"
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if numbers:
        return float(numbers[-1].replace(",", "")), "last number"
    return None, "none"


def run_code(code: str, tests: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="flint-code-") as tmp:
        path = Path(tmp, "solution.py")
        path.write_text(f"{code}\n\n{tests}\nprint('ALL_TESTS_PASSED')\n")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        passed = proc.returncode == 0 and "ALL_TESTS_PASSED" in proc.stdout
        return passed, proc.stderr.strip().splitlines()[
            -1
        ] if proc.stderr.strip() else ""


def cmd_quality(args: argparse.Namespace) -> int:
    problems: list[str] = []
    warnings: list[str] = []
    base, model = args.base_url, args.model

    def ask(prompt: str, max_tokens: int, temperature: float | None = None):
        try:
            return chat(base, model, prompt, max_tokens, temperature)
        except (urllib.error.URLError, OSError, KeyError) as exc:
            return {"content": "", "reasoning": "", "finish_reason": f"error: {exc}"}

    greedy = [ask(COHERENCE_PROMPT, 2048, temperature=0.0) for _ in range(2)]
    issues = coherence_issues(greedy[0])
    print("greedy coherence:", greedy[0]["content"][:800])
    if issues:
        problems.append("greedy coherence: " + "; ".join(issues))
    if greedy[0]["content"] != greedy[1]["content"]:
        warnings.append("two greedy runs of the same prompt differ")

    def solve(item: tuple[str, int]) -> dict[str, Any]:
        question, expected = item
        prompt = (
            "Solve the problem. Show brief working, then finish with a final "
            f"line of the form 'ANSWER: <integer>'.\n\nProblem: {question}"
        )
        result = ask(prompt, args.max_tokens)
        got, mode = parse_answer(result["content"])
        return {
            "question": question,
            "expected": expected,
            "got": got,
            "parsed": mode,
            "passed": got is not None and abs(got - expected) < 1e-6,
            "finish_reason": result["finish_reason"],
        }

    def code(item: tuple[str, str, str]) -> dict[str, Any]:
        name, spec, tests = item
        prompt = (
            f"Write a Python function {spec}. Reply with a single ```python code "
            "block that defines the function (plus any imports it needs); do not "
            "include tests or example usage."
        )
        result = ask(prompt, args.max_tokens)
        blocks = re.findall(
            r"```(?:python|py)?[ \t]*\n(.*?)```", result["content"], re.S
        )
        source = blocks[-1] if blocks else result["content"]
        passed, error = run_code(source, tests)
        return {
            "task": name,
            "passed": passed,
            "error": error,
            "finish_reason": result["finish_reason"],
        }

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        math_results = list(pool.map(solve, MATH))
        code_results = list(pool.map(code, CODE))
    math_score = sum(r["passed"] for r in math_results)
    code_score = sum(r["passed"] for r in code_results)
    for r in math_results:
        print(
            f"math {'PASS' if r['passed'] else 'FAIL'}: expected {r['expected']}, "
            f"got {r['got']} ({r['parsed']}, {r['finish_reason']})"
        )
    for r in code_results:
        print(f"code {'PASS' if r['passed'] else 'FAIL'}: {r['task']} {r['error']}")
    if math_score < args.min_math:
        problems.append(f"arithmetic {math_score}/{len(MATH)} < {args.min_math}")
    if code_score < args.min_code:
        problems.append(f"coding {code_score}/{len(CODE)} < {args.min_code}")
    report = {
        "greedy": greedy,
        "coherence_issues": issues,
        "math": math_results,
        "code": code_results,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "| Probe | Result |",
        "| --- | --- |",
        f"| Greedy coherence | {'PASS' if not issues else cell('; '.join(issues))} |",
        f"| Greedy repeat identical | {greedy[0]['content'] == greedy[1]['content']} |",
        f"| Arithmetic | {math_score}/{len(MATH)} (need {args.min_math}) |",
        f"| Coding (executed asserts) | {code_score}/{len(CODE)} "
        f"(need {args.min_code}) |",
    ]
    Path(args.markdown).write_text("\n".join(lines) + "\n")
    return finish(problems, warnings)


# ---------------------------------------------------------- bench-summary
def cmd_bench_summary(args: argparse.Namespace) -> int:
    rows = []
    for path in sorted(glob.glob(os.path.join(args.raw, "*.json"))):
        label, _, length = Path(path).stem.rpartition("-")
        data = json.loads(Path(path).read_text())
        ttfts = data.get("ttfts") or []
        itls = data.get("itls") or []
        inputs = data.get("input_lens") or []
        # Drop the first (cold) request when there are others.
        keep = list(range(1, len(ttfts))) if len(ttfts) > 1 else list(range(len(ttfts)))
        ttft = [ttfts[i] * 1000 for i in keep if ttfts[i]]
        decode = [
            len(itls[i]) / sum(itls[i])
            for i in keep
            if i < len(itls) and sum(itls[i]) > 0
        ]
        rows.append(
            {
                "config": label,
                "target": int(length),
                "input": int(statistics.median(inputs)) if inputs else None,
                "ttft_ms": round(statistics.median(ttft), 1) if ttft else None,
                "decode_tok_s": round(statistics.median(decode), 2) if decode else None,
                "measured": len(keep),
                "failed": data.get("failed"),
            }
        )
    rows.sort(key=lambda r: (r["config"], r["target"]))
    lines = [
        "| Config | Input tokens | TTFT ms (median) | Decode tok/s (median) "
        "| Measured | Failed |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['input']} | {r['ttft_ms']} | {r['decode_tok_s']} | "
            f"{r['measured']} | {r['failed']} |"
        )
    text = "\n".join(lines) + "\n"
    Path(args.markdown).write_text(text)
    print(text)
    Path(args.markdown).with_suffix(".json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    if not rows:
        return finish(["no benchmark results"], [])
    return finish(
        [],
        [
            f"{r['config']} at {r['target']}: no timing"
            for r in rows
            if r["ttft_ms"] is None
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("topo")
    p.add_argument("--out", required=True)
    p.add_argument("--markdown", required=True)
    p = sub.add_parser("verify-build")
    p.add_argument("--markdown", required=True)
    p = sub.add_parser("checkpoint")
    p.add_argument("model_dir")
    p.add_argument("--markdown", required=True)
    p = sub.add_parser("junit")
    p.add_argument("xml")
    for name in ("smoke", "quality"):
        p = sub.add_parser(name)
        p.add_argument("--base-url", required=True)
        p.add_argument("--model", default="flint")
        p.add_argument("--out", required=True)
        p.add_argument("--markdown", required=True)
        p.add_argument("--max-tokens", type=int, default=4096)
        p.add_argument("--min-math", type=int, default=8)
        p.add_argument("--min-code", type=int, default=4)
        p.add_argument("--parallel", type=int, default=2)
    p = sub.add_parser("bench-summary")
    p.add_argument("--raw", required=True)
    p.add_argument("--markdown", required=True)
    args = parser.parse_args()
    handler = {
        "topo": cmd_topo,
        "verify-build": cmd_verify_build,
        "checkpoint": cmd_checkpoint,
        "junit": cmd_junit,
        "smoke": cmd_smoke,
        "quality": cmd_quality,
        "bench-summary": cmd_bench_summary,
    }[args.cmd]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
PY
}

# -------------------------------------------------------------------- main
write_helper
log "results: $RUN_DIR"
log "stages: ${STAGES[*]}"
for stage in "${STAGES[@]}"; do
  run_stage "$stage"
done
server_down
write_summary
if [[ -z ${RESULTS_DIR:-} ]]; then
  ln -sfn "$RUN_DIR" "$REPO/bench_results/flint_sm70/latest" 2>/dev/null
fi
log "summary: $RUN_DIR/summary.md"
cut -f1,2 "$RUN_DIR/stages.tsv" | column -t 2>/dev/null || cut -f1,2 "$RUN_DIR/stages.tsv"
if cut -f2 "$RUN_DIR/stages.tsv" | grep -qx FAIL; then
  exit 1
fi
exit 0
