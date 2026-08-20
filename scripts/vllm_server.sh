#!/usr/bin/env bash
# Start/stop/inspect the vLLM inference server used for GRPO rollouts.
#
# Mirrors cs336_alignment/vllm_utils.py:start_server so that an externally
# managed server exposes the same routes the training script expects
# (/get_world_size, /init_weight_transfer_engine, /update_weights,
# /pause, /resume). Use with VLLMServer(launch_server=False).
#
# Usage: scripts/vllm_server.sh {start|stop|restart|status|logs|doctor}
# Override defaults via env, e.g. GPU=0 PORT=8001 scripts/vllm_server.sh start

set -uo pipefail

# MODEL_ID is a local snapshot dir so vLLM loads from disk without a hub lookup.
# SERVED_MODEL_NAME decouples the API-facing id from that path: without it the
# only accepted "model" value is the full snapshot path, and clients sending the
# familiar hub id get a 404.
# MODEL_ID="${MODEL_ID:-/home/zezhen/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B/snapshots/a1847dff35000b4271fa70afc5db10fd29fedbdf}"
MODEL_ID="${MODEL_ID:-/home/zezhen/assignment5-alignment/outputs/OLMo-2-0425-1B-sft-v1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-allenai/OLMo-2-0425-1B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
GPU="${GPU:-1}"
SEED="${SEED:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
SHUTDOWN_TIMEOUT="${SHUTDOWN_TIMEOUT:-30}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/outputs/vllm"
LOG_FILE="${LOG_DIR}/server_${PORT}.log"
PGID_FILE="${LOG_DIR}/server_${PORT}.pgid"

# Serve from the repo's own .venv. pyproject pins vllm==0.19.1, which is what
# the training script imports; NCCL weight transfer is a handshake between the
# two processes, so they must be the same version. (A 0.27.1 install also hides
# the RLHF routes behind VLLM_SERVER_DEV_MODE, which 0.19.1 attaches
# unconditionally -- see api_server.py:198-215.)
VLLM_BIN="${VLLM_BIN:-${REPO_ROOT}/.venv/bin/vllm}"
VENV_PYTHON="${VENV_PYTHON:-${REPO_ROOT}/.venv/bin/python}"

# Workaround for installs where vllm's C extension resolves cublasGemmEx at load
# time but torch dlopens libcublas with local scope ("undefined symbol:
# cublasGemmEx"). Preloading puts the libs in the global namespace first.
# Only set when they actually exist -- a bogus LD_PRELOAD warns on every exec.
SITE_PACKAGES="$("${VENV_PYTHON}" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)"
CUBLAS_PRELOAD=""
for _lib_dir in "${SITE_PACKAGES}/nvidia/cu13/lib" "${SITE_PACKAGES}/nvidia/cublas/lib"; do
  for _sover in 13 12; do
    if [[ -e "${_lib_dir}/libcublas.so.${_sover}" ]]; then
      CUBLAS_PRELOAD="${_lib_dir}/libcublas.so.${_sover}:${_lib_dir}/libcublasLt.so.${_sover}"
      break 2
    fi
  done
done

BASE_URL="http://${HOST}:${PORT}"

# Must stay in sync with kill_existing_vllm_server() in vllm_utils.py so both
# this script and the Python helper agree on what "the server" is.
PKILL_PATTERN="vllm serve .* --port ${PORT}"

# Routes the training script depends on. /v1/completions is registered
# unconditionally; the rest only appear with VLLM_SERVER_DEV_MODE=1 and
# --weight-transfer-config, which is the usual cause of a 404.
REQUIRED_ROUTES=(
  /v1/completions
  /get_world_size
  /init_weight_transfer_engine
  /update_weights
  /pause
  /resume
)

log() { printf '[vllm_server] %s\n' "$*"; }
die() { printf '[vllm_server] error: %s\n' "$*" >&2; exit 1; }

# The API server's cmdline matches PKILL_PATTERN, but its EngineCore child
# overwrites its process title to "VLLM::EngineCore" and matches nothing. If the
# parent dies alone, that child is reparented to init and keeps its ~74 GiB of
# GPU memory forever, invisible to any cmdline-based lookup. Track the setsid
# process group instead, which covers the whole tree.
server_pgid() {
  [[ -r "${PGID_FILE}" ]] && cat "${PGID_FILE}" 2>/dev/null
}

server_pids() {
  {
    pgrep -f "${PKILL_PATTERN}" 2>/dev/null
    local pgid
    pgid="$(server_pgid)"
    [[ -n "${pgid}" ]] && pgrep -g "${pgid}" 2>/dev/null
  } | sort -un
}

health_ok() {
  curl -sf --max-time 5 "${BASE_URL}/health" >/dev/null 2>&1
}

# Print the required routes that are missing from the live server's OpenAPI
# schema. Empty output means the server is fully wired up.
missing_routes() {
  local schema
  schema="$(curl -sf --max-time 10 "${BASE_URL}/openapi.json" 2>/dev/null)" || return 1
  printf '%s' "${schema}" | python3 -c '
import json, sys
required = sys.argv[1:]
try:
    paths = set(json.load(sys.stdin).get("paths", {}))
except Exception:
    sys.exit(1)
for route in required:
    if route not in paths:
        print(route)
' "${REQUIRED_ROUTES[@]}"
}

do_start() {
  if health_ok; then
    die "something is already serving ${BASE_URL} (pids: $(server_pids | tr '\n' ' ')). Run 'stop' or 'restart' first."
  fi

  # Nothing is answering /health, but a leftover EngineCore can still be holding
  # the GPU. Starting on top of it fails with "Free memory on device ... is less
  # than desired GPU memory utilization", so clear it out first.
  if [[ -n "$(server_pids)" ]]; then
    log "no healthy server, but found leftover pids: $(server_pids | tr '\n' ' ') -- cleaning up"
    do_stop
  fi

  mkdir -p "${LOG_DIR}"
  log "model=${MODEL_ID} served_as=${SERVED_MODEL_NAME} gpu=${GPU} port=${PORT} log=${LOG_FILE}"

  # setsid puts the server in its own process group so 'stop' can signal the
  # whole tree (vLLM forks workers) without touching this shell.
  # The weight-transfer JSON is single-quoted: it contains double quotes that
  # bash would otherwise strip before argparse ever sees them.
  # VLLM_USE_FLASHINFER_SAMPLER=0 forces the native PyTorch top-k/top-p path.
  # FlashInfer JIT-compiles its sampling kernels on first use, and nvcc on this
  # host cannot find cuda_runtime.h (CUDA toolkit installed without headers),
  # which killed EngineCore during warmup. See topk_topp_sampler.py:45.
  # HF_HUB_OFFLINE=1: no outbound DNS here, so skip the hub lookup entirely.
  # VLLM_HOST_IP: get_ip() probes IPv4 by connecting to 8.8.8.8, which fails on
  # this network-isolated host, so it falls back to a global IPv6 address --
  # but StatelessProcessGroup.create() hardcodes AF_INET (utils.py:462) and
  # cannot bind it. Both processes are local, so pin loopback.
  CUDA_VISIBLE_DEVICES="${GPU}" \
  LD_PRELOAD="${CUBLAS_PRELOAD}" \
  VLLM_HOST_IP=127.0.0.1 \
  VLLM_SERVER_DEV_MODE=1 \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  HF_HUB_OFFLINE=1 \
  VLLM_LOGGING_LEVEL="${LOG_LEVEL}" \
  setsid nohup "${VLLM_BIN}" serve "${MODEL_ID}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --seed "${SEED}" \
    --tensor-parallel-size 1 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --weight-transfer-config '{"backend": "nccl"}' \
    --load-format "${LOAD_FORMAT}" \
    >>"${LOG_FILE}" 2>&1 &

  # setsid makes the server its own process-group leader, so its pid is the pgid
  # that every descendant (EngineCore included) inherits.
  ps -o pgid= -p "$!" 2>/dev/null | tr -d ' ' >"${PGID_FILE}"

  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  while ((SECONDS < deadline)); do
    if health_ok; then
      log "healthy after ~$((SECONDS))s"
      do_status
      return $?
    fi
    if [[ -z "$(server_pids)" ]]; then
      log "server process exited during startup; last 40 log lines:"
      tail -n 120 "${LOG_FILE}" >&2
      die "startup failed; the root cause is usually well above this tail: ${LOG_FILE}"
    fi
    sleep 2
  done

  tail -n 40 "${LOG_FILE}" >&2
  die "timed out after ${STARTUP_TIMEOUT}s waiting for ${BASE_URL}/health"
}

# Signal the whole process group, then the pattern matches, so an EngineCore
# orphaned by a dead API server still gets reaped and releases its GPU memory.
signal_server() {
  local sig="$1" pgid
  pgid="$(server_pgid)"
  [[ -n "${pgid}" ]] && kill "-${sig}" -- "-${pgid}" 2>/dev/null
  pkill "-${sig}" -f "${PKILL_PATTERN}" 2>/dev/null
  return 0
}

do_stop() {
  local pids
  pids="$(server_pids)"
  if [[ -z "${pids}" ]]; then
    log "no server matching '${PKILL_PATTERN}' or pgid $(server_pgid)"
    rm -f "${PGID_FILE}"
    return 0
  fi

  log "SIGTERM -> pids: $(echo "${pids}" | tr '\n' ' ')"
  signal_server TERM

  local deadline=$((SECONDS + SHUTDOWN_TIMEOUT))
  while ((SECONDS < deadline)); do
    [[ -z "$(server_pids)" ]] && { log "stopped"; rm -f "${PGID_FILE}"; return 0; }
    sleep 1
  done

  log "still alive after ${SHUTDOWN_TIMEOUT}s, sending SIGKILL"
  signal_server KILL
  sleep 2
  [[ -z "$(server_pids)" ]] || die "failed to kill $(server_pids | tr '\n' ' ')"
  rm -f "${PGID_FILE}"
  log "stopped (SIGKILL)"
}

do_status() {
  if ! health_ok; then
    log "DOWN: no healthy server at ${BASE_URL}"
    return 1
  fi
  log "UP: ${BASE_URL} (pids: $(server_pids | tr '\n' ' '))"

  local missing
  if ! missing="$(missing_routes)"; then
    log "WARN: could not read ${BASE_URL}/openapi.json"
    return 1
  fi
  if [[ -n "${missing}" ]]; then
    log "MISSING ROUTES:"
    printf '  %s\n' ${missing}
    log "=> relaunch with VLLM_SERVER_DEV_MODE=1 and --weight-transfer-config (this script's 'restart')"
    return 1
  fi
  log "all ${#REQUIRED_ROUTES[@]} required routes present"

  if command -v nvidia-smi >/dev/null 2>&1; then
    log "GPU processes (server should be on GPU ${GPU} only):"
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
  fi
}

# Cheap preflight for the "installed vLLM is too old" case: --weight-transfer-config
# is a recent flag, and without it the weight-sync endpoints never register.
do_doctor() {
  [[ -x "${VLLM_BIN}" ]] || die "no vllm at ${VLLM_BIN} (set VLLM_BIN=...)"
  log "server vllm: ${VLLM_BIN}"
  log "server vllm version: $(LD_PRELOAD="${CUBLAS_PRELOAD}" "${VLLM_BIN}" --version 2>&1 | tail -n 1)"

  # The trainer and the server exchange weights over NCCL, so a version skew
  # between the env running train_evaluate_gsm8k.py and the env serving is a
  # real (and very confusing) failure mode.
  local trainer_version
  trainer_version="$("${VENV_PYTHON}" -c 'import vllm; print(vllm.__version__)' 2>&1 | tail -n 1)"
  log "trainer vllm version: ${trainer_version}"

  # --help=all, not --help: modern vLLM's plain --help is an ~80-line summary of
  # config *groups* and lists no individual flags, so grepping it always misses.
  log "checking 'vllm serve --help=all' for --weight-transfer-config (slow, imports torch)..."
  if LD_PRELOAD="${CUBLAS_PRELOAD}" "${VLLM_BIN}" serve --help=all 2>&1 | grep -q -- '--weight-transfer-config'; then
    log "OK: --weight-transfer-config supported"
  else
    log "MISSING: this vLLM has no --weight-transfer-config; the weight-sync"
    log "         endpoints cannot exist regardless of launch flags. Upgrade vLLM."
  fi

  if LD_PRELOAD="${CUBLAS_PRELOAD}" "${VENV_PYTHON}" -c 'import vllm.distributed.weight_transfer.nccl_engine' 2>/dev/null; then
    log "OK: vllm.distributed.weight_transfer.nccl_engine importable by the trainer"
  else
    log "MISSING: cannot import vllm.distributed.weight_transfer.nccl_engine"
    log "         (init_weight_sync would fail at vllm_utils.py:223)"
  fi
}

case "${1:-}" in
  start)   do_start ;;
  stop)    do_stop ;;
  restart) do_stop && do_start ;;
  status)  do_status ;;
  logs)    tail -n "${LINES:-100}" -f "${LOG_FILE}" ;;
  doctor)  do_doctor ;;
  *)       die "usage: $0 {start|stop|restart|status|logs|doctor}" ;;
esac
