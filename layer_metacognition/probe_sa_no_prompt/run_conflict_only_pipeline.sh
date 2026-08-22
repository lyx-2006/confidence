#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_ROOT="${REPO_ROOT}/layer_metacognition/output/Final_v4_run_no_sa"
PREFLIGHT_DIR="${REPO_ROOT}/layer_metacognition/output/Final_v4_run_no_sa_preflight/baseline"
FULL_DIR="${OUTPUT_ROOT}/baseline"
SMOKE_DIR="${FULL_DIR}/stage_no_sa_prediction_probe_smoke"
FORMAL_DIR="${FULL_DIR}/stage_no_sa_prediction_probe"
LOG_PATH="${OUTPUT_ROOT}/conflict_only_pipeline.log"
PID_PATH="${OUTPUT_ROOT}/conflict_only_pipeline.pid"

mkdir -p "${OUTPUT_ROOT}"
if [[ -f "${PID_PATH}" ]]; then
    previous_pid="$(tr -d '[:space:]' < "${PID_PATH}")"
    if [[ "${previous_pid}" =~ ^[0-9]+$ ]] && kill -0 "${previous_pid}" 2>/dev/null; then
        echo "An active conflict-only pipeline already exists: PID ${previous_pid}" >&2
        exit 2
    fi
fi
printf '%s\n' "$$" > "${PID_PATH}"
exec > >(tee -a "${LOG_PATH}") 2>&1

cleanup() {
    rm -f "${PID_PATH}"
}
trap cleanup EXIT

stage() {
    local name="$1"
    shift
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START ${name}"
    if "$@"; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] END ${name} exit=0"
    else
        local status=$?
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] END ${name} exit=${status}; pipeline stopped"
        exit "${status}"
    fi
}

stage preflight_validation \
    python -u -m layer_metacognition.probe_sa_no_prompt.panl_preflight \
    --experiment-dir "${PREFLIGHT_DIR}" \
    --layers 27 \
    --expected-items 2

source_args=(
    python -u -m layer_metacognition.run_v3_v4_source_experiment
    --output-dir "${OUTPUT_ROOT}"
    --versions v4
    --attribution-mode none
    --source-prompt-variant baseline
    --conditions conflict_easy conflict_hard
    --analysis_mode LMhead
    --skip-attention
    --skip-layer-readout
    --skip_confidence
    --save_hidden_state 10 12 14 16 18 20 22 24 26 27
    --save_hidden_state_positions ptnl pit ac lat panl
)
if [[ -f "${FULL_DIR}/config.json" && "$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' "${FULL_DIR}/progress.json" 2>/dev/null || true)" != "complete" ]]; then
    source_args+=(--resume)
fi
if [[ ! -f "${FULL_DIR}/progress.json" ]] || [[ "$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' "${FULL_DIR}/progress.json" 2>/dev/null || true)" != "complete" ]]; then
    stage no_sa_capture "${source_args[@]}"
fi
if [[ ! -f "${FULL_DIR}/progress.json" ]] || [[ "$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' "${FULL_DIR}/progress.json" 2>/dev/null || true)" != "complete" ]]; then
    echo "no-SA capture did not produce progress.status=complete" >&2
    exit 1
fi

smoke_args=(
    python -u -m layer_metacognition.probe_sa_no_prompt.run_no_sa_prediction_probe
    --joint-experiment-dir "${REPO_ROOT}/layer_metacognition/output/Final_v4_run_sa_prediction/answer_basis_9"
    --no-sa-experiment-dir "${FULL_DIR}"
    --split-assignments "${REPO_ROOT}/layer_metacognition/output/Final_v4_run_sa_prediction/answer_basis_9/stage_sa_prediction_probe/split_assignments.json"
    --output-dir "${SMOKE_DIR}"
    --layers 10
    --positions ac
    --max-samples 25
    --bootstrap-repeats 100
    --device auto
)
if [[ -f "${SMOKE_DIR}/run_config.json" ]]; then
    smoke_args+=(--resume)
fi
if [[ ! -f "${SMOKE_DIR}/summary.json" ]]; then
    stage no_sa_smoke_probe "${smoke_args[@]}"
fi
[[ -f "${SMOKE_DIR}/summary.json" ]] || { echo "Smoke probe summary missing" >&2; exit 1; }

formal_args=(
    python -u -m layer_metacognition.probe_sa_no_prompt.run_no_sa_prediction_probe
    --joint-experiment-dir "${REPO_ROOT}/layer_metacognition/output/Final_v4_run_sa_prediction/answer_basis_9"
    --no-sa-experiment-dir "${FULL_DIR}"
    --split-assignments "${REPO_ROOT}/layer_metacognition/output/Final_v4_run_sa_prediction/answer_basis_9/stage_sa_prediction_probe/split_assignments.json"
    --output-dir "${FORMAL_DIR}"
    --device auto
)
if [[ -f "${FORMAL_DIR}/run_config.json" ]]; then
    formal_args+=(--resume)
fi
if [[ ! -f "${FORMAL_DIR}/summary.json" ]]; then
    stage formal_no_sa_probe "${formal_args[@]}"
fi
[[ -f "${FORMAL_DIR}/summary.json" ]] || { echo "Formal probe summary missing" >&2; exit 1; }
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] PIPELINE COMPLETE"
