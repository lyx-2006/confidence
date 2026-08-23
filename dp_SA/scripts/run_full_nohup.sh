#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p dp_SA/outputs/logs
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec python -u -m dp_SA.run_pipeline --resume > dp_SA/outputs/logs/pipeline.log 2>&1
