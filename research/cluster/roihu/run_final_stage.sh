#!/bin/bash
# Submit from research/ after creating logs/, supplying your CSC account with
# sbatch --account=YOUR_PROJECT --export=ALL,STAGE=smoke,LG_RUN_RESEARCH_WORKFLOW=1 cluster/roihu/run_final_stage.sh
#SBATCH --partition=gpumedium
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:gh200:1
set -euo pipefail
: "${STAGE:?Set STAGE=smoke,prepare,delta,train,evaluate}"
if [[ "${LG_RUN_RESEARCH_WORKFLOW:-0}" != 1 ]]; then
    echo "Optional workflow disabled. Set LG_RUN_RESEARCH_WORKFLOW=1 after reading the reproduction notebook." >&2
    exit 2
fi
script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${LG_PROJECT:-}" ]]; then
    if [[ -f "${SLURM_SUBMIT_DIR:-}/scripts/run_final_stage.py" ]]; then
        export LG_PROJECT="$SLURM_SUBMIT_DIR"
    else
        export LG_PROJECT="$script_root"
    fi
fi
module load "${LG_PYTORCH_MODULE:-python-pytorch/2.13}"
if [[ -n "${LG_RESEARCH_ENV:-}" ]]; then
    source "$LG_RESEARCH_ENV/bin/activate"
fi
export PYTHONNOUSERSITE=1
cd "$LG_PROJECT"
python scripts/verify_environment.py
args=("$STAGE")
if [[ -n "${CONFIG_NAME:-}" ]]; then args+=(--config "$CONFIG_NAME"); fi
if [[ -n "${SMOKE_RECEIPT:-}" ]]; then args+=(--receipt "$SMOKE_RECEIPT"); fi
if [[ -n "${MODEL_REGISTRY:-}" ]]; then args+=(--registry "$MODEL_REGISTRY"); fi
python scripts/run_final_stage.py "${args[@]}"
