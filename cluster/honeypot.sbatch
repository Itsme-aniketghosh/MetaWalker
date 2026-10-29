#!/bin/bash
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 20:00:00
#SBATCH -o logs/%x-%A_%a.out
# One honeypot search per array task: task id -> (METHOD_LIST[id % k], search seed id / k). Needs SPACE and BUDGET
# (and any HP_* options) via --export. Submit from a work dir holding the policies; c30xx nodes lack AVX2 (jaxlib):
#   sbatch -J hp --array=0-9 --exclude=<c30xx nodes> \
#          --export=ALL,SPACE=manyh,BUDGET=200,HP_FREE=1,HP_CACHE=1,METHOD_LIST="cmaes rlbo:s0" ~/autohp/cluster/honeypot.sbatch
set -euo pipefail
: "${BUDGET:?set BUDGET}" "${SPACE:?set SPACE}" "${METHOD_LIST:?set METHOD_LIST}"
read -ra METHODS <<< "$METHOD_LIST"
m=${METHODS[$((SLURM_ARRAY_TASK_ID % ${#METHODS[@]}))]}
s=$((SLURM_ARRAY_TASK_ID / ${#METHODS[@]}))
REPO=${REPO:-$HOME/autohp}  # where this repo is cloned on the cluster
module load python/3.13.5
export PYTHONPATH="$REPO/src:$REPO" NAV_OUT="$SLURM_SUBMIT_DIR" XLA_FLAGS="--xla_force_host_platform_device_count=1"
cd "$SLURM_SUBMIT_DIR"
"$REPO/.venv/bin/python" -W ignore -m experiments.honeypot.search "$SPACE" "$m" "$s" "$BUDGET"
