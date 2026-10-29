#!/bin/bash
#SBATCH -p short
#SBATCH -c 32
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%A_%a.out
# Runs `python -m MODULE ARGS...` from the work dir it was submitted in (that dir is $NAV_OUT: policies and results
# are read and written there); the word TASK in ARGS becomes the array task id. Examples, from a work dir:
#   sbatch -J train --array=0-29 ~/autohp/cluster/run.sbatch experiments.benchmark train TASK
#   sbatch -J eval -c 16 --array=0-39 ~/autohp/cluster/run.sbatch experiments.benchmark eval test TASK 40
#   sbatch -J rlbo -c 16 -t 6:00:00 --dependency=singleton --array=0 --export=ALL,RLBO_WALL=19800 \
#          ~/autohp/cluster/run.sbatch experiments.rlbo_bench train TASK     # chained 6 h chunks resume from checkpoints
set -euo pipefail
REPO=${REPO:-$HOME/autohp}  # where this repo is cloned on the cluster
module load python/3.13.5
export PYTHONPATH="$REPO/src:$REPO" NAV_OUT="$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"
"$REPO/.venv/bin/python" -W ignore -m "${@//TASK/${SLURM_ARRAY_TASK_ID:-0}}"
