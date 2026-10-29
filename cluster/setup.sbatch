#!/bin/bash
#SBATCH -J nav-setup
#SBATCH -p short
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 01:30:00
#SBATCH -o setup-%j.out
# Builds $REPO/.venv with everything the experiments use. Run from the repo root: sbatch cluster/setup.sbatch
# (the login node kills heavy jobs such as venv builds, so this runs as a job).
set -euo pipefail
module load python/3.13.5
cd "$SLURM_SUBMIT_DIR"
[ -x .venv/bin/pip ] || { rm -rf .venv; python3 -m venv .venv; }
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -q -e ".[sklearn,optuna,test]" "jax[cpu]" flax optax nevergrad smac pandas
.venv/bin/python cluster/patch_nevergrad.py
.venv/bin/python -c "import metawalker, jax, torch, optuna, smac, nevergrad; print('env ok')"
