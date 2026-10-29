#!/bin/bash
#SBATCH -J nav-test
#SBATCH -p short
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o test-%j.out
# Full check of a checkout: the test suite (incl. the bit-exact regression tests, the GP baseline and the JAX
# honeypot environment) and the benchmark smoke runs. Run from the repo root: sbatch cluster/test.sbatch
set -euo pipefail
module load python/3.13.5
cd "$SLURM_SUBMIT_DIR"
PY="$PWD/.venv/bin/python"  # absolute: the smoke runs cd into a temp dir
export PYTHONPATH="$PWD/src:$PWD" NAV_CPUS=4
$PY -m pytest -q -p no:cacheprovider
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
for p in rlcma_mlp cov_hp mom; do cp src/metawalker/policies/${p}_s0.npy "$W/policy_${p}_s0.npy"; done
(cd "$W" && NAV_OUT="$W" $PY -W ignore -m experiments.benchmark smoke > smoke.log && tail -n 3 smoke.log \
         && NAV_OUT="$W" $PY -W ignore -m experiments.benchmark smoke_vb | tail -n 2 \
         && NAV_OUT="$W" $PY -W ignore -m experiments.benchmark smoke_hpx | tail -n 1)
echo "all checks passed"
