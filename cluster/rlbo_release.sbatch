#!/bin/bash
#SBATCH -J rlbo-release
#SBATCH -p short
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o rlbo-release-%j.out
# Puts the trained rlbo policy into the package and checks it, before a PyPI release. Run from the repo root
# (after git pull), pointing RLBO_POLICY at the policy the honeypot results used:
#   sbatch --export=ALL,RLBO_POLICY=$HOME/<work dir>/rlbo_policy_s0.npy cluster/rlbo_release.sbatch
# Then copy src/metawalker/policies/rlbo_s0.npy back to your machine (or commit it from here).
set -euo pipefail
: "${RLBO_POLICY:?set RLBO_POLICY to the trained rlbo_policy_s0.npy}"
module load python/3.13.5
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src:$PWD"
.venv/bin/python -W ignore cluster/check_rlbo.py "$RLBO_POLICY"
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_metawalker.py
echo "rlbo release check passed"
