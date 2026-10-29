# MetaWalker: learned search policies for honeypot and hyperparameter search

The Python package is `metawalker` (`pip install -e .`, then `import metawalker`).

RL-trained black-box optimizers for expensive, noisy, many-parameter searches. The motivating application is
AI-safety "honeypot" search: find the gridworld level and PPO settings (169 parameters) under which a PPO agent
takes a planted exploit instead of the honest task.

## Proposal → code

Each part of the proposal (`proposal_notes.md`) and where it lives. `src/metawalker/` is the library
(`pip install -e .`); `experiments/` holds the scripts that produce every result in `results/`. "Written by" follows
the proposal's LLM-assistance disclosure: Aniket wrote the base algorithms; Claude (Claude Code) wrote the
experiment scripts, the shell scripts and the other supporting code.

| Proposal | Code | Written by |
|---|---|---|
| The MDP: one search per episode, budget, noise, invalid points | `src/metawalker/problems.py` (`Problem`) | Aniket |
| Sparse reward: −log10(final regret / random-point value) | `problems.py` (`Problem.score`) | Aniket |
| Dense reward: log improvement per step | `src/metawalker/walker.py` (`navigate`, the `actor` branch) | Aniket |
| Walker (Table 1) | `walker.py`; network layout, action sets, features in `src/metawalker/policy.py` | Aniket |
| RL-controlled CMA-ES (Table 1) | `src/metawalker/cma.py` (`cmaes` with a policy) | Aniket |
| rlbo (Table 1) | `src/metawalker/rlbo.py` | Aniket |
| OpenAI-ES | update step: `src/metawalker/es.py`; training loops: `experiments/benchmark.py` (`train`), `experiments/rlbo_bench.py` | Aniket (update), Claude (loops) |
| PPO and branching double DQN | `experiments/rlalgos.py` | Claude |
| Baselines: CMA-ES, 1/5th-rule walkers | `cma.py` (`cmaes`), `walker.py` (`navigate` with no policy) | Aniket |
| Baselines: Optuna GP / TPE, SMAC3, NGOpt, random search | `src/metawalker/baselines.py` | Claude |
| MetaWalker-Bench: test functions and the flat honeypot-like family | `problems.py`; suites and reports in `experiments/benchmark.py` | Aniket (functions), Claude (suites) |
| Real honeypot: JAX PPO gridworld, 169 genes, validity checks | `experiments/honeypot/env.py` | Claude |
| Comparison 1: same number of tries | `experiments/benchmark.py` (`test`, `hp` suites), `experiments/compare_hpo.py`, `experiments/rlbo_bench.py` | Claude |
| Comparison 2: same running time as GP | `experiments/timematched.py`, `experiments/rerun_gp.py` | Claude |
| Comparison 3: budgets 300–10,000 | `experiments/benchmark.py` (`budgets` suite) | Claude |
| Comparison 4: real honeypot, same PPO compute | `experiments/honeypot/search.py`, `results/honeypot/summarize.py` | Claude |
| Comparison 5: ablations, what the policies learned | ablation variants in `experiments/benchmark.py` (`hpx_norecheck`, `hpx_nojump`); `experiments/analysis/` | Claude |
| Risk: re-test the best on new seeds | `experiments/honeypot/search.py` (`remeasure`), `src/metawalker/certify.py` | Claude |
| Risk: Wilson confidence intervals | `experiments/benchmark.py` (`wilson`) | Claude |
| Risk: self-checks for bugs | `tests/` | Claude |
| Cluster runs (CPU-only Explorer) | `cluster/*.sbatch` | Claude |

**The rest** is not in the proposal. It is the package's user-facing API, all written by Claude: `api.py`
(`minimize`), `sklearn.py` (`MetaWalkerCV`), `asktell.py`, `space.py`, `compare.py`, `recommend.py`,
`importance.py`, `train.py`, and `experiments/demo.py`.

`tests/` checks the library and the experiments. `tests/test_regression.py` confirms that every optimizer
reproduces the pre-refactor code bit for bit. `cluster/` holds the Slurm scripts.

## Results

### Benchmark functions (`results/reports/`)

Rotated and shifted sphere, ellipsoid, rastrigin, rosenbrock, ackley and levy, plus a honeypot-like plateau
family, at n = 5–500.

| Comparison | Result |
|---|---|
| rlbo vs Optuna GP, equal evaluations (1,000) | rlbo better on 76–100% of instances at every n; 23 s vs 4,296 s per run |
| rlbo vs CMA-ES, equal evaluations | rlbo better on 91–100% for n ≥ 20; tie at n = 5 |
| RL-CMA vs GP, same wall-clock as GP | RL-CMA better on 90–100% of 4,620 runs at every n (plain CMA-ES too) |
| Walkers on the plateau family | about −1.5 vs −0.6 to −0.9 for CMA-ES (log10 regret, lower is better) |

### Real honeypot, 169 parameters (`results/honeypot/`; `python results/honeypot/summarize.py`)

Equal PPO compute: 200 PPO-costing evaluations per search, structurally invalid levels free. The score is the
certified exploit rate: the best of the top candidates, re-measured on fresh PPO seeds. 5 search seeds per method.

| Method | Score |
|---|---|
| **rlbo** (3 policies, 15 searches) | **0.88** |
| CMA-ES | 0.75–0.77 |
| walker with jump / re-check + level cache | 0.69 |
| Optuna GP | 0.55 |

### What the policies learned (`results/analysis/policy_report.md`)

- The walker jumps on 70% of its steps until it finds signal, then searches locally and re-checks its best
  point. On the real honeypot, removing re-check drops it from 0.81 to 0.48, and removing jump to 0.59.
- The budget-aware walkers front-load their gains. That wins small budgets and loses to CMA-ES at 10,000
  evaluations.
- Three silent training bugs are documented in the reports:
  - a seed leak that started walkers next to the optimum;
  - ES rank ties turning into random steps;
  - a BLAS thread pile-up that slowed training about 300×.

## Quick start

```bash
pip install -e ".[sklearn,optuna,test]"          # the library (+ torch for Optuna's GP, jax/flax/optax for the honeypot)
pytest                                            # library, regression and experiment tests
python -m experiments.benchmark smoke_hpx         # quick self-check of the walker and its new actions
```

```python
from metawalker import minimize, Float, Int
result = minimize(train_and_score, {"lr": Float(1e-5, 1e-1, log=True), "layers": Int(1, 8)}, budget=300,
                  maximize=True, recheck=3)
```

See `docs/package.md` for the library API, `proposal_notes.md` for the project proposal and `docs/plan.md` for
the project plan. The cluster runs
used Northeastern's Explorer: run `sbatch cluster/setup.sbatch` once, then use the generic `cluster/run.sbatch`
and `cluster/honeypot.sbatch`. Usage examples are in their headers.

MIT licensed.
