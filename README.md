# MetaWalker: learned optimizers for expensive, noisy, many-parameter searches

These are black-box optimizers that learned how to search. Each one was trained with evolution strategies on
cheap synthetic problems and then frozen, so at run time it needs only numpy, and a policy is a few KB. I built
them to find reward-hacking honeypots: the gridworld level and PPO settings (169 parameters) under which PPO
takes a planted exploit instead of doing the task. They work just as well as an ordinary hyperparameter tuner.

Install with `pip install metawalker` ([PyPI](https://pypi.org/project/metawalker/)), or try it first in the
[Colab demo](https://colab.research.google.com/drive/1nKxKFCsVaD7sqL6rE6NkNPjR3K4qFE5y?usp=sharing): no setup,
a few minutes. The story behind it is in the [write-up](docs/blog_post.md), and the full API in
[docs/package.md](docs/package.md).

## Quick start

```bash
pip install "metawalker[sklearn]"     # [sklearn] adds MetaWalkerCV; plain `pip install metawalker` needs only numpy, scipy, cma
```

**1. Optimize any function.** It takes a dict of parameters and returns a number to minimize
(`maximize=True` if higher is better). Failed configurations can raise or return `nan`; they count as failed trials.

```python
import math
import metawalker as mw

def val_loss(p):  # stands in for "train a model, return its validation loss"
    return (math.log10(p["lr"]) + 3) ** 2 + 0.1 * (p["layers"] - 5) ** 2 + {"relu": 0.2, "gelu": 0.0}[p["act"]]

space = {"lr": mw.Float(1e-5, 1e-1, log=True),  # log-scale float
         "layers": mw.Int(1, 8),                 # integer
         "act": ["relu", "gelu"]}                # categorical

result = mw.minimize(val_loss, space, budget=200, seed=0)  # method="auto"; drop seed for a fresh run
print(result.best_params)  # {'lr': 0.0010005..., 'layers': 5, 'act': 'gelu'}: the optimum (18 of 20 seeds find it)
```

**2. Drop it into scikit-learn** in place of `GridSearchCV` / `RandomizedSearchCV` (same `cv`, `scoring`,
`best_params_`, `best_estimator_`, `cv_results_`):

```python
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from metawalker.sklearn import MetaWalkerCV

X, y = load_digits(return_X_y=True)
search = MetaWalkerCV(SVC(), {"C": mw.Float(1e-3, 1e3, log=True), "gamma": mw.Float(1e-5, 1e-1, log=True)},
                      n_iter=30, cv=3).fit(X, y)
print(search.best_params_, round(search.best_score_, 3))
```

**3. Pick methods for your problem.** Run candidates on your own loss, plus simulated stand-ins shaped like it,
and let `compare()` say how often and how fast each one gets there, and which to use:

```python
res = mw.compare({"mine": (val_loss, space),
                  "flat stand-in": mw.make_problem("plateau", n=20, noise=0.01)},
                 methods=["rlbo", "walker_hp_warm", "rlcma", "cmaes", "random"], budget=200, seeds=3)
print(res.success_table())    # share of runs reaching the target, evals and seconds to get there
panel = res.suggest_panel(3)  # best panel of up to 3 methods (often a single method)
print(panel.summary())
result = mw.minimize(val_loss, space, budget=600, method=panel)
```

`res.plot()` draws the share of runs at the target against evaluations and against seconds (needs matplotlib).

**4. Plug in your own optimizer:** a walker you trained with `mw.train()`, or any function. After `register()`
the name works everywhere a built-in method does (`minimize`, panels, `compare`, `MetaWalkerCV`):

```python
import itertools
import numpy as np

def grid(f, n, budget, rng):  # call f(x) on points of the unit cube [0, 1]^n until the budget is spent
    for x in itertools.product(np.linspace(0, 1, 5), repeat=n):
        if f.evals >= budget:
            break
        f(np.array(x))

mw.register("grid", grid)
print(mw.minimize(val_loss, space, budget=50, method="grid").best_value)
```

### Which method

| Your search | `method=` | Why |
|---|---|---|
| Not sure | `"auto"` (default) | CMA-ES below 20 parameters, the learned CMA-ES from 20 up |
| Many parameters, some structure | `"rlbo"` | strongest honeypots in our search; beat CMA-ES on 91–100% of test problems from 20 parameters up |
| Flat, rare signal (most points score the same) | `"walker_hp_warm"` | random warm-up, then a learned walker; found a 50-D hidden region in 13/20 runs vs 2/20 for CMA-ES |
| Unknown landscape, no time to test | `"panel"` | splits the budget across learned CMA-ES, walker and CMA-ES |
| Each evaluation costs minutes or more, up to ~200 parameters | `"gp"` (needs `metawalker[optuna]`) | Optuna GP; strong per evaluation but ~1 h of its own overhead per 1,000 evaluations |

Every method, and the rest of the API (`certify`, `importance`, `AskTell`, `train`, Optuna sampler), is in
[docs/package.md](docs/package.md).

## Reproduce the research

```bash
git clone https://github.com/Itsme-aniketghosh/MetaWalker.git && cd MetaWalker
pip install -e ".[sklearn,optuna,test]"       # library + tests; add torch for Optuna's GP
pip install "jax[cpu]" flax optax              # only for the honeypot gridworld (experiments/honeypot/)
pytest                                         # library, bit-for-bit regression and experiment tests
python -m experiments.benchmark smoke_hpx      # one-minute self-check of the walker and its new actions
```

`experiments/` holds the scripts behind every number in `results/`. The cluster runs used Northeastern's
Explorer: `sbatch cluster/setup.sbatch` once to build the environment, then `cluster/run.sbatch` (benchmarks and
training) and `cluster/honeypot.sbatch` (honeypot searches); usage examples are in each script's header.
`cluster/rlbo_release.sbatch` checks a trained rlbo policy before it ships in the package.

The regression tests compare every optimizer bit for bit against recorded outputs. Floating point differs across
platforms and math libraries, so they only run where those outputs were recorded (`tests/golden/`) and skip
everywhere else. A plain `pip install` gets OpenBLAS numpy, which is enough to change the numbers.

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

## Proposal → code

Each part of the proposal (`CS4180_project_proposal.md`) and where it lives. `src/metawalker/` is the library
(`pip install -e .`); `experiments/` holds the scripts that produce every result in `results/`. "Written by" follows
the proposal's LLM-assistance disclosure: Aniket wrote the base algorithms; the rest is supporting code (the
experiment scripts and shell scripts).

| Proposal | Code | Written by |
|---|---|---|
| The MDP: one search per episode, budget, noise, invalid points | `src/metawalker/problems.py` (`Problem`) | Aniket |
| Sparse reward: −log10(final regret / random-point value) | `problems.py` (`Problem.score`) | Aniket |
| Dense reward: log improvement per step | `src/metawalker/walker.py` (`navigate`, the `actor` branch) | Aniket |
| Walker (Table 1) | `walker.py`; network layout, action sets, features in `src/metawalker/policy.py` | Aniket |
| RL-controlled CMA-ES (Table 1) | `src/metawalker/cma.py` (`cmaes` with a policy) | Aniket |
| rlbo (Table 1) | `src/metawalker/rlbo.py` | Aniket |
| OpenAI-ES | update step: `src/metawalker/es.py`; training loops: `experiments/benchmark.py` (`train`), `experiments/rlbo_bench.py` | Aniket (update), supporting code (loops) |
| PPO and branching double DQN | `experiments/rlalgos.py` | supporting code |
| Baselines: CMA-ES, 1/5th-rule walkers | `cma.py` (`cmaes`), `walker.py` (`navigate` with no policy) | Aniket |
| Baselines: Optuna GP / TPE, SMAC3, NGOpt, random search | `src/metawalker/baselines.py` | supporting code |
| MetaWalker-Bench: test functions and the flat honeypot-like family | `problems.py`; suites and reports in `experiments/benchmark.py` | Aniket (functions), supporting code (suites) |
| Real honeypot: JAX PPO gridworld, 169 genes, validity checks | `experiments/honeypot/env.py` | supporting code |
| Comparison 1: same number of tries | `experiments/benchmark.py` (`test`, `hp` suites), `experiments/compare_hpo.py`, `experiments/rlbo_bench.py` | supporting code |
| Comparison 2: same running time as GP | `experiments/timematched.py`, `experiments/rerun_gp.py` | supporting code |
| Comparison 3: budgets 300–10,000 | `experiments/benchmark.py` (`budgets` suite) | supporting code |
| Comparison 4: real honeypot, same PPO compute | `experiments/honeypot/search.py`, `results/honeypot/summarize.py` | supporting code |
| Comparison 5: ablations, what the policies learned | ablation variants in `experiments/benchmark.py` (`hpx_norecheck`, `hpx_nojump`); `experiments/analysis/` | supporting code |
| Risk: re-test the best on new seeds | `experiments/honeypot/search.py` (`remeasure`), `src/metawalker/certify.py` | supporting code |
| Risk: Wilson confidence intervals | `experiments/benchmark.py` (`wilson`) | supporting code |
| Risk: self-checks for bugs | `tests/` | supporting code |
| Cluster runs (CPU-only Explorer) | `cluster/*.sbatch` | supporting code |

Everything else is the package's user-facing API, which the proposal doesn't cover and which is all
supporting code: `api.py` (`minimize`, `register`), `sklearn.py` (`MetaWalkerCV`), `asktell.py`, `space.py`,
`compare.py`, `recommend.py`, `importance.py`, `train.py`, and `experiments/demo.py`.

## Project documents

The course proposal is [CS4180_project_proposal.md](CS4180_project_proposal.md) (PDF alongside), and the
project plan is [docs/plan.md](docs/plan.md).

MIT licensed.
