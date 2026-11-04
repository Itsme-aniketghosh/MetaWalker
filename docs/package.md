# metawalker

Learned black-box optimizers for **big, complex search spaces**, packaged as a drop-in replacement for
scikit-learn's `GridSearchCV` / `RandomizedSearchCV` and as a plain `minimize()` function.

The optimizers were trained once, with evolution strategies, on cheap synthetic problems. After that they run
on numpy alone, and each policy is a few KB.

```bash
pip install "metawalker[sklearn]"        # from the repo: pip install -e ".[sklearn]"
```

**Try it with no setup:** [Colab demo](https://colab.research.google.com/drive/1nKxKFCsVaD7sqL6rE6NkNPjR3K4qFE5y?usp=sharing).
Code and experiments: [GitHub](https://github.com/Itsme-aniketghosh/MetaWalker).

## Use it

```python
from metawalker import minimize, Float, Int, Categorical

space = {"lr": Float(1e-5, 1e-1, log=True), "layers": Int(1, 8), "act": ["relu", "tanh"], "dropout": (0.0, 0.5)}
result = minimize(train_and_score, space, budget=300, method="auto", maximize=True, recheck=3)
result.best_params, result.best_value, result.recheck_values
```

As a scikit-learn search estimator (it uses the same internal hook as GridSearchCV, so `cv`, `scoring`,
`refit`, `n_jobs`, `cv_results_`, `best_params_` and `best_estimator_` all behave the same):

```python
from metawalker.sklearn import MetaWalkerCV

search = MetaWalkerCV(model, {"C": Float(1e-3, 1e3, log=True), "kernel": ["rbf", "linear"]},
                     n_iter=100, method="rlcma", cv=5).fit(X, y)
```

A GridSearchCV `param_grid` (a dict of value lists) works unchanged as `param_space`.

## Choose your own panel: test methods on your own losses

You know your problem best, so the package helps you measure which method fits it. Give `compare()` your own
loss functions, and optionally some simulated stand-ins shaped like your problem. It runs every method several
times on each, with the same budget, and reports average rank, head-to-head win rates and time per run:

```python
from metawalker import compare, make_problem, minimize

res = compare({"my_loss": (my_fn, my_space),
               "stand-in": make_problem("plateau", n=80, noise=0.01)},  # sphere, ellipsoid, rastrigin,
              methods=["rlcma", "walker_hp", "cmaes", "gp"],          # rosenbrock, ackley, levy, plateau
              budget=200, seeds=5)
print(res.summary())          # average rank, win rates (* = clear: 95% bootstrap interval excludes 0.5), time
print(res.success_table())    # how often each method reaches a target, and how many evals / seconds it took
print(res.budget_table())     # average rank at 1/8, 1/4, 1/2 and all of the budget, from the same runs
res.plot()                    # share of runs at the target vs evaluations and vs seconds (needs matplotlib)

panel = res.suggest_panel(3)  # best panel of up to 3 methods; often one method with the whole budget
print(panel.summary())
minimize(my_fn, my_space, budget=600, method=panel)             # it is a list, so pass it straight in
```

The simulated problems come from the same generator as our benchmark: rotated and shifted, with a known
optimum, so `compare()` scores them by true regret rather than by lucky noisy readings.

**Success** means a run ends at or below a target score and stays there. The default target, per problem,
is the score the best quarter of all runs reached. Pass `target=` a number, or `{problem: number}`, to set
your own. Seconds include the optimizer's own time, so a slow optimizer is penalized as it would be in
practice.

`suggest_panel(k)` scores a panel of j methods on each run's first `budget // j` evaluations, because that is
what a panel gives each method. It tries every size from 1 to k and keeps the one that reaches the target on
the most runs. Splitting the budget often hurts when the signal is rare, so the answer is sometimes a single
method. Rankings at smaller budgets, and the panel scores, come from prefixes of the full runs. A learned or
CMA-based method plans for its full budget, so these are approximations; rerun `compare()` at the budget you
will use to confirm.

## Bring your own optimizer

`register()` adds any optimizer under a method name. After that, the name works everywhere a built-in one does:
`minimize()`, panels, `compare()`, `MetaWalkerCV` and `AskTell`. Decoding into your space, failed trials,
caching, `warmup` and `recheck` are handled for you.

```python
import itertools, numpy as np
import metawalker as mw

# a walker you trained on your own kind of problem (the object, or the file it was saved to)
mw.register("my_walker", "my_optimizer.npz")

# or any function: call f(x) on points in the unit cube [0, 1]^n until f.evals reaches budget (lower is better)
def grid(f, n, budget, rng):
    for x in itertools.product(np.linspace(0, 1, 5), repeat=n):
        if f.evals >= budget:
            break
        f(np.array(x))
mw.register("grid", grid)

mw.compare({"mine": (my_fn, my_space)}, methods=["my_walker", "grid", "rlcma"], seeds=5)
mw.minimize(my_fn, my_space, budget=300, method=["my_walker", "cmaes"])     # in a panel
MetaWalkerCV(model, space, n_iter=50, method="grid").fit(X, y)              # in a scikit-learn search
```

With `compare(n_jobs > 1)`, define your function at the top level of a module so it can be sent to the worker
processes, the same rule as for the objective.

## What else it does for you

```python
import metawalker as mw

# 1. don't know your landscape? probe it (~8% of the budget), then let it pick from the benchmark's rules
res = mw.minimize_auto(my_fn, my_space, budget=500)
print(res.recommendation.reasons)   # e.g. "92% of probes returned the same value: a plateau objective -> walker_hp"

# 2. a best result you can trust: re-test the top candidates on fresh evaluations, pick by mean, report a CI
cert = mw.certify(my_fn, res, top_k=5, repeats=5)
print(cert.summary())               # cert.shrinkage = how much the raw best score was inflated by noise

# 3. which parameters mattered, which direction helps, and the best range (free, from the search history)
print(mw.importance(res).summary())

# 4. train an optimizer specialized to YOUR kind of problem, on cheap simulated stand-ins
policy = mw.train([lambda: my_simulator_factory()], init="walker_hp", generations=50, n_jobs=8)
policy.save("my_optimizer.npz"); policy.minimize(my_fn, my_space, budget=300)

# 5. evaluate on your own workers / cluster, or plug into an existing Optuna study
with mw.AskTell(my_space, budget=200, method="rlcma") as at:
    while not at.done:
        tid, params = at.ask(); at.tell(tid, run_somewhere(params))
from metawalker.asktell import MetaWalkerSampler   # optuna.create_study(sampler=MetaWalkerSampler(dists, budget=200))

# 6. compare methods on your own losses, in parallel
mw.compare({"mine": (my_fn, my_space)}, methods=["rlcma", "gp", "cmaes"], seeds=5, n_jobs=8)
```

Notes:
- `compare(n_jobs>1)` and `train(n_jobs>1)` need picklable functions, plus an `if __name__ == "__main__":` guard on Windows.
- `certify()` and `importance()` read the search direction from the Result.
- Importance computed from an optimizer's history is biased toward the regions that optimizer sampled. Random-search histories give the cleanest estimate.

## Which method

| method | what it is | where it did best in our benchmark |
|---|---|---|
| `rlcma` | CMA-ES whose step size and restarts are set by a learned policy | 50–500 parameters: beat CMA-ES on 72–99% of test problems and TPE on 95%+ |
| `walker` | a learned multi-start walker; one small network shared by every parameter, so it runs at any size | tied CMA-ES overall, won at 500 parameters; ~1 s of optimizer overhead |
| `walker_hp` | the walker, also trained on plateau-shaped objectives | flat, mostly-zero objectives (e.g. searching for RL honeypots); fastest to a valid high-exploit level in a 169-parameter honeypot search |
| `walker_hp_warm` | `walker_hp` after a Latin-hypercube warm-up of 20% of the budget (`warmup=0.2`) | **recommended for rare-signal objectives**, where only a tiny region scores at all: on a 0.5%-live plateau it found the region in 9/10 runs, against 1/10 (10 parameters) and 5/10 (50) for `walker_hp` |
| `walker_rule` | hand-coded walker distilled from the learned one: move each coordinate with p = 0.6, pull 0.5 toward the best walker, step size `s *= exp(0.5 (success EMA - 0.35))` | as good as `walker`, with no network (mean rank 1.69 vs 1.83 over 12 problems) |
| `cmaes` | CMA-ES with IPOP restarts (pycma) | small, noisy problems (under ~20 parameters) |
| `cmaes_rule` | CMA-ES with the hand rule distilled from `rlcma`: sigma × 0.8 after a generation that did not improve, unless stalled 5+ generations or the population is flat | ties `rlcma` (wins 50%) and beats `cmaes` on 78% of runs; `rlcma` stays ahead on Levy/Rastrigin at 10 parameters |
| `rlbo` | a learned acquisition policy over cheap surrogates: each step it fits local GPs and a global quadratic, proposes 48 candidates, and the policy picks one | the strongest honeypots in our 169-parameter search (PPO exploit rate 0.88 vs 0.76 for CMA-ES); beat CMA-ES on 91–100% of test problems from 20 parameters up and Optuna GP on 76–100%, in ~10–25 s of overhead per 1000-evaluation run |
| `gp` | Optuna Gaussian-process Bayesian optimization (needs `metawalker[optuna]`) | **expensive** objectives up to ~200 parameters: tied with `rlcma` overall and better at 100–200 parameters, but it costs ~1 hour of its own compute per 1000-evaluation run, and that cost grows fast with evaluations and dimensions |
| `tpe` | Optuna TPE (needs `metawalker[optuna]`) | baseline; `rlcma` beat it on 70–100% of problems |
| `random` | random search | baseline |
| `auto` | `cmaes` below 20 parameters, `rlcma` from 20 up; `gp` up to 200 parameters if you pass `expensive=True` | |
| `[...]` | any list of methods = your own panel: the budget is split across them and the best is kept | |
| your name | anything you added with `register()`: a walker you trained, or a function such as grid search | |
| `panel` | the default panel: `rlcma`, `walker_hp`, `cmaes` | when you don't know the landscape |

Any method also takes `warmup=f`. It spends that fraction of the budget on Latin-hypercube points first, and
those points count toward the budget and appear in the history. The walkers then start from the best 4 points
and the CMA-based methods from the best one, while the other methods simply run on what is left. Use it when the
signal is rare: without it the walkers shrink and restart instead of exploring, so plain random search finds a
tiny live region more often than they do. The distilled rules come from the policy study (`results/analysis/policy_report.md`). The numbers
above are medians over 3 seeds at budget 1000, from sphere, Rosenbrock, Rastrigin, Ackley, Levy and plateau at
10 and 50 parameters. `compare()` accepts all of these methods.

**Benchmark scope.** Rotated and shifted synthetic functions (sphere, ellipsoid, Rastrigin, Rosenbrock, plus
held-out Ackley and Levy) with and without noise, a honeypot-like plateau family, and a real RL honeypot
search. Budgets were 300–10,000 evaluations. Each learned method was trained 3 times from scratch, with
results reported per training run. A comparison against Optuna GP, Optuna multivariate TPE, SMAC3 and
Nevergrad NGOpt is in progress.

## Behavior to know

- **Invalid configurations:** if your function raises or returns nan/inf, that counts as a failed trial
  (`result.n_failed`). The search keeps going.
- **Noisy objectives:** the best observed value is biased upward (the winner's curse). Use `recheck=k` to
  re-evaluate the winner k more times.
- **Categorical parameters** are one coordinate split into equal bins. That's fine for a few choices, but
  unordered categoricals with many choices are a known weak spot.
- **1-parameter searches** with the CMA-based methods add an ignored dummy coordinate, because pycma needs 2+.
- The `panel` budget is split equally, with no racing between methods.

## Tests

`pytest` from the repo root. `tests/test_regression.py` checks that every optimizer reproduces the
pre-refactor benchmark code bit for bit (`tests/golden.json`).
