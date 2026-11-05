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

# a policy you trained on your own kind of problem (the object, or the file it was saved to)
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

## Tune an RL agent, or train an optimizer for your own kind of problem

Tuning an RL agent on your own environment is an ordinary search. Wrap one training run as a function that
returns the score you care about:

```python
import metawalker as mw

def train_agent(p, timesteps=200_000, seed=0):  # one training run on your env; returns mean evaluation return
    agent = make_agent(my_env, lr=p["lr"], gamma=p["gamma"], ent_coef=p["ent_coef"], seed=seed)
    agent.learn(timesteps)
    return evaluate(agent, my_env, episodes=20)

space = {"lr": mw.Float(1e-5, 1e-2, log=True), "gamma": mw.Float(0.9, 0.9999),
         "ent_coef": mw.Float(1e-4, 0.1, log=True)}
res = mw.minimize(train_agent, space, budget=100, maximize=True, recheck=3)  # recheck: RL returns are noisy
res = mw.minimize(train_agent, space, budget=150, method="panel", maximize=True, n_jobs=3)  # 3 methods at once
```

If you run searches like this often (new environments, new agent variants), you can also train the optimizer
itself for your kind of problem. That is how the shipped policies were made: trained on cheap synthetic
functions, then used unchanged on the expensive problem. `train()` takes factories that return a fresh
`(function, space)` on every call, fine-tunes one of the shipped policies with evolution strategies, and returns
a policy you can save, register and compare:

```python
import random

def like_my_problem():  # a fresh cheap instance each call, shaped like yours: 12 knobs, mostly flat, noisy
    return mw.make_problem("plateau", n=12, noise=0.05, seed=random.randrange(10**9))

policy = mw.train([like_my_problem], init="walker_hp", budget=100, generations=50, n_jobs=8)
policy.save("my_tuner.npz")
mw.register("my_tuner", "my_tuner.npz")
res = mw.minimize(train_agent, space, budget=100, method="my_tuner", maximize=True, recheck=3)
```

- `init` picks the policy to fine-tune: `"walker_hp"`, `"walker"`, `"rlcma"` or `"rlbo"`, or `"scratch"` for a
  walker from random weights. Fine-tuning is far faster and more stable than starting over.
- The factories can return anything `minimize()` accepts: a `make_problem()` stand-in, your own analytic
  proxy, or a tiny version of the real thing (a small env, a few hundred steps). Pass `maximize=True` if they
  return a reward. A function with a `.regret(params)` method (as `make_problem()` problems have) is scored by
  true regret, which is much less noisy than the best observed value.
- Cost: each generation runs `2 * pairs * instances_per_gen` searches (320 by default) of `budget` evaluations
  each, so a stand-in has to evaluate in milliseconds. Real RL runs are almost always too slow for this. Train
  on a stand-in, then search the real environment with the result. rlbo costs about 20 times a walker per
  evaluation, so keep its training budgets small.
- Train with about the budget you will search with. Check the result on instances it never trained on, with
  `compare()` against the shipped policies, before you trust it.
- `callback(gen, policy)` runs after every generation (save checkpoints there), and `policy.history` holds
  the mean training score per generation. It is noisy, because every generation draws new instances.
- With `n_jobs > 1`, the functions the factories return are sent to worker processes, so they must be picklable:
  a `make_problem()` problem, an instance of a top-level class, or a top-level function, not a lambda.

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

# 4. train an optimizer for YOUR kind of problem on cheap stand-ins (see "Tune an RL agent" above)
policy = mw.train([like_my_problem], init="rlcma", generations=50, n_jobs=8)
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
| `rlbo` | a learned acquisition policy over cheap surrogates: each step it fits local GPs and a global quadratic, proposes 48 candidates, and the policy picks one | the strongest honeypots of any single method in our 169-parameter search (PPO exploit rate 0.79 vs 0.70 for CMA-ES, on fresh PPO seeds); beat CMA-ES on 91–100% of test problems from 20 parameters up and Optuna GP on 76–100%, in ~10–25 s of overhead per 1000-evaluation run |
| `gp` | Optuna Gaussian-process Bayesian optimization (needs `metawalker[optuna]`) | **expensive** objectives up to ~200 parameters: tied with `rlcma` overall and better at 100–200 parameters, but it costs ~1 hour of its own compute per 1000-evaluation run, and that cost grows fast with evaluations and dimensions |
| `tpe` | Optuna TPE (needs `metawalker[optuna]`) | baseline; `rlcma` beat it on 70–100% of problems |
| `random` | random search | baseline |
| `auto` | `cmaes` below 20 parameters, `rlcma` from 20 up; `gp` up to 200 parameters if you pass `expensive=True` | |
| `[...]` | any list of methods = your own panel: the budget is split across them and the best is kept | |
| your name | anything you added with `register()`: a policy you fine-tuned with `train()`, or a function such as grid search | |
| `panel` | the default panel: `rlbo`, `rlcma`, `cmaes` | when you don't know the landscape; `allocate="race"` gives the better methods more of the budget, `allocate="each"` gives every method all of it. On the honeypot, a split panel (`rlcma`, `walker_hp`, `cmaes`) reached 0.90 on fresh PPO seeds vs 0.79 for `rlbo` alone, at the same compute |

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
- **How a panel shares compute** (`allocate=`):
  - `"split"` (default): the budget is split equally, so a panel costs the same as one method.
  - `"each"`: every method gets the whole budget, and the best result is kept. It's never worse than the panel's
    best method, but it costs `len(panel) x budget` evaluations. With `n_jobs`, the wall-clock time is the same as
    one method's.
  - `"race"`: successive halving. All methods start with an equal slice, the worse half is dropped after each
    round, and the survivors share the rest, so the better methods get more of the budget (3 methods, budget 200:
    the winner gets about 123). The first round is short, so it favors methods that start fast, and on a flat
    objective where nothing has scored yet it can't tell methods apart. `result.allocation` shows who got what.
- `minimize(..., n_jobs=3)` runs a panel's methods at the same time, in threads, with the same result. That helps when your function spends its time
  outside Python (torch, jax, a subprocess, a remote job), and it must be safe to call from several threads.

## Tests

`pytest` from the repo root. `tests/test_regression.py` checks that every optimizer reproduces the
pre-refactor benchmark code bit for bit (`tests/golden.json`).
