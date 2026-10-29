"""Standard optimizers behind the same interface: random search, Optuna (TPE / GP), SMAC3, Nevergrad NGOpt.
Each takes prob (n, budget, evals, __call__) and rng, and stops when prob.evals reaches prob.budget."""
from pathlib import Path

import numpy as np


def random_search(prob, rng):
    while prob.evals < prob.budget:
        prob(rng.uniform(0, 1, prob.n))


def until_budget(prob):
    """Optuna callback: stop once prob's budget is spent (a problem may not count some evaluations)."""
    return lambda study, trial: study.stop() if prob.evals >= prob.budget else None


def optuna_search(prob, rng, sampler='tpe', **kw):
    """sampler: 'tpe' or 'gp' (Optuna's Gaussian-process BO); kw go to the sampler (e.g. multivariate=True)."""
    try:
        import optuna
    except ImportError as e:
        raise ImportError(f"method={sampler!r} needs optuna: pip install metawalker[optuna]") from e
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cls = {'tpe': optuna.samplers.TPESampler, 'gp': optuna.samplers.GPSampler}[sampler]
    seen = []

    def objective(tr):
        y = prob(np.array([tr.suggest_float(f'x{i}', 0, 1) for i in range(prob.n)]))
        if np.isfinite(y):
            seen.append(y)
            return y
        # failed trial: report it as worse than anything seen, since an infinite value breaks the GP model
        return max(seen) + (max(seen) - min(seen)) + 1.0 if seen else 1e6

    study = optuna.create_study(sampler=cls(seed=int(rng.integers(2 ** 31)), **kw))
    if prob.evals < prob.budget:  # after a warm-up, only what is left
        study.optimize(objective, callbacks=[until_budget(prob)])  # an exception (e.g. a time limit) ends the study


def smac_hpo(prob, rng):
    import shutil, tempfile
    from ConfigSpace import ConfigurationSpace, Float
    from smac import HyperparameterOptimizationFacade, Scenario
    from smac.runhistory.dataclasses import TrialValue
    seed = int(rng.integers(2 ** 31))
    cs = ConfigurationSpace(seed=seed)
    cs.add([Float(f'x{i}', (0.0, 1.0)) for i in range(prob.n)])
    out = tempfile.mkdtemp(prefix='smac_')
    try:
        sc = Scenario(cs, deterministic=True, n_trials=prob.budget, seed=seed, output_directory=Path(out))
        smac = HyperparameterOptimizationFacade(sc, lambda config, seed=0: 0.0, overwrite=True, logging_level=40)
        while prob.evals < prob.budget:  # ask/tell, so TimeLimit propagates instead of being logged as a crash
            info = smac.ask()
            smac.tell(info, TrialValue(cost=float(prob(np.array([info.config[f'x{i}'] for i in range(prob.n)])))))
    finally:
        shutil.rmtree(out, ignore_errors=True)


def ngopt(prob, rng):
    import nevergrad as ng
    param = ng.p.Array(shape=(prob.n,)).set_bounds(0.0, 1.0)
    param.random_state = np.random.RandomState(int(rng.integers(2 ** 31)))
    opt = ng.optimizers.NGOpt(parametrization=param, budget=prob.budget, num_workers=1)
    while prob.evals < prob.budget:
        c = opt.ask()
        opt.tell(c, float(prob(c.value)))
