"""Ask/tell interface: you run the evaluations (your own workers, a cluster queue), metawalker proposes the points.

    from metawalker.asktell import AskTell
    with AskTell(space, budget=200, method='rlcma', seed=0) as opt:
        while not opt.done:
            tid, params = opt.ask()
            opt.tell(tid, run_somewhere(params))   # None or nan = failed trial
    opt.result()

The optimizer is the same code minimize() runs, driven from a background thread, so for the same seed and
the same values it asks exactly the points minimize() would evaluate.

shortcut: one trial in flight at a time (the shipped optimizers are sequential, and CMA-ES generations are
fed one point at a time); upgrade to batch asks only if an optimizer gets a true batch mode. For parallel
evaluation, run several AskTell / minimize instances (different seeds or methods) side by side, e.g. with
compare(..., n_jobs=k).
"""
import math
import queue
import threading

from .api import minimize


class _Closed(BaseException):
    """Unwinds the optimizer thread on close(); a BaseException so _Objective doesn't count it as a failed trial."""


_FINISHED = object()


class AskTell:
    def __init__(self, space, budget, method='auto', maximize=False, seed=None):
        self.budget = budget
        self._asks, self._tells = queue.Queue(), queue.Queue()
        self._closed = False
        self._pending = None     # trial id handed out by ask() and not yet told
        self._next = None        # (trial_id, params) waiting for ask(), or _FINISHED
        self._result = self._error = None
        self._n_asked = 0
        self._thread = threading.Thread(target=self._run, args=(space, budget, method, maximize, seed),
                                        daemon=True)
        self._thread.start()
        self._advance()

    def _fn(self, params):
        if self._closed:
            raise _Closed
        self._asks.put(params)
        v = self._tells.get()
        if v is _Closed:
            raise _Closed
        return math.nan if v is None else v

    def _run(self, *args):
        try:
            self._result = minimize(self._fn, *args)
        except _Closed:
            pass
        except BaseException as e:  # surfaced in the caller's thread by _advance()
            self._error = e
        finally:
            self._asks.put(_FINISHED)

    def _advance(self):
        """Wait until the optimizer proposes its next point or finishes, so `done` is exact."""
        item = self._asks.get()
        if item is _FINISHED:
            self._thread.join()
            self._next = _FINISHED
            if self._error is not None:
                raise self._error
        else:
            self._next = (self._n_asked, item)

    @property
    def done(self):
        return self._next is _FINISHED or self._closed

    def ask(self):
        """-> (trial_id, params). Call tell() for it before the next ask()."""
        if self._closed:
            raise RuntimeError('AskTell is closed')
        if self._next is _FINISHED:
            raise RuntimeError(f'budget of {self.budget} evaluations is used up; call result()')
        if self._pending is not None:
            raise RuntimeError(f'trial {self._pending} is still in flight; tell() its value before asking again '
                               '(one trial at a time; run several AskTell instances for parallel evaluation)')
        tid, params = self._next
        self._pending, self._n_asked = tid, self._n_asked + 1
        return tid, dict(params)

    def tell(self, trial_id, value):
        """value: the objective value; None or nan (or inf) = failed trial."""
        if self._closed:
            raise RuntimeError('AskTell is closed')
        if self._pending is None or trial_id != self._pending:
            raise ValueError(f'unknown trial id {trial_id!r}: the only trial in flight is {self._pending!r}')
        self._pending = None
        self._tells.put(None if value is None else float(value))
        self._advance()

    def result(self):
        """The metawalker Result, once the budget is used up."""
        if self._next is not _FINISHED:
            raise RuntimeError('result() is available once done is True (the whole budget has been told)')
        return self._result

    def close(self):
        """Stop the optimizer thread. Safe to call twice; the instance can't be used afterwards."""
        if self._closed:
            return
        self._closed = True
        if self._thread.is_alive():
            self._tells.put(_Closed)  # unblocks the optimizer if it is waiting for a value
            self._thread.join(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


try:
    import optuna
    from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
    from optuna.trial import TrialState
except ImportError:  # MetaWalkerSampler needs optuna; the rest of this module doesn't
    optuna = None


if optuna is not None:
    from .space import Categorical, Float, Int

    class MetaWalkerSampler(optuna.samplers.BaseSampler):
        """Optuna sampler that proposes trials with a metawalker method (via AskTell).

            sampler = MetaWalkerSampler({'lr': FloatDistribution(1e-5, 1e-1, log=True), 'act': CategoricalDistribution(['a', 'b'])},
                                 budget=100, method='cmaes', seed=0)
            optuna.create_study(sampler=sampler).optimize(objective, n_trials=100)

        Limits: a fixed search space given up front (float without step, int with step 1, categorical);
        single objective; study.optimize(n_jobs=1); at most `budget` trials. Parameters outside the space
        are sampled at random. Failed or pruned trials count as failed metawalker trials.
        """

        def __init__(self, search_space, budget, method='auto', seed=None):
            spec = {}
            for k, d in search_space.items():
                if isinstance(d, FloatDistribution) and d.step is None:
                    spec[k] = Float(d.low, d.high, log=d.log)
                elif isinstance(d, IntDistribution) and d.step == 1:
                    spec[k] = Int(d.low, d.high, log=d.log)
                elif isinstance(d, CategoricalDistribution):
                    spec[k] = Categorical(d.choices)
                else:
                    raise ValueError(f'parameter {k!r}: unsupported distribution {d!r}')
            self._space, self._spec, self._budget, self._method, self._seed = search_space, spec, budget, method, seed
            self._at, self._trials = None, {}   # optuna trial number -> AskTell trial id
            self._fallback = optuna.samplers.RandomSampler(seed=seed)

        def infer_relative_search_space(self, study, trial):
            return dict(self._space)

        def sample_relative(self, study, trial, search_space):
            if self._at is None:
                self._at = AskTell(self._spec, self._budget, self._method,
                                   maximize=study.direction == optuna.study.StudyDirection.MAXIMIZE, seed=self._seed)
            tid, params = self._at.ask()
            self._trials[trial.number] = tid
            for k, d in self._space.items():  # guard float round-off at the bounds
                if isinstance(d, FloatDistribution):
                    params[k] = min(max(params[k], d.low), d.high)
            return params

        def sample_independent(self, study, trial, param_name, param_distribution):
            return self._fallback.sample_independent(study, trial, param_name, param_distribution)

        def after_trial(self, study, trial, state, values):
            tid = self._trials.pop(trial.number, None)
            if tid is not None:
                self._at.tell(tid, values[0] if state == TrialState.COMPLETE and values else None)
