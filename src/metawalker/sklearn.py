"""MetaWalkerCV: a scikit-learn search estimator, drop-in for GridSearchCV / RandomizedSearchCV.

It plugs into BaseSearchCV's own `_run_search` hook (the one GridSearchCV and RandomizedSearchCV use), so
cross-validation, scoring, refit, n_jobs, cv_results_, best_params_ and best_estimator_ behave the same.
"""
import numpy as np
from sklearn.model_selection._search import BaseSearchCV

from .api import minimize


class MetaWalkerCV(BaseSearchCV):
    """Search `param_space` with a learned or classic optimizer, `n_iter` candidate evaluations in total.

    param_space: {name: spec}; specs as in metawalker.space (Float/Int/Categorical, (low, high) or a list),
                 so a GridSearchCV param_grid dict (lists of values) works unchanged.
    method: see metawalker.minimize ('auto', 'rlcma', 'walker', 'walker_hp', 'cmaes', 'tpe', 'random', 'panel').
    """

    _parameter_constraints = {**BaseSearchCV._parameter_constraints,
                              'param_space': 'no_validation', 'n_iter': 'no_validation',
                              'method': 'no_validation', 'random_state': 'no_validation'}

    def __init__(self, estimator, param_space, *, n_iter=50, method='auto', scoring=None, n_jobs=None,
                 refit=True, cv=None, verbose=0, random_state=None, error_score=np.nan, return_train_score=False):
        super().__init__(estimator=estimator, scoring=scoring, n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
                         error_score=error_score, return_train_score=return_train_score)
        self.param_space, self.n_iter, self.method, self.random_state = param_space, n_iter, method, random_state

    def _run_search(self, evaluate_candidates):
        def score(params):
            results = evaluate_candidates([params])
            key = 'mean_test_score' if 'mean_test_score' in results else \
                f'mean_test_{self.refit if isinstance(self.refit, str) else next(iter(self.scorer_))}'
            return results[key][-1]  # nan for failed fits -> counted as a failed trial

        rs = self.random_state
        seed = rs if rs is None or isinstance(rs, (int, np.integer)) \
            else int((rs.integers if hasattr(rs, 'integers') else rs.randint)(2 ** 31))  # Generator or RandomState
        minimize(score, self.param_space, budget=self.n_iter, method=self.method, maximize=True, seed=seed)
