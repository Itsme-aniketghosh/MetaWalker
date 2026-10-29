"""Parameter importance from a search's evaluation history (no extra evaluations).

    from metawalker.importance import importance
    imp = importance(result, maximize=True)
    print(imp.summary())

Each parameter is cut into quantile bins (categoricals: one bin per category) and scored by binned eta², the
share of the objective's variance its bins explain. Scores are the chance-corrected eta² (epsilon²) of each
parameter, normalized to sum to 1. Numeric parameters also get |Spearman rho| and a direction.

Caveat: an optimizer's history is concentrated where the optimizer went, so importance computed from it is
biased toward those regions (a parameter the optimizer fixed early looks unimportant). Histories from
method='random' give the cleanest estimate.
"""
import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .space import Categorical, Space


@dataclass
class ParamImportance:
    name: str
    kind: str                 # 'numeric' or 'categorical'
    score: float              # share of the explained variation, in [0, 1]
    eta2: float               # binned eta²
    eta2_ci: tuple            # bootstrap 95% CI on eta²
    rho: float                # signed Spearman rho of value vs goodness (nan for categoricals)
    direction: str            # 'higher is better', 'lower is better', 'no clear trend', or 'best: <category>'
    best: object              # (low, high) of the best bin for numeric params, the best category otherwise
    best_mean: float          # mean objective in that bin


@dataclass
class Importance:
    scores: dict              # param -> score, sums to ~1
    details: dict             # param -> ParamImportance
    interactions: dict = field(default_factory=dict)  # (a, b) -> eta²(a, b) - eta²(a) - eta²(b), top params only
    n_used: int = 0
    n_failed: int = 0
    explained: float = 0.0    # sum of main-effect eta², capped at 1
    maximize: bool = False

    def summary(self):
        lines = [f'Parameter importance from {self.n_used} trials ({self.n_failed} failed trials excluded); '
                 f'main effects explain ~{self.explained:.0%} of the variance.']
        for i, name in enumerate(sorted(self.scores, key=self.scores.get, reverse=True), 1):
            d = self.details[name]
            best = (f'best range {_fmt(d.best[0])}..{_fmt(d.best[1])}' if d.kind == 'numeric'
                    else f'best category {d.best!r}')
            rho = f', |rho| {abs(d.rho):.2f}' if d.kind == 'numeric' else ''
            trend = '' if d.kind == 'categorical' else f'; {d.direction}'
            lines.append(f'{i:3d}. {name}: {d.score:.0%} of the explained variation '
                         f'(eta^2 {d.eta2:.2f} [{d.eta2_ci[0]:.2f}, {d.eta2_ci[1]:.2f}]{rho}){trend}; {best} '
                         f'(mean {_fmt(d.best_mean)})')
        top = [(k, v) for k, v in sorted(self.interactions.items(), key=lambda kv: -kv[1]) if v > 0][:3]
        if top:
            lines.append('Interactions (eta^2 beyond both main effects, not chance-corrected): '
                         + ', '.join(f'{a}:{b} {v:.2f}' for (a, b), v in top))
        lines.append('Note: from an optimizer\'s history this is biased toward the regions it sampled; '
                     'random-search histories give the cleanest estimate.')
        return '\n'.join(lines)


def _fmt(v):
    return f'{v:.3g}' if isinstance(v, (float, np.floating)) else str(v)


def _rank(a):
    """Ranks with average ties (argsort-based, like scipy.stats.rankdata)."""
    order = np.argsort(a, kind='stable')
    s = a[order]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(a)])
    ranks = np.empty(len(a))
    ranks[order] = np.repeat(starts + (counts - 1) / 2, counts)
    return ranks


def _spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / den) if den > 0 else 0.0


def _eta2(bins, y, k):
    """eta² per row: bins, y are (rows, n) arrays of bin codes in [0, k) and objective values."""
    rows = y.shape[0]
    y = y - y.mean(1, keepdims=True)  # centered, so between-bin SS is sum(total² / count) without cancellation
    flat = (bins + k * np.arange(rows)[:, None]).ravel()
    cnt = np.bincount(flat, minlength=rows * k).reshape(rows, k)
    tot = np.bincount(flat, y.ravel(), minlength=rows * k).reshape(rows, k)
    ss_between = (tot ** 2 / np.maximum(cnt, 1)).sum(1)
    ss_total = (y ** 2).sum(1)
    return np.where(ss_total > 0, np.clip(ss_between / np.where(ss_total > 0, ss_total, 1), 0, 1), 0.0)


def _bin_numeric(x, n_bins):
    if len(np.unique(x)) <= n_bins:
        return np.unique(x, return_inverse=True)[1]
    inner = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.unique(np.searchsorted(inner, x, side='right'), return_inverse=True)[1]  # drop empty bins


def importance(result_or_history, space=None, maximize=None, n_bins=5, n_boot=200, seed=0):
    """Which parameters mattered, from a Result (or a [(params, value)] history). See the module docstring.

    space: the search space (dict or Space); only needed to treat numeric choices of a Categorical as
           categories. Without it, parameters whose values are all numbers are numeric.
    maximize: whether higher objective values are better (sets the direction and best-region readouts);
              None reads it from the Result (False for a bare history).
    Failed trials (nan/inf values) are excluded and counted.
    Importance from an optimizer's history is biased toward the regions the optimizer sampled, so
    random-search histories give the cleanest estimate.
    """
    if maximize is None:
        maximize = bool(getattr(result_or_history, 'maximize', False))
    hist = getattr(result_or_history, 'history', result_or_history)
    y_all = np.array([float(v) for _, v in hist])
    ok = np.isfinite(y_all)
    rows = [p for (p, _), good in zip(hist, ok) if good]
    y = y_all[ok]
    n = len(y)
    if n < 3:
        raise ValueError(f'need at least 3 successful trials, got {n}')
    goodness = y if maximize else -y
    cat_names = set()
    if space is not None:
        sp = space if isinstance(space, Space) else Space(space)
        cat_names = {k for k, s in zip(sp.names, sp.specs) if isinstance(s, Categorical)}
    names = list(rows[0])

    rng = np.random.default_rng(seed)
    boot = rng.integers(n, size=(n_boot, n)) if n_boot > 0 else None
    details, bins_of = {}, {}
    for name in names:
        vals = [r[name] for r in rows]
        numeric = name not in cat_names and all(
            isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) for v in vals)
        if numeric:
            x = np.array(vals, float)
            b = _bin_numeric(x, n_bins)
        else:
            codes = {}
            b = np.array([codes.setdefault(v, len(codes)) for v in vals])
            cats = list(codes)
        k = int(b.max()) + 1
        bins_of[name] = (b, k)
        eta2 = float(_eta2(b[None], y[None], k)[0])
        ci = tuple(np.percentile(_eta2(b[boot], y[boot], k), [2.5, 97.5])) if boot is not None else (eta2, eta2)
        cnt = np.bincount(b, minlength=k)
        gmean = np.bincount(b, goodness, minlength=k) / np.maximum(cnt, 1)
        best_bin = int(np.argmax(np.where(cnt > 0, gmean, -np.inf)))
        best_mean = float(gmean[best_bin] if maximize else -gmean[best_bin])
        if numeric:
            rho = _spearman(x, goodness)
            # shortcut: |rho| * sqrt(n - 1) < 2 is a rough 5% test of "no monotone trend"
            direction = ('no clear trend' if abs(rho) * math.sqrt(n - 1) < 2
                         else 'higher is better' if rho > 0 else 'lower is better')
            sel = x[b == best_bin]
            best = (float(sel.min()), float(sel.max()))
        else:
            rho, best = math.nan, cats[best_bin]
            direction = f'best: {best!r}'
        details[name] = ParamImportance(name, 'numeric' if numeric else 'categorical', 0.0, eta2,
                                        (float(ci[0]), float(ci[1])), rho, direction, best, best_mean)

    # chance-corrected eta² (epsilon²): eta² of a useless parameter with k bins is ~(k - 1) / (n - 1)
    adj = {nm: max(0.0, d.eta2 - (bins_of[nm][1] - 1) * (1 - d.eta2) / max(n - bins_of[nm][1], 1))
           for nm, d in details.items()}
    if sum(adj.values()) == 0:
        adj = {nm: d.eta2 for nm, d in details.items()}
    total = sum(adj.values())
    for nm, d in details.items():
        d.score = adj[nm] / total if total > 0 else 1.0 / len(details)
    scores = {nm: d.score for nm, d in details.items()}

    interactions = {}
    top = sorted(scores, key=scores.get, reverse=True)[:5]
    for a, c in combinations(top, 2):
        (ba, ka), (bc, kc) = bins_of[a], bins_of[c]
        joint = float(_eta2((ba * kc + bc)[None], y[None], ka * kc)[0])
        interactions[(a, c)] = joint - details[a].eta2 - details[c].eta2
    explained = min(1.0, sum(d.eta2 for d in details.values()))
    return Importance(scores, details, interactions, n, int((~ok).sum()), explained, maximize)
