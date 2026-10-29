"""Search spaces. Every parameter maps to one coordinate of the unit box the optimizers search.

Accepted per-parameter specs:
  Float(low, high, log=False), Int(low, high, log=False), Categorical([...])
  (low, high)            -> Float, or Int if both bounds are ints
  [v1, v2, ...]          -> Categorical (so a GridSearchCV param_grid works as-is)
"""
import math

import numpy as np


class Float:
    def __init__(self, low, high, log=False):
        if not low < high:
            raise ValueError(f'Float needs low < high, got {low}, {high}')
        if log and low <= 0:
            raise ValueError('log scale needs low > 0')
        self.low, self.high, self.log = float(low), float(high), log

    def decode(self, u):
        if self.log:
            return float(math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low))))
        return float(self.low + u * (self.high - self.low))

    def encode(self, v):
        if self.log:
            return (math.log(v) - math.log(self.low)) / (math.log(self.high) - math.log(self.low))
        return (v - self.low) / (self.high - self.low)

    def __repr__(self):
        return f'Float({self.low}, {self.high}, log={self.log})'


class Int(Float):
    def decode(self, u):
        return int(min(self.high, max(self.low, round(super().decode(u)))))

    def __repr__(self):
        return f'Int({int(self.low)}, {int(self.high)}, log={self.log})'


class Categorical:
    """shortcut: one coordinate cut into equal bins (no ordering knowledge); fine for a few choices,
    upgrade to one-hot coordinates if categoricals with many unordered choices matter."""

    def __init__(self, choices):
        self.choices = list(choices)
        if not self.choices:
            raise ValueError('Categorical needs at least one choice')

    def decode(self, u):
        return self.choices[min(int(u * len(self.choices)), len(self.choices) - 1)]

    def encode(self, v):
        return (self.choices.index(v) + 0.5) / len(self.choices)

    def __repr__(self):
        return f'Categorical({self.choices})'


def _as_spec(name, spec):
    if isinstance(spec, (Float, Categorical)):
        return spec
    if isinstance(spec, tuple) and len(spec) == 2:
        lo, hi = spec
        if isinstance(lo, (int, np.integer)) and isinstance(hi, (int, np.integer)):
            return Int(lo, hi)
        return Float(lo, hi)
    if isinstance(spec, (list, np.ndarray, range)):
        return Categorical(spec)
    raise TypeError(f'parameter {name!r}: unsupported spec {spec!r} (use Float, Int, Categorical, (low, high) or a list)')


class Space:
    def __init__(self, space):
        if not isinstance(space, dict) or not space:
            raise TypeError('space must be a non-empty dict {name: spec}')
        self.names = list(space)
        self.specs = [_as_spec(k, v) for k, v in space.items()]

    @property
    def n(self):
        return len(self.specs)

    def decode(self, x):
        x = np.clip(np.asarray(x, float), 0.0, 1.0)
        return {k: s.decode(float(u)) for k, s, u in zip(self.names, self.specs, x)}

    def encode(self, params):
        return np.array([s.encode(params[k]) for k, s in zip(self.names, self.specs)])
