"""OpenAI-ES pieces shared by every trainer: tie-aware centred ranks and the antithetic Adam ascent step."""
import numpy as np


def centered_ranks(fit):
    """Ranks in [-0.5, 0.5], ties sharing their average rank. argsort ranks broke ties by index, so a tied
    antithetic pair still pushed theta along -eps: pure noise steps, and on the plateau family most candidates tie."""
    _, inv, cnt = np.unique(fit, return_inverse=True, return_counts=True)
    return (np.cumsum(cnt) - (cnt + 1) / 2)[inv] / (len(fit) - 1) - 0.5


def es_step(theta, m, v, fit, eps, sigma, lr, gen):
    """One ES update from antithetic fitnesses fit = [f(theta + sigma eps), f(theta - sigma eps)] (higher is
    better). Returns the new (theta, m, v)."""
    pairs = len(eps)
    ranks = centered_ranks(fit)
    grad = ((ranks[:pairs] - ranks[pairs:]) @ eps) / (2 * pairs * sigma)
    m = 0.9 * m + 0.1 * grad; v = 0.999 * v + 0.001 * grad ** 2  # Adam ascent
    theta = theta + lr * (m / (1 - 0.9 ** (gen + 1))) / (np.sqrt(v / (1 - 0.999 ** (gen + 1))) + 1e-8)
    return theta, m, v
