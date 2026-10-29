# Policy interpretability, redone on the leak-free policies (2026-10-25)

The first study (not kept; superseded by this one) used policies trained with the seed leak (training started each walker
next to the optimum). This redo covers the policies that now matter:
- `hpx_walker` (round E, with the new actions: re-check, jump, ×4/×8 steps);
- `hpx_base` (the same training without the new actions);
- the budget-aware walkers `cov_vb` s0–s2 (round C).

Code: `redo.py`; full output: `redo_report.txt`. The handoff numbers come from the real honeypot logs (`handoff_phases.json`, pulled from `~/honeypotE/runs`).

## 1. `hpx_walker` is a random jumper with a local search attached

Instances come from the training family (plateau with 1% live, 5 binomial trials per evaluation, n ∈ {20, 50, 100, 169}, budget 600), as 3 disjoint blocks of 24.

**Robust across seeds.** In each block it chooses move 37–42%, re-check 10–12% and jump 46–53% of the time. The ×4/×8 steps are used on 0% of moves.

| When | Move | Re-check | Jump | Pull to best (moves) |
|---|---|---|---|---|
| Before any signal | 0.24 | 0.05 | 0.70 | 0.94 |
| After signal | 0.54 | 0.16 | 0.30 | 0.91 |
| Budget 0–25% | 0.29 | 0.09 | 0.62 | |
| Budget 75–100% | 0.54 | 0.17 | 0.29 | |

With no signal it mostly samples random points, which is what random search was beating the old walkers with. Once it has signal it moves locally and re-checks more. Re-checks rise toward the end of the budget.

**Ablations.** Same instances and random numbers. An action is removed by setting its logit to −1e9. "Found region" means the score is below −0.5.

| Variant | Median score | Found region | Better than full |
|---|---|---|---|
| Full | −0.41 | **0.44** | – |
| No re-check | 0.00 | 0.11 | 0.24 |
| No jump | 0.00 | 0.33 | 0.37 |
| No long steps | −0.33 | 0.43 | 0.47 |
| None of the three | 0.00 | 0.06 | 0.20 |
| `hpx_base` (trained without them) | −0.15 | 0.15 | 0.34 |

- **Re-check and jump carry the gain. Long steps are dead weight.**
- Removing re-check hurts the most, even though re-check never moves the walker. My hypothesis, not tested: with only 5 noisy trials, a weak first hit near the edge of the live region needs confirming before the walker commits to it.
- **Caveat:** these are new instances from the training family, not a held-out family. The real-honeypot version (no re-check / no jump, 5 seeds each) is queued as `hpA-abl` in `~/honeypotE`.

## 2. Budget-aware walkers (`cov_vb`): early convergers

n = 100, 3 policy seeds × 4 problem seeds. Score is the median log10 regret, lower is better. "logmult" is the mean log step multiplier the policy picks; below 0 means it shrinks the step.

| Problem | Budget | Walker | CMA-ES | Walker: share of improvement in quarter 1 / quarter 4 |
|---|---|---|---|---|
| Sphere | 300 | −0.34 | −0.37 | 0.85 / 0.03 |
| Sphere | 1000 | −0.43 | −0.67 | 0.77 / 0.04 |
| Sphere | 10000 | −0.78 | −1.50 | 0.74 / 0.04 |
| Rastrigin | 300 | −0.19 | −0.17 | 0.84 / 0.03 |
| Rastrigin | 10000 | −0.32 | −0.47 | 0.76 / 0.03 |
| Honeypot family | 300 | −0.39 | −0.19 | 0.05 / 0.23 |
| Honeypot family | 1000 | **−1.48** | −0.56 | 0.26 / 0.07 |
| Honeypot family | 10000 | **−1.55** | −0.89 | 0.89 / 0.00 |

- **The policy shrinks its step in every quarter** (logmult −0.02 to −0.33) and pulls toward the best point on 81–90% of steps.
- **At 300 evaluations, CMA-ES spreads its gains over the run (49% in the first quarter), while the walker makes 85% of its gains early.** That is why the walker wins small budgets.
- **At 10,000 evaluations both front-load on this log scale, but CMA-ES gets much further on smooth functions** (−1.50 vs −0.78 on sphere). My reading: CMA-ES's covariance-adapted steps converge faster than the walker's ±1-per-coordinate moves, which is how it beats the walkers at large budgets.
- **On the plateau family the walker wins at every budget.**

## 3. The handoff puzzle, from the real honeypot logs

The puzzle: the handoff (walker for the first 30%, then CMA-ES) scored 0.82 with `hpx_base` but only 0.51 with `hpx_walker`.

- **The new actions do help the walker phase.** `hpx_walker`'s first 30% has 2–22% valid levels and a few levels at exploit 1.0. `hpx_base`'s has 0–3% valid.
- **The final pick undoes it.** Dozens of levels end tied at a search score of 1.0, and the top-3 re-check breaks ties by first-seen order. So it re-measures the earliest hits, which are often lucky one-off levels. Example: `hpx_walker` seed 3's top pick had a search score of 1.0, then re-measured at 0.00 with 0% valid.
- **CMA-ES can be trapped.** The starting step size comes from the spread of the walker's top 10 points. In seed 2 the walker's points were all invalid and tightly packed, so CMA-ES started small and stayed stuck: 5% valid, final score 0.

**Fixes for the product:**
- When the search score saturates, re-check more of the tied top candidates (e.g. the top 10), or break ties by number of measurements instead of by first-seen order.
- Put a floor under the handoff's starting step size.
