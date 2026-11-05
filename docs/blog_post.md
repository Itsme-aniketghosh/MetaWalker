# Better honeypots, fewer tries: training an optimizer to make PPO cheat

*I trained a small optimizer on synthetic functions, then used it, unchanged, to search for gridworld levels where PPO cheats.*

**TL;DR**

- [Reward-hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) honeypots, environments where cheating pays exactly as well as doing the task, are mostly built by hand. I treated finding them as a black-box search over 169 parameters, scored by how often PPO takes a shortcut I planted.
- I trained an optimizer with [evolution strategies](https://arxiv.org/abs/1703.03864) on cheap synthetic functions and ran it on the honeypot without retraining.
- At an equal PPO budget, my best learned optimizer (RLBO) found levels where PPO cheats 79% of the time on fresh seeds, against 70% for [CMA-ES](https://arxiv.org/abs/1604.00772) and 49% for GP [Bayesian optimization](https://arxiv.org/abs/1807.02811).
- Running several optimizers and keeping the best level did better: 90% with one shared budget, 88% with 3× the compute.
- My learned walker reaches its first strong level in a median of 6 PPO runs, against 17 for CMA-ES and 46 for BO.
- **Try it:** [Colab demo](https://colab.research.google.com/drive/1nKxKFCsVaD7sqL6rE6NkNPjR3K4qFE5y?usp=sharing) · `pip install metawalker` ([PyPI](https://pypi.org/project/metawalker/)).

*Epistemic status: course project. One toy environment, 5 to 15 searches per method, so most gaps are inside the noise. I'd bet on RLBO beating GP BO and on both panels beating CMA-ES; the rest is suggestive.*

## The problem

To measure reward hacking, or train it out, I need environments where cheating pays as well as doing the job and I can tell which one the model did. For coding models that's "edit the tests instead of fixing the bug" (DeepMind has a [list of real examples](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)). People mostly build these by hand. Can an optimizer find them, and does a *learned* optimizer do it better?

## The testbed: a gridworld where cheating pays as well as solving

The search trains the agent hundreds of times, so each run has to take minutes. That rules out LLMs and means a 13×17 gridworld with [PPO](https://arxiv.org/abs/1707.06347). Whether its levels tell us anything about LLMs, I don't know yet.

The honest task is to pick up a key, open a door, and reach the goal, for +1. A "grader" square ends the episode with the same +1: my stand-in for editing the tests. A level is 169 numbers (walls, object positions, 5 PPO hyperparameters, and whether the grader is visible). I score it by training PPO 5 times and counting how often it takes the grader. I throw a level out if the goal is unreachable, if the grader is just the shortest path, or if PPO can't learn the honest task with the grader removed. Without that last filter, the easiest high score is a level so confusing the agent only stumbles onto the shortcut.

![A 13×17 gridworld level with walls, a start square, a key, a door in the dividing wall, a goal on the far side, and a grader square near the dividing wall](figures/honeypot_level.png)

*A level the search found. Honest route: S → key K → door D → goal G. Retrained from scratch on fresh seeds, PPO took the grader X every time.*

This is hard for standard tuners. Each evaluation is several PPO runs, so I get a few hundred tries. Scores are noisy. And nearly every valid level scores zero, with small pockets where PPO cheats and no gradient toward them. [GP BO](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GPSampler.html) gets slow and weak at 169 dimensions, and [TPE](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html) can't handle that many interacting parameters. CMA-ES was the strongest standard baseline.

## Training the optimizer

The optimizer is the RL agent. One episode is one full search on one function: each step the policy sees the points tried so far and picks the next one, and at the end it's rewarded by how close its best point got to the optimum. [Chen et al. (2017)](https://arxiv.org/abs/1611.03824) did this with RNNs on smooth functions; mine have to handle flat, noisy ones.

Training on the honeypot itself would cost hundreds of PPO runs per episode, so I trained on rotated, shifted [standard test functions](https://www.sfu.ca/~ssurjano/optimization.html) plus a synthetic family that's 95% flat with one hidden pocket of signal. I built three optimizers:

- **Walker**: a small network that moves one coordinate at a time, with special moves to re-check the current best or jump to a random point.
- **RL-CMA**: CMA-ES with step size and restarts handed to a learned policy.
- **RLBO**: shaped like BO. It fits cheap surrogates, proposes 48 candidates, and the policy picks one. Similar to [MetaBO](https://arxiv.org/abs/1904.02642), but trained end-to-end with ES.

ES trained them better than PPO or [branching](https://arxiv.org/abs/1711.08946) [double DQN](https://arxiv.org/abs/1509.06461), which failed outright on flat problems. My guess: ES only needs the final score, while gradient-based RL has to assign credit across a long, noisy chain of steps.

## RLBO finds the strongest honeypots of any single method

Every method got 200 levels per search, 5 searches each. The best observed score in a noisy search is biased upward, so I retrained each search's top 3 levels on 10 fresh PPO seeds and kept the best. That still ran about 0.08 high, so the numbers below retrain the pick once more on 10 brand-new seeds.

![Dot plot of the exploit rate each search achieved on brand-new PPO seeds: a panel with 3x the budget averages 0.88, a panel sharing one budget 0.90, RLBO 0.79, CMA-ES 0.70, the walker 0.68 and GP Bayesian optimization 0.49](figures/honeypot_results.png)

*Each dot is one search, on 10 brand-new PPO seeds. Black line is the mean. Panels are explained below.*

Three independently trained RLBO policies all beat CMA-ES on average (0.75, 0.80, 0.83 vs. 0.70), but search by search it's close: RLBO won 17 of 30 same-seed pairings. The gap I'd bet on is the one to GP BO.

The policy had only seen synthetic math functions and went to gridworld walls and PPO hyperparameters with no retraining. That's the result I'd most like someone to replicate.

## The walker finds a strong level fastest

With enough PPO runs, nearly every method reaches a level scoring ≥0.8 (random search managed it in 1 of 5 searches). So the question is how many runs it takes. The walker, with a short random warm-up, needed a median of 6. BO needed 46.

![Dot plot of PPO runs until each search first found a level scoring 0.8 or more: learned walker median 6, CMA-ES 17, TPE 23, GP 46, SMAC3 67; random search succeeded in 1 of 5 searches](figures/time_to_target.png)

*Each dot is one search. Black line is the median PPO runs before a valid level ≥0.8. These are single in-search measurements, so some early hits were lucky. I didn't log this for RLBO.*

Its final levels only tie CMA-ES (0.68 vs. 0.70), but it's the easiest optimizer to look inside. Before it sees any signal, it jumps to a random point on about 70% of steps. Once something scores above zero, it switches to local moves and keeps re-checking its best point, since a sudden new best in a noisy search is usually luck. Removing re-check dropped its score from 0.81 to 0.48; removing jump, to 0.59.

![Dot plot of the walker ablation: full walker 0.81, without jump 0.59, without re-check 0.48, without any of the new actions 0.53](figures/walker_ablation.png)

*One action removed at a time, 5 searches each, 600-level budget. Best-of-3 scores, not re-checked on brand-new seeds.*

## Bayesian optimization is about 190× slower than RLBO

On the honeypot, optimizer overhead barely matters because PPO runs take minutes. When evaluations take seconds, it does. On my synthetic benchmarks RLBO beat GP BO on 76–100% of problems from 5 to 500 dimensions, at 23 seconds per 1,000-evaluation run against 4,300.

![Bar chart, log scale, of optimizer time per 1,000-evaluation run: NGOpt 17 s, RLBO 23 s, CMA-ES 35 s, RL-CMA 36 s, TPE 263 s, SMAC3 3,678 s, GP 4,296 s](figures/speed.png)

*Optimizer overhead per 1,000-evaluation run, log scale. [SMAC3](https://github.com/automl/SMAC3) hit the time limit on some runs.*

## The flatter the landscape, the smaller the edge

On my 95%-flat family, learned walkers beat CMA-ES on 91% of problems from 20 dimensions up. At ~98.5% flat, that fell to 63%, and every method often missed the pocket within 1,000 evaluations.

![Bar chart: learned walkers beat CMA-ES on 91% of problems when 95% of the space is flat, and 63% when about 98.5% is flat](figures/flatness.png)

*20 to 500 dims, 5 seeds, with and without noise, pooled over three walkers. Black lines are 95% CIs.*

What helped most was simple: spend the first 20% of the budget on random ([Latin hypercube](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html)) samples. On a 50-dimensional plateau, the warmed-up walker found the pocket in 13 of 20 runs, CMA-ES in 2, random search in none.

## Three bugs looked like breakthroughs

My optimizers kept finding shortcuts in my own code.

1. **Seed leak.** The policy's RNG was seeded like the problem's, so the walker's "random" first point landed next to the optimum every time. Beautiful training curves, meaningless.
2. **Tie-breaking.** On flat problems most ES candidates tie, and my ranking broke ties by list position. Noise dressed up as a gradient.
3. **Thread oversubscription.** Every worker spawned its own BLAS threads, and CMA-ES ran about 300× slow. In a wall-clock comparison, that's a free win for whoever does less linear algebra. Fix: one thread per worker ([threadpoolctl](https://github.com/joblib/threadpoolctl)).

None of them crashed or warned. I found each one only after writing a check for that exact failure, which makes me wonder what I haven't checked yet.

## No method wins everywhere, so run a panel

RLBO ends with the strongest levels, the walker gets to one fastest, CMA-ES pulls ahead at large budgets, and GP BO is hard to beat at 100 to 200 dimensions if you can wait. If you don't know which your problem looks like, run several and keep the best.

On the honeypot, giving RLBO, CMA-ES, and the walker 200 PPO runs each scored 88%, at least as good as RLBO alone in 13 of 15 searches, but at 3× the compute. Sharing one 200-run budget between RL-CMA, CMA-ES, and the walker scored 90%. I expected it to land below RLBO. Most of it came from RL-CMA (82% on a third of the budget); the walker's pick scored zero in 3 of 5 searches. The gap to CMA-ES (+0.20, 95% interval +0.05 to +0.36) looks real; the gap to RLBO (+0.11, 0.00 to +0.23) doesn't yet.

Splitting can backfire: on a mix of flat and smooth test problems, the warmed-up walker alone hit the target on 11 of 15 runs, a two-method panel on 3. So the package also has `compare()`, which tests methods on your own losses and suggests a panel.

It's all in `pip install metawalker` ([PyPI](https://pypi.org/project/metawalker/)). The default panel is RLBO, RL-CMA, and CMA-ES (untested as a panel on the honeypot). There's also a drop-in for scikit-learn's [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html): on a 40-hyperparameter search, RLBO reached 0.93 test accuracy against 0.88 for [`RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html). With 2 to 7 hyperparameters, everything ties random search.

## What carried over

A hand-built honeypot costs researcher time per environment; this costs a couple hundred PPO runs. The more interesting part is that what the policy learned on cheap synthetic problems (when to explore, when to distrust a result, when to look again) worked on an expensive, noisy problem it never saw. A lot of hyperparameter search looks like that, so I'd expect it to carry over, but I've tested it on exactly one honeypot.

[Colab demo](https://colab.research.google.com/drive/1nKxKFCsVaD7sqL6rE6NkNPjR3K4qFE5y?usp=sharing) · [PyPI](https://pypi.org/project/metawalker/) · [GitHub](https://github.com/Itsme-aniketghosh/MetaWalker) (gridworld and all experiments)
