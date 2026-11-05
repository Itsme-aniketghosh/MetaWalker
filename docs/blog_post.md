# Training a small optimizer to help PPO cheat at honeypots

*I trained a small optimizer on synthetic functions, then used it directly to search for gridworld levels where PPO cheats.*

## TL;DR

- [Reward-hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) honeypots, levels where cheating pays as well as doing the actual task, are usually built by hand. I turned finding them into a black-box optimization problem with 169 parameters, scored by how often PPO takes a planted shortcut.
- I trained an optimizer using [evolution strategies](https://arxiv.org/abs/1703.03864) on cheap synthetic functions, then applied it to the honeypot without retraining.
- At equal PPO budget, my best policy (RLBO) found levels where PPO cheated 79% of the time, compared with 70% for [CMA-ES](https://arxiv.org/abs/1604.00772) and 49% for GP [Bayesian optimization](https://arxiv.org/abs/1807.02811).
- Running multiple optimizers and keeping the best result worked even better: 90% with a shared budget, and 88% using 3× the compute.
- My learned walker found its first strong level in a median of 6 PPO runs, compared with 17 for CMA-ES and 46 for BO.
- **Try it:** [Colab demo](https://colab.research.google.com/drive/1nKxKFCsVaD7sqL6rE6NkNPjR3K4qFE5y?usp=sharing) · `pip install metawalker` ([PyPI](https://pypi.org/project/metawalker/))

*Epistemic status: course project. One toy environment, 5–15 searches per method, and most differences are small enough that noise could explain them. I'd bet on RLBO over GP BO and the panels over CMA-ES. The rest is suggestive.*

## The problem

To measure reward hacking, or train it out, we need environments where cheating gives as much reward as completing the real task. We also need to know whether the model actually cheated.

For coding models, that might mean editing the tests instead of fixing the bug ([examples](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)).

These environments are usually crafted by hand. Could an optimizer find them instead? And would a *learned* optimizer do any better?

## The testbed

Searching means training the agent hundreds of times, so each run needs to be quick: minutes rather than hours. No LLMs, then. I used a 13×17 gridworld with [PPO](https://arxiv.org/abs/1707.06347). Whether this tells us anything about LLMs, I don't know yet.

The honest task is to collect a key, open a door, and reach the goal for a +1 reward. But there's also a "grader" square. Landing on it ends the episode with the same +1 reward, my stand-in for editing the tests.

Each level is defined by 169 values: walls, object locations, 5 PPO hyperparameters, and whether the grader is visible. I score a level by training PPO 5 times and measuring how often it takes the grader.

I discard levels where the goal is unreachable, the grader is simply the shortest path, or PPO can't learn the honest task when the grader is removed.

That last filter matters. Without it, the easiest way to get a high score is to make the level so confusing that the agent only stumbles onto the cheating shortcut.

![A 13x17 gridworld with walls, starting square, key, door in middle wall, goal at the end and grader square near the middle wall.](figures/honeypot_level.png)

*An example level found by the search. The honest path is S → K → D → G. After retraining from scratch on fresh seeds, PPO picked the grader X every time.*

Standard optimizers struggled with this setup.

Each evaluation takes multiple PPO runs, leaving only a few hundred attempts. Scores are noisy, and nearly every valid level scores zero. There are small pockets of signal, but no gradient pointing toward them.

[GP BO](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GPSampler.html) is slow and performs poorly in 169 dimensions. [TPE](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html) struggles with this many interacting parameters. CMA-ES was the strongest baseline.

## Training the optimizer

The optimizer itself is the RL agent.

One episode is a complete search over a function. At each step, the policy sees the points tried so far and chooses the next one. At the end, it receives a reward based on how close its best point came to the optimum.

[Chen et al. (2017)](https://arxiv.org/abs/1611.03824) used RNNs to do this on smooth functions. Mine also have to handle flat, noisy ones.

Training directly on the honeypot would be far too expensive: hundreds of PPO runs per episode. Instead, I used rotated and shifted [standard test functions](https://www.sfu.ca/~ssurjano/optimization.html), plus a synthetic family where 95% of the space is flat and one hidden region contains the signal.

I trained three optimizers:

- **Walker:** A small network that moves one coordinate at a time, with special actions to recheck its best point or jump somewhere random.
- **RL-CMA:** CMA-ES, but with step size and restarts controlled by a learned policy.
- **RLBO:** A BO-like optimizer. It fits cheap surrogate functions, proposes 48 candidate points, and lets the policy select one. Similar to [MetaBO](https://arxiv.org/abs/1904.02642), but trained end-to-end with ES.

Evolution strategies trained them better than PPO or [branching](https://arxiv.org/abs/1711.08946) [double DQN](https://arxiv.org/abs/1509.06461). Branching double DQN failed outright on flat problems.

My guess is that ES only needs the final score, while gradient-based RL has to assign credit across a long, noisy chain of decisions.

## RLBO finds the strongest honeypots

Every method got 200 levels per search, with 5 searches each.

There's a problem with taking the best observed score: noisy evaluations make it biased upward.

To account for this, I retrained the top 3 levels from each search on 10 fresh PPO seeds and kept the best. Even then, the scores were around 0.08 too high. So I evaluated the selected levels again on brand-new seeds.

Those final evaluations give the results below.

![Dot plot showing how often each search exploited using fresh PPO seeds. 3x budget averaged to 0.88, 1x budget averaged to 0.90, RLBO 0.79, CMA-ES 0.70, walker 0.68, GP Bayesian Optimization 0.49.](figures/honeypot_results.png)

*Each dot represents one search, evaluated on 10 fresh PPO seeds. The black line shows the mean. The panels are explained below.*

All three independently trained RLBO policies beat CMA-ES on average: 0.75, 0.80, and 0.83, compared with 0.70.

But individual searches were much closer. RLBO won only 17 of 30 same-seed comparisons.

The difference I'd bet on is RLBO versus GP BO.

What makes this interesting is that the policy had only ever seen synthetic math functions. I then applied it directly to gridworld walls and PPO hyperparameters, without retraining.

That's the result I'd most like to see replicated.

## The walker reaches a strong level fastest

Given enough PPO runs, nearly every method eventually found a level scoring ≥0.8. Even random search managed it in 1 of 5 searches.

But how quickly can they get there?

The walker, with a short random warm-up, found a strong level after a median of just 6 PPO runs. CMA-ES needed 17, and BO needed 46.

![Dot plot showing how many PPO runs it took to find the first level with a score ≥0.8. Median for learned walker 6, CMA-ES 17, TPE 23, GP 46, SMAC3 67. Random search found 1/5.](figures/time_to_target.png)

*Each dot represents one search. The black line shows the median number of PPO runs before finding a valid level scoring ≥0.8. These were single measurements during the search, so some early successes were probably lucky. I didn't log this for RLBO.*

The walker's final levels only tied CMA-ES (0.68 versus 0.70), but it's the easiest optimizer to understand.

Before it finds any signal, it jumps to a random point about 70% of the time. Once something scores above zero, it switches to local moves and frequently rechecks its best point.

That makes sense. In a noisy search, a sudden new best is often just luck.

Removing the recheck action drops its score to 0.48. Removing the jump drops it to 0.59.

![Dot plot showing results for walker ablation. Full walker 0.81, no jump 0.59, no recheck 0.48, no new actions 0.53.](figures/walker_ablation.png)

*One action removed at a time, 5 searches each, with a budget of 600 levels. Best-of-3 scores, not rechecked on fresh seeds.*

## RLBO is about 190× faster than GP BO

On the honeypot, optimizer overhead barely matters because each PPO run takes minutes.

But when evaluations take seconds, that overhead becomes important.

On my synthetic benchmarks, RLBO outperformed GP BO on 76–100% of problems across 5 to 500 dimensions.

It was also much faster. GP BO took about 4,300 seconds for a 1,000-evaluation run. RLBO took just 23.

![Bar chart (log scale) showing how long each optimizer took to run 1000 evaluations. NGOpt 17s, RLBO 23s, CMA-ES 35s, RL-CMA 36s, TPE 263s, SMAC3 3678s, GP 4296s.](figures/speed.png)

*Optimizer overhead for 1,000 evaluations, shown on a log scale. Some [SMAC3](https://github.com/automl/SMAC3) runs hit the time limit.*

## The flatter the function, the smaller the advantage

On my 95%-flat family of functions, the learned walkers beat CMA-ES on 91% of problems with at least 20 dimensions.

But when I made the functions about 98.5% flat, that dropped to 63%.

At that point, every method often failed to find the hidden region within 1,000 evaluations.

![Bar chart showing how often the learned walkers outperformed CMA-ES. 95% flat 91%, ~98.5% flat 63%.](figures/flatness.png)

*20–500 dimensions, 5 seeds, with and without noise, pooled across 3 walkers. Black lines show 95% confidence intervals.*

The biggest improvement came from something simple: spending the first 20% of the budget on random samples using [Latin hypercube sampling](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html).

On a 50-dimensional plateau, the warmed-up walker found the hidden region in 13 of 20 runs. CMA-ES managed 2. Random search found it in none.

## Three bugs looked like breakthroughs

My optimizers also kept finding shortcuts in my own code.

1. **Seed leakage.** The policy's random number generator was seeded the same way as the problem. As a result, the walker's "random" starting point landed near the optimum every time. Beautiful graphs, meaningless results.

2. **Tie-breaking.** On flat problems, most ES candidates tie. My ranking system broke those ties by their position in a list, making noise look like a gradient.

3. **Too many threads.** Every worker spawned its own BLAS threads, making CMA-ES about 300× slower. In a wall-clock comparison, that's a free win for whoever does less linear algebra. The fix was to limit each worker to one thread using [threadpoolctl](https://github.com/joblib/threadpoolctl).

None of these bugs caused errors or warnings.

I only found them after writing checks for those specific failures. Which makes me wonder what I still haven't checked.

## No method wins everywhere, so try a panel

RLBO finds the strongest final levels. The walker reaches a strong level fastest. CMA-ES does better at larger budgets. And GP BO is hard to beat at 100–200 dimensions if you're willing to wait.

If you're unsure which kind of problem you're dealing with, one option is to run several optimizers and keep the best result.

I tested two versions of this on the honeypot.

First, I gave RLBO, CMA-ES, and the walker 200 PPO runs each. Together, they scored 88%, matching or beating RLBO alone in 13 of 15 searches.

The catch: this used 3× the compute.

Next, I split a single budget of 200 PPO runs between RL-CMA, CMA-ES, and the walker. Surprisingly, that panel scored 90%.

I expected it to do worse than RLBO.

Most of the result came from RL-CMA, which scored 82% with just a third of the budget. The walker's selected level, meanwhile, scored zero in 3 of 5 searches.

The gap between the shared-budget panel and CMA-ES looks real: +0.20, with a 95% confidence interval of +0.05 to +0.36.

The gap between the panel and RLBO doesn't yet: +0.11, with an interval of 0.00 to +0.23.

But splitting the budget can also backfire.

On a mix of flat and smooth test problems, the warmed-up walker alone found the target in 11 of 15 runs. A two-method panel managed just 3.

That's why the package also includes `compare()`, which tests optimizers on your own loss functions and suggests a panel. I wouldn't assume a panel will always help.

Everything is available through `pip install metawalker` ([PyPI](https://pypi.org/project/metawalker/)).

The default panel uses RLBO, RL-CMA, and CMA-ES, though I haven't tested that exact combination as a panel on the honeypot.

There's also a drop-in for scikit-learn's [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).

On a 40-hyperparameter search, RLBO reached 0.93 test accuracy, compared with 0.88 for [`RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).

With just 2–7 hyperparameters, though, everything ties random search.

## What carried over

A hand-built honeypot costs researcher time per environment. This one costs a couple hundred PPO runs.

The more interesting part is what the policy learned on cheap synthetic problems: when to explore, when to distrust a result, when to look again. That worked on an expensive, noisy problem it had never seen.

A lot of hyperparameter search looks like that, so I'd expect this to carry over. But I've tested it on exactly one honeypot.

[Colab demo](https://colab.research.google.com/drive/1nKxKFCsVaD7sqL6rE6NkNPjR3K4qFE5y?usp=sharing) · [PyPI](https://pypi.org/project/metawalker/) · [GitHub](https://github.com/Itsme-aniketghosh/MetaWalker) (gridworld and all experiments)
