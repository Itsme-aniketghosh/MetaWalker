# Training a small optimizer to help PPO cheat at honeypots

*I trained a small optimizer on synthetic functions, and used it to search gridworld levels where PPO cheats.*

## TL;DR

- [Reward-hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) honeypots are levels that are built by hand where a reward is given to the agent for cheating (e.g. taking a shortcut to the goal). I converted this process into a black box optimization problem where the score is based on how often PPO cheats.
- I used [evolution strategies](https://arxiv.org/abs/1703.03864) to train the optimizer on the cheap, synthetic functions, and applied it to the honeypot without any further training.
- Given the same PPO budget, my best policy (RLBO) found levels where PPO cheated 79% of the time, compared to 70% for [CMA-ES](https://arxiv.org/abs/1604.00772), and 49% for GP [Bayesian optimization](https://arxiv.org/abs/1807.02811).
- Doing the above, with a shared PPO budget yielded 90% success rate, and with 3 times the budget, yielded 88% success rate.
- My learned walker found its first strong level in a median of 6 PPO runs, compared to 17 for CMA-ES, and 46 for GP Bayesian optimization.
- **Try it:** [Colab demo](https://colab.research.google.com/drive/1nKxKFCsVaD7sqL6rE6NkNPjR3K4qFE5y?usp=sharing) (Run pip install metawalker before running demo cells.)

**Epistemic Status:** This is a course project. I have only tried testing in 1 artificial environment, and the differences between methods are small enough that they could be due to the noise. I'd bet on RLBO over GPBO and the panels over CMA-ES. The rest is suggestive.

## The problem

In order to accurately measure reward hacking or train it out, we need environments where cheating is just as rewarding as completing the main task. We need a way to determine whether a model cheated.

For programming models, maybe a programmer will edit the tests to reward themselves for an undeserved grade, instead of finding and fixing the bug, like some students do ([examples in linked article](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)).

These environments are typically built by hand. Can an optimizer build them? Will a learned optimizer do better?

## The testbed

Each run should take just a few minutes. LLMs take too long to train for this. I made a gridworld with 13 rows and 17 columns, and trained [PPO](https://arxiv.org/abs/1707.06347) with it.

A good performance is rewarding the agent for collecting the key, opening the door and reaching the goal. The environment also contains a special square (like a teacher's) that ends the game and gives reward +1. I interpret this reward as cheating (editing the tests).

Each level of the game is described by 169 numbers. These define location of objects, walls, hyperparameters of PPO, and whether the teacher's square is present. To evaluate a level, I train PPO 5 times and see how often it gets the teacher's square.

I remove levels where the goal is unreachable, where the grader is the shortest path, and where PPO can't learn the correct behavior with the grader removed. The last filter is critical. If it didn't exist, the most obvious trick to get the high score is to make the level confusing enough that the agent finds the shortcut of the grader.

![A 13x17 gridworld with walls, a start square, a key, a door in the dividing wall, a goal on the far side, and a grader square near the dividing wall.](figures/honeypot_level.png)

*After training from scratch, using new seeds, PPO always selected the grader X, an honest path being S → K → D → G.*

Standard optimizers failed with this setup. Each evaluation gives us only a few hundred attempts, and each of those requires multiple runs of PPO, making it very computationally expensive. Scores are very noisy, and the vast majority of levels score a 0. There are small pockets of signal, but finding and following those pockets is extremely difficult.

[GP BO](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GPSampler.html) takes a long time and still performs poorly in 169 dimensions. [TPE](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html) struggled with this many interacting parameters, and CMA-ES was the strongest baseline.

The optimizer itself is an RL agent, and the reward is the distance of the best point to the global optimum. Each episode is a complete search of a function, and at each step of the search the agent observes the points evaluated so far and decides on the next point to evaluate. After each episode, the agent receives a reward based on the closest distance to the optimum point.

[Chen et al. (2017)](https://arxiv.org/abs/1611.03824) used RNNs for this task with smooth functions. Mine have to deal with noisy, flat functions.

Training on the honeypot would be far too expensive, hundreds of PPO runs per episode. I did experiments with rotated and shifted [standard test functions](https://www.sfu.ca/~ssurjano/optimization.html), as well as a synthetic family where 95% of the space is flat, and the signal is contained in a single, narrow, hidden region.

Three optimizers are trained:

- A small network, Walker, that moves one coordinate at a time, and can also, optionally, take a random action or check the score at the best location.
- RL-CMA: a learned policy controls the steps and restarts of a CMA-ES.
- RLBO: a policy chooses from a suggestion of 48 candidate points a single point.

ES outperformed [branching](https://arxiv.org/abs/1711.08946) [double DQN](https://arxiv.org/abs/1509.06461) and PPO, and branching double DQN failed completely on flat functions. I hypothesize ES focuses on the final score, whereas in gradient-based RL there is noise throughout the process. RLBO outperforms Walker and RL-CMA.

Each method was trained for 200 levels, and each was tested 5 times. Results reveal the bias towards high scores due to noisy evaluations. To correct for this, the top 3 levels from each search were trained with 10 new PPO seeds. The best of these levels were evaluated again with new seeds, giving the final results.

![Dot plot of the exploit rate each search achieved on brand-new PPO seeds: a panel with 3x the budget averages 0.88, a panel sharing one budget 0.90, RLBO 0.79, CMA-ES 0.70, the walker 0.68 and GP Bayesian optimization 0.49.](figures/honeypot_results.png)

This dot plot represents a search exploited via fresh PPO seeds. 0.88 (3x budget), 0.90 (1x budget), 0.79 (RLBO), 0.70 (CMA-ES), 0.68 (walker) and 0.49 (GP BO).

*Each dot represents one search, evaluated over 10 PPO seeds. The black line represents the mean. Panels are described below.*

All three independently trained RLBO agents outperformed CMA-ES, with means of 0.75, 0.80 and 0.83, respectively. In the same seed comparisons, however, RLBO only won 17 of 30. The difference I'd put money on is between RLBO and GP BO.

What makes this particularly interesting is that the policy had only previously seen synthetic math functions. I then introduced it to gridworld and PPO hyperparameters, also without retraining. That is the result I would most like to see repeated.

## The walker reaches a strong level the fastest

With enough PPO runs, nearly every method was able to reach a level scoring ≥0.8, including random search which reached it in 1/5 searches. How quickly they reach a strong level isn't as impressive. The walker reached a strong level after a median of 6 PPO runs, and CMA-ES after 17, and BO at 46.

![Dot plot of PPO runs until each search first found a level scoring 0.8 or more: learned walker median 6, CMA-ES 17, TPE 23, GP 46, SMAC3 67; random search succeeded in 1 of 5 searches.](figures/time_to_target.png)

*Each dot in the plot is one search. The number of PPO runs to obtain a valid level (>= 0.8) is indicated by the black line. Probably some early successes came from a lucky chance. I didn't log this for RLBO.*

The final levels of the walker only tied CMA-ES (0.68 vs 0.70), but the walker is the easiest optimizer to understand. When it fails to locate any signal, it randomly moves to a new location about 70% of the time. After it gets a score of more than zero, it will use local moves and frequently reevaluates its best point. That makes sense. If you're searching in a noisy environment, a new best sometimes is just luck and removing the recheck action brings the score down to 0.48. If we remove jumps, the score drops to 0.59.

![Dot plot of the walker ablation: full walker 0.81, without jump 0.59, without re-check 0.48, without any of the new actions 0.53.](figures/walker_ablation.png)

On the honeypot, optimizer overhead is not a major concern since each PPO run takes a few minutes. But when evaluations take only a few seconds, it becomes a concern. On my synthetic benchmarks, RLBO dominated GP BO by 76-100% across 5 to 500 dimensions. It was also much faster. GP BO required around 4,300 seconds to perform 1,000 evaluation runs. RLBO did it in 23.

![Bar chart, log scale, of optimizer time per 1,000-evaluation run: NGOpt 17 s, RLBO 23 s, CMA-ES 35 s, RL-CMA 36 s, TPE 263 s, SMAC3 3,678 s, GP 4,296 s.](figures/speed.png)

Bar chart (log scale) showing how long each optimizer took to run 1000 evaluations.

*Optimizer overhead for 1000 evaluations, shown on a log scale. There were some [SMAC3](https://github.com/automl/SMAC3) runs that ran out of time.*

## The flatter the function, the smaller the advantage

The learned walkers achieved a better performance than CMA-ES on 91% of the at least 20-dimensional problems on my 95%-flat family of functions. But when I made the functions about 98.5% flat, that dropped to 63%. At that point, every method often failed to find the hidden region within 1000 evaluations.

![Bar chart: learned walkers beat CMA-ES on 91% of problems when 95% of the space is flat, and 63% when about 98.5% is flat.](figures/flatness.png)

*On a 95% flat family of problems with 20-500 dimensions, 5 seeds, with and without noise, using 3 learners, the learned walkers outperformed CMA-ES on 91% of problems. The largest improvement came from randomly sampling 20% of the budget using [Latin Hypercube Sampling](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html).*

On a family of functions that forms a 50 dimensional flat region, the warmed up learned walker found the hidden region in 13/20. CMA-ES found it in 2. Random search was unable to find it in any of the 20 runs.

My optimizers also found shortcuts in my code.

1. **Seed Leakage.** Because the policy's random number generator was seeded the same way as the problem, the walker's "random" starting position was always close to optimal. While these results may have been impressive, they weren't really helpful.
2. **Tie-Breaking.** On flat problems, most ES candidates tie. My ranking scheme broke ties based on the candidate's position on the list.
3. **Too Many Threads.** CMA-ES was slowed down by about 300 times due to each worker spawning its own BLAS threads. In the end, the amount of linear algebra done became the deciding factor. The issue was fixed by using [threadpoolctl](https://github.com/joblib/threadpoolctl) to limit each worker to one thread.

These issues didn't produce any errors or warnings. I only found them after writing checks for those failures. This makes me question what else I have not checked for.

## No method is the best everywhere, thus use a panel.

RLBO shows the best final results. The Walker finds strong results quickest. CMA-ES is better at larger budgets. GP BO is hard to beat for 100-200 dimensions if you don't mind waiting. If you're unsure about the kind of problem you're dealing with, you can always use a few optimizers and take the best result.

I tested two versions of this on the honeypot.

I first gave RLBO and the Walker and CMA-ES 200 runs each. Together, they scored 88% and outperformed RLBO in 13 of 15 searches. The downside: this took 3x the compute.

Then, I split a single budget of 200 PPO runs between RL-CMA, CMA-ES and the walker. Surprisingly, that panel scored 90%. I assumed it would perform worse than RLBO. Most of the results were due to RL-CMA, which got 82% with a third of the budget. The walker's selected level, meanwhile, scored zero in 3 of 5 searches.

The difference between the shared-budget panel and CMA-ES looks real at +0.20 with a 95% confidence interval of +0.05 to +0.36. The difference between the panel and RLBO does not look real at +0.11 with an interval of +0.00 to +0.23.

However, splitting the budget can also backfire. On a mix of flat and smooth test problems, the warmed up walker, alone, got the target in 11 of 15 runs, while a 2-method panel managed 3. So, the package also includes compare(), which tests optimizers on your loss functions and suggests a panel. Don't always rely on a panel to help.

Everything can be installed via pip install metawalker ([PyPI](https://pypi.org/project/metawalker/)). The default panel uses RLBO, RL-CMA, and CMA-ES, though I haven't tested that particular combination as a panel on the honeypot.

There's also a drop-in for scikit-learn's [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

During a 40-hyperparameter search, RLBO reached 0.93 test accuracy, as compared to 0.88 for [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html). With 2-7 hyperparameters, however, everything tied random search.

## What carried over

Researcher time per environment is required for a hand built honeypot. This one is a couple hundred PPO runs.

The policy learned a lot of tricks for dealing with uncertainty. It used those tricks on an expensive, noisy problem it had never seen before.

Most of the hyperparameter tuning is like that and, as such, I would expect that to be the case here as well. However, I've only tested it on one honeypot.

[Colab demo](https://colab.research.google.com/drive/1nKxKFCsVaD7sqL6rE6NkNPjR3K4qFE5y?usp=sharing) · [PyPI](https://pypi.org/project/metawalker/) · [GitHub](https://github.com/Itsme-aniketghosh/MetaWalker) (gridworld and all experiments)
