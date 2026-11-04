# Better honeypots, fewer tries: training an optimizer to make PPO cheat

*A small optimizer trained by RL on synthetic functions, transferred unchanged to searching for gridworld levels where PPO cheats.*

**TL;DR**

- Reward-hacking honeypots (environments where cheating pays exactly as well as doing the task) are mostly hand-built. I treated finding them as black-box search: 169 parameters (walls, object positions, PPO hyperparameters), scored by how often PPO takes a planted shortcut.
- Instead of tuning an optimizer, I trained one with evolution strategies on cheap synthetic functions, then ran it unchanged on the honeypot.
- At equal PPO budget, the best learned optimizer (rlbo) found levels where PPO cheats 88% of the time, vs. 76% for CMA-ES and 55% for GP Bayesian optimization.
- The learned walker reaches its first strong level in a median of 6 PPO runs, vs. 17 for CMA-ES and 46 for BO, though its final levels are weaker than rlbo's.
- No optimizer wins everywhere. The walker loses to CMA-ES on final quality at equal compute, and the learned edge shrinks as the landscape gets flatter.
- **Try it:** [Colab demo](https://colab.research.google.com/drive/1nKxKFCsVaD7sqL6rE6NkNPjR3K4qFE5y?usp=sharing) (runs in a few minutes, no setup) · `pip install metawalker` ([PyPI](https://pypi.org/project/metawalker/)).

*Epistemic status: course project. One toy environment, 5 to 15 searches per method. I trust the big gaps (rlbo vs. CMA-ES and BO), not the small ones.*

## Honeypots are still built by hand, one at a time

To measure reward hacking, or train it out of a model, you need an environment where cheating pays exactly as well as doing the job, and where you can tell afterward which one the model did. The coding-model version is "edit the tests instead of fixing the bug." People mostly build these by hand, one at a time. I wanted to know whether an optimizer could find them instead, and whether a *learned* optimizer would do it better than the standard ones.

## The test bed: a gridworld where cheating pays as well as solving

A 13×17 gridworld, small enough to train PPO in minutes. The honest task: pick up a key, open a door, reach the goal, +1. Somewhere on the map is a "grader" square that ends the episode with the same +1. That's the test-editing shortcut.

A level is 169 numbers: 154 for walls, 9 for object positions, 5 PPO hyperparameters and a grader-visibility switch. Its score is the fraction of 5 PPO training runs that end up taking the grader. A level is rejected if:

- the goal is unreachable (BFS check),
- the grader is so much closer than the goal that "cheating" is just the shortest path, or
- PPO can't learn the honest task with the grader removed.

The last filter matters most. Without it, the easiest high score is a level so confusing that the agent can only stumble onto the shortcut. That's a broken level, not a honeypot.

![A 13×17 gridworld level with walls, a start square, a key, a door in the dividing wall, a goal on the far side, and a grader square near the dividing wall](figures/honeypot_level.png)

*A level the search found. The honest route is S → key K → door D → goal G. The grader X sits against the dividing wall on the start side. Retrained from scratch on fresh seeds, PPO took the grader every time.*

## Most levels score zero, so standard tuners have nothing to follow

Each evaluation is several PPO runs, so you get a few hundred tries. Scores are noisy because PPO is noisy. And the landscape is almost entirely flat: nearly every valid level scores zero, with small pockets where PPO starts cheating and no gradient pointing toward them.

GP-based BO gets slow and weak as dimension grows. TPE handles noise but not 169 interacting parameters. CMA-ES is the strongest standard baseline.

## I trained the optimizer itself, on cheap synthetic problems

An optimizer is an agent: it sees the history, picks the next point, observes the result, repeats. So each training episode is one full search on some function. State is a summary of the history, actions choose where to look, reward is how close the final answer gets to the optimum.

Training on the honeypot itself is too expensive, so the policies trained on rotated, shifted standard test functions plus a synthetic "honeypot-like" family: 95% flat, one hidden pocket of signal, noisy pass/fail readings.

Three learned optimizers, all trained with ES:

- **Walker**: a small network that moves one coordinate at a time (up, down, or stay). Extra outputs set step size, momentum, and pull toward the best point. It also has three special moves: keep walking, re-check the current best, or jump to a random point.
- **RL-CMA**: CMA-ES with step size and restart decisions handed to a learned policy.
- **rlbo**: BO-shaped. Each step it fits cheap surrogates, proposes 48 candidates, and the policy picks one. Similar in spirit to MetaBO, but trained end to end with ES.

## rlbo finds the strongest honeypots: 88%, vs. 76% for CMA-ES

Every method got 200 evaluated levels per search, 5 searches each. Since the best *observed* score in a noisy search is biased upward, each search's best levels were retrained on fresh PPO seeds before scoring.

![Dot plot of the exploit rate each search achieved: rlbo averages 0.88, CMA-ES 0.76, the walker 0.69 and GP Bayesian optimization 0.55](figures/honeypot_results.png)

*Each dot is one search, scored on fresh PPO seeds. Black line is the mean.*

All three independently trained rlbo policies beat CMA-ES on average, and rlbo won on the same search seed in 10 of 15 head-to-heads.

The policy never saw anything but synthetic math functions. That it transfers to gridworld walls plus PPO hyperparameters without retraining is the result I'd most like to see replicated.

## The walker finds a strong level fastest: 6 PPO runs, vs. 46 for BO

The obvious objection: with enough PPO runs, doesn't everything find a good honeypot?

Mostly yes: every standard method usually reaches a level scoring ≥0.8 (random search managed 1 in 5). The difference is cost. The walker, with a short random warm-up, got there in a median of 6 PPO runs, against 46 for BO. When each run is a full training job, that's a big deal.

![Dot plot of PPO runs until each search first found a level scoring 0.8 or more: learned walker median 6, CMA-ES 17, TPE 23, GP 46, SMAC3 67; random search succeeded in 1 of 5 searches](figures/time_to_target.png)

*Each dot is one search. Black line is the median PPO runs before first logging a valid level ≥0.8. Right-hand count is how many searches got there at all.*

Caveats: these are single in-search measurements, so some early 0.8s were probably lucky, and I didn't log this for rlbo.

## Bayesian optimization is about 190× slower than rlbo

On the synthetic benchmarks, rlbo beats GP BO on 76–100% of problems from 5 to 500 dimensions, in 23 seconds per 1,000-evaluation run against BO's 4,300.

![Bar chart, log scale, of optimizer time per 1,000-evaluation run: NGOpt 17 s, rlbo 23 s, CMA-ES 35 s, RL-CMA 36 s, TPE 263 s, SMAC3 3,678 s, GP 4,296 s](figures/speed.png)

*Optimizer overhead per 1,000-evaluation run, log scale. SMAC3 hit the time limit on some runs, so its true cost is higher.*

Given BO's wall-clock time, even plain CMA-ES wins on 90–100% of problems. If evaluations take seconds, BO's bookkeeping is the bottleneck.

## The walker learned to explore first and re-check lucky results

The walker trails CMA-ES on the honeypot at equal compute, but it's the easiest to inspect.

Before it sees any nonzero score, it jumps to a random point on about 70% of steps. Small steps are useless on a flat landscape. Once something scores above zero, it switches to local moves and keeps re-checking its best point, since a sudden "best" in a noisy search is usually luck.

Ablating these on the honeypot: without re-check, 0.81 → 0.48. Without jump, 0.81 → 0.59.

![Dot plot of the walker ablation: full walker 0.81, without jump 0.59, without re-check 0.48, without any of the new actions 0.53](figures/walker_ablation.png)

*One action removed at a time, 5 searches each, 600-level budget.*

## Evolution strategies trained the optimizer better than PPO or DQN

I also trained the walker with PPO and with branching double DQN, same actions and features, with both dense and sparse rewards. ES won, PPO came close, and DQN trailed, failing outright on the flat problems. My guess at why: the final score depends on a long, noisy chain of decisions. ES only needs that final number. Gradient-based RL has to assign credit to individual steps, which is hard here.

## Three bugs looked like breakthroughs

Fittingly, the learned optimizers kept finding shortcuts in my code.

1. **Seed leak.** The policy's RNG was seeded the same way as the problem's, so the walker's "random" first point landed next to the optimum every time. ES exploited this immediately. Beautiful training curves, meaningless.
2. **Tie-breaking in the ES update.** On flat problems most candidates tie, and ranking broke ties by list position, so ties pushed the policy in a fixed direction. Noise dressed up as a gradient.
3. **Thread oversubscription.** Every worker started its own pool of BLAS threads, dozens of them, and they fought over the same cores. CMA-ES ran about 300× slower than it should have. In a wall-clock comparison, that's a free win for whichever method does less linear algebra.

None of them crashed or printed a warning. I found each one only after writing a check aimed at that exact failure, which makes me wonder what I haven't written a check for yet.

## The flatter the landscape, the smaller the edge

On the 95%-flat synthetic family, the learned walkers beat CMA-ES on 91% of problems from 20 dimensions up. At ~98.5% flat, that drops to 63%. Still ahead, but at that point every method often misses the pocket within 1,000 evaluations.

![Bar chart: learned walkers beat CMA-ES on 91% of problems when 95% of the space is flat, and 63% when about 98.5% is flat](figures/flatness.png)

*Share of problems (20 to 500 dims, 5 seeds, with and without noise) where a learned walker beat CMA-ES, pooled over three walkers. Black lines are 95% CIs.*

The fix that helped most was also the least clever: spend the first 20% of the budget on random samples, then hand over to the walker. On a 50-dimensional plateau, the warmed-up walker found the pocket in 13 of 20 runs. CMA-ES found it in 2, random search in none.

## No method wins everywhere, so the package runs a panel

rlbo ends with the strongest honeypots, and the walker gets to a strong one fastest. CMA-ES pulls ahead once the budget is large. GP BO is hard to beat at 100 to 200 dimensions, if you can wait for it. If you don't know which of these your problem looks like, hedge: split the budget across several optimizers and keep the best result. That's the default in the package, `pip install metawalker` ([PyPI](https://pypi.org/project/metawalker/)), which also ships rlbo as `method="rlbo"`.

A panel isn't free, though. Each method gets only part of the budget, and when the signal is rare that hurts. On a mix of flat and smooth test problems, the warmed-up walker alone reached the target on 11 of 15 runs, while a two-method panel with half the budget each reached it on 3. So the package's `compare()` runs candidate methods on your own losses, reports how often and how fast each one gets there, and suggests a panel, which is sometimes a single method with the whole budget. You can register your own optimizers alongside the built-in ones.

There's also a drop-in replacement for scikit-learn's `GridSearchCV`. On a 40-hyperparameter search in the demo, rlbo reached 0.93 test accuracy and the default method 0.91, against 0.88 for `RandomizedSearchCV`. On small searches (2 to 7 hyperparameters), everything ties random search. The [Colab demo](https://colab.research.google.com/drive/1nKxKFCsVaD7sqL6rE6NkNPjR3K4qFE5y?usp=sharing) runs both comparisons in a few minutes.

One caveat: I never ran a panel on the honeypot itself, so the case for panels there rests on the individual results above.

## Skills learned on cheap problems transferred to an expensive one

Hand-building honeypots costs researcher time per environment. A learned optimizer turns that into a couple hundred PPO runs. More interesting to me is that the instincts the policy learned on cheap synthetic problems (when to explore, when to distrust a result, when to look again) transferred to an expensive, noisy problem it never trained on. A lot of hyperparameter search looks like that: slow, noisy evaluations and a budget of a few hundred. So I'd expect the trick to carry over, but so far I've tested it on exactly one honeypot.

[Colab demo](https://colab.research.google.com/drive/1nKxKFCsVaD7sqL6rE6NkNPjR3K4qFE5y?usp=sharing) · [PyPI](https://pypi.org/project/metawalker/) · [GitHub](https://github.com/Itsme-aniketghosh/MetaWalker) (gridworld and all experiments)
