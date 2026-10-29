### Scenario `plateau`: 1000 evaluations per run

Median log10 regret (lower is better; 0 = a typical random point), average rank over 30 problem instances, and median evaluations and seconds per run.

| method | plateau-50 | plateau-100 | plateau-200 | avg rank | evals | s/run |
|---|---|---|---|---|---|---|
| walker | -1.48 | -1.48 | -1.60 | 2.50 | 1000 | 0.3 |
| gp | -1.34 | -0.99 | -1.36 | 3.10 | 1000 (3 hit the 7200 s cap) | 2829.1 |
| walker_hp | -1.42 | -1.36 | -1.47 | 3.13 | 1000 | 0.4 |
| tpe | -1.20 | -1.07 | -1.13 | 3.53 | 1000 | 107.5 |
| rlcma | -1.23 | -1.18 | -0.73 | 3.80 | 1000 | 0.3 |
| cmaes | -0.46 | -0.58 | -0.63 | 5.60 | 1000 | 0.3 |
| random | -0.31 | -0.30 | -0.48 | 6.33 | 1000 | 0.1 |

Win rate of **walker** (our best by average rank) against each baseline, ties count 1/2:

| vs | win rate | 95% Wilson CI | instances |
|---|---|---|---|
| cmaes | 0.90 | [0.74, 0.97] | 30 |
| tpe | 0.70 | [0.52, 0.83] | 30 |
| gp | 0.67 | [0.49, 0.81] | 30 |
| random | 0.93 | [0.79, 0.98] | 30 |

### Scenario `wallclock`: 60 s of wall-clock per run

Median log10 regret (lower is better; 0 = a typical random point), average rank over 50 problem instances, and median evaluations and seconds per run.

| method | plateau-100 | ellipsoid-200 | rastrigin-200 | plateau-50 | plateau-200 | avg rank | evals | s/run |
|---|---|---|---|---|---|---|---|---|
| cmaes | -1.71 | -1.59 | -0.71 | -1.61 | -1.77 | 1.84 | 149938 | 60.0 |
| rlcma | -1.78 | -1.53 | -0.43 | -1.65 | -1.60 | 2.42 | 180846 | 60.0 |
| walker | -1.59 | -1.40 | -0.38 | -1.64 | -1.62 | 2.92 | 170496 | 60.0 |
| walker_hp | -1.61 | -1.35 | -0.32 | -1.50 | -1.52 | 3.46 | 149454 | 60.0 |
| tpe | -1.13 | -0.76 | -0.11 | -1.28 | -1.10 | 5.40 | 587 | 60.1 |
| random | -1.30 | -0.55 | -0.11 | -1.27 | -1.02 | 5.78 | 409682 | 60.0 |
| gp | -0.91 | -0.57 | -0.14 | -0.92 | -1.11 | 6.18 | 180 | 60.4 |

Win rate of **rlcma** (our best by average rank) against each baseline, ties count 1/2:

| vs | win rate | 95% Wilson CI | instances |
|---|---|---|---|
| cmaes | 0.26 | [0.16, 0.40] | 50 |
| tpe | 0.94 | [0.84, 0.98] | 50 |
| gp | 1.00 | [0.93, 1.00] | 50 |
| random | 0.94 | [0.84, 0.98] | 50 |
