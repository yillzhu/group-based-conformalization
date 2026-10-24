# Cross-trained workforce-planning experiment

This experiment is a deliberately small test of the paper's proposed practical
niche.  A planner chooses `d` dedicated workers for each of two request classes
and `f` more expensive cross-trained workers.  Daily total workload and its class
mix are random.  Dedicated workers serve their own class; the flexible pool then
covers the combined overflow.  A decision violates the service requirement when
unmet workload remains.

The experiment compares four discrete-grid decisions:

1. **Point PTO:** optimize subject to a fitted conditional-mean constraint.
2. **Global conformal PTO:** optimize subject to a fitted upper quantile plus one
   global one-sided conformal correction.
3. **Group-based:** design a decision-relevant partition on a separate sample,
   then estimate cellwise corrections on untouched final calibration data.
4. **Oracle:** use an independent, large Monte Carlo panel.

The main configuration is a stress test for a common prescriptive-learning
problem: historical outcomes were collected under relatively conservative
policies.  The regression model is therefore trained only on decisions with at
least ten dedicated workers per class, while a later randomized pilot supplies
partition-design and final-calibration observations across the full decision
grid.  The global correction must absorb extrapolation errors from the entire
pilot distribution.  A local correction can help only if the cost-effective
reliable solution lies in a better-learned part of the grid.

Setting `training_min_dedicated=4` removes this shift and serves as a negative
control.  If a well-trained base quantile model already adapts to the relevant
uncertainty, the global correction should be close to zero and group-based
calibration has no principled reason to improve it.

The three-way training/partition-design/final-calibration split is intentional.
It removes the paper's known problem of using the same calibration responses to
choose a partition and to calibrate its cells.  It does **not** by itself turn
cell-average predictive coverage into pointwise coverage at the optimized
decision.  The independent Monte Carlo evaluation is therefore an empirical
test, not a claimed finite-sample optimizer guarantee.

This first model is workforce planning, not yet a queueing model: it represents
one-period workload coverage and pooled recourse but not arrival times, waiting
times, abandonment, or a call-center service-level constraint.  The queueing
upgrade is warranted only if the proposed mechanism is visible here.

Run a quick diagnostic with:

```bash
python experiments/workforce/run.py --smoke
```

The full default uses larger independent samples:

```bash
python experiments/workforce/run.py
```

For the 100-seed comparison used to decide whether the example is promising:

```bash
python experiments/workforce/run.py --replications 100 \
  --output results/corrected/workforce/stress_test.json
python experiments/workforce/run.py --replications 100 --negative-control \
  --output results/corrected/workforce/negative_control.json
```

The default refuses to overwrite an existing result file.  Each JSON contains
the exact configuration and selected decision for every seed, as well as means,
empirical reliability rates, and paired cost-comparison frequencies.

## Initial 100-seed diagnostic

The following results use seeds 0--99, 8,000 observations each for training and
partition design, 10,000 untouched calibration observations, and 30,000
independent evaluation scenarios per seed.

| Configuration and method | Mean cost | Mean violation probability | Runs at or below 10% violation |
| --- | ---: | ---: | ---: |
| Historical-policy stress test: global | 41.952 | 0.0759 | 100/100 |
| Historical-policy stress test: group-based | 40.404 | 0.0891 | 98/100 |
| Historical-policy stress test: oracle | 39.600 | 0.0958 | 100/100 |
| Uniform-training control: global | 39.048 | 0.1019 | 77/100 |
| Uniform-training control: group-based | 39.472 | 0.0996 | 78/100 |
| Uniform-training control: oracle | 39.600 | 0.0958 | 100/100 |

In the stress test, group-based calibration was cheaper than global calibration
in 62 runs, tied it in 34, and cost more in four. Two localized decisions had
violation probabilities around 12.5--12.7%, so the result does not justify an
optimizer-level guarantee. In the negative control, localization did not improve
cost or reliability. These failures are retained because they expose precisely
where group-average conformal validity does not transfer to a selected point.
