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

### Equal-information correction

The preceding table records the initial diagnostic, but it omits an essential
baseline. The partition-design sample is labeled. A fair comparison can use the
same observations to refit the quantile model, while continuing to reserve the
final calibration sample. After adding that baseline and rerunning seeds 0--99,
the apparent RCC advantage disappeared:

| Method | Mean cost | Mean violation probability | Runs at or below 10% violation |
| --- | ---: | ---: | ---: |
| Original global conformal | 41.952 | 0.0759 | 100/100 |
| Refitted global conformal | 39.912 | 0.0931 | 100/100 |
| Group-based | 40.708 | 0.0863 | 99/100 |
| Oracle | 39.600 | 0.0958 | 100/100 |

The refitted global method is cheaper and more reliable than the group-based
method. Consequently, the initial stress-test result should not be cited as
evidence that RCC adds value. It remains useful as a warning that information
budgets, and not only nominal training sample sizes, must be equal across methods.

A preliminary 30-seed sensitivity sweep reinforces that interpretation. With
the historical-policy cutoff fixed at ten dedicated workers per class, increasing
the split limit gave the following group-based results (the global mean cost was
41.840 and all 30 global decisions met the empirical reliability target):

| Split limit | Mean group cost | Reliable group runs |
| ---: | ---: | ---: |
| 0 | 41.840 | 30/30 |
| 1 | 41.360 | 30/30 |
| 2 | 40.960 | 30/30 |
| 4 | 40.880 | 30/30 |
| 6 | 40.573 | 30/30 |
| 8 | 40.200 | 28/30 |
| 10 | 40.307 | 28/30 |

Thus, split depth behaves like a statistical regularization parameter: finer
localization can reduce conservatism but eventually exposes the selected decision
to noisier, more selective corrections. A separate cutoff sweep showed essentially
no benefit under uniform training, with progressively larger cost reductions as
the training support became narrower. These are exploratory sensitivity results,
not tuned final comparisons.

## Queueing follow-up

`run_queueing.py` upgrades the workload proxy to a four-hour, two-class
multiskill queue. Daily call volume and class mix vary, arrivals are Poisson,
service times are lognormal, and flexible agents take the oldest waiting call
after dedicated agents serve their own classes. A day violates the constraint
when either class fails to have 80% of calls answered within one minute.

```bash
python experiments/workforce/run_queueing.py --smoke --replications 1
python experiments/workforce/run_queueing.py --replications 30
```

The implementation compares point PTO, one-sided and two-endpoint CQR global
conformal PTO, global conformal PTO after equal-information model refitting,
calibration restricted to a known
decision domain, residual-heterogeneity rectangular and oblique partitions,
the optimization-driven RCC partition from Chapter 4, fixed predicted-risk
strata, and a Monte Carlo oracle. The optimization-driven partition uses a
separate design sample to construct hyperplanes from regional extrema and an
untouched final sample to calibrate the resulting cells. Oblique alternatives
are included because the dedicated/flexible substitution frontier is diagonal
rather than axis aligned.

The initial queueing configuration trained the predictor on incumbent policies
with seven or eight dedicated workers per class, while partition design and
final calibration sampled the full grid. Before adding the refitting baseline,
a frozen 30-seed run looked exceptionally favorable to RCC:

| Method | Returned solution | Reliable solution | Mean cost when returned |
| --- | ---: | ---: | ---: |
| Global conformal PTO | 14/30 | 0/30 | 17.257 |
| Rectangular group-based | 30/30 | 30/30 | 16.213 |
| Oblique group-based | 30/30 | 30/30 | 16.187 |
| Oracle | 30/30 | 30/30 | 16.000 |

A separate 30,000-day evaluation placed the violation probabilities of the two
RCC-selected frontier plans at 0.0810 for `(8,0)` and 0.0850 for `(7,1)`; the
nearby cheaper plan `(7,0)` was clearly unreliable at 0.2049.

This is **not** yet a defensible showcase. The partition-design observations are
labeled. Using those same labels to refit the quadratic quantile model, followed
by ordinary calibration on the untouched final split, produced a reliable
oracle-cost solution in nine of the first ten seeds and a conservative reliable
solution in the tenth. With a deliberately linear predictor, the refitted global
method was reliable and oracle-cost in all ten diagnostic seeds, while RCC was
often unsafe or infeasible. A known-domain calibration baseline also matches or
beats RCC when the operational domain is specified in advance.

Together, the workload and queueing examples identify a necessary condition for a convincing
RCC application: localized calibration must add value beyond both (i) refitting
the predictive model with the same labeled information and (ii) calibrating on
any operational subgroup known before seeing calibration responses. Until an
example passes those comparisons, the queueing study should be presented as a
negative result and design diagnostic, not evidence of RCC excellence.

### Optimization-driven RCC diagnostic

The earlier queueing results called residual-heterogeneity partitions RCC, but
their split rule did not match Chapter 4. The implementation now also includes
`rcc_group_based`, a discrete-grid version of the chapter's rule. It fits both
quantile endpoints, forms the CQR score
`max(y - upper, lower - y)`, and recursively splits an ambiguous region with a
hyperplane normal to the segment joining the regional minimizer and maximizer
of the conformalized upper endpoint. On the discrete grid, the cut is moved to
the nearest parallel gap that leaves the required number of design observations
on both sides. Final corrections use only the independent calibration sample.

A preliminary ten-seed smoke sensitivity run, evaluated with the stored
4,000-day oracle, shows the locality/mass tradeoff directly:

| Maximum splits | Mean terminal cells | Reliable RCC selections |
| ---: | ---: | ---: |
| 4 | 5.0 | 0/10 |
| 6 | 7.0 | 0/10 |
| 10 | 11.0 | 1/10 |
| 16 | 16.3 | 6/10 |

The recursion cannot refine much further at the smoke design-sample size because
of the minimum cell-mass requirement. More importantly, the failures illustrate
the difference between group-average and optimizer-level validity. For seed 3
with 16 splits, RCC selected `(4,3)`, whose oracle violation probability is
0.28075. The four staffing plans in its terminal cell have uniform-average
violation probability 0.08194: the optimizer selected the riskiest point inside
an acceptable-looking group. Each RCC result now records this selected-cell
diagnostic, and the summary reports the average optimizer-selection gap.

These numbers are diagnostic rather than a final benchmark: they use the smoke
training, design, and calibration sizes. They nevertheless give the call-center
example a useful theoretical role. It exhibits exactly why restricted
conditional coverage over a region does not automatically provide a guarantee
at a data-dependent optimizer inside that region, and why the shrinking-cell
conditions in the chapter matter.
