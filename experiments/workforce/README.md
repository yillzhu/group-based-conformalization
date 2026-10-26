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

## Continuous-FTE RCC experiment

`run_continuous.py` returns to the domain assumed by the original Chapter 4
code. The decisions are real-valued dedicated and flexible FTE capacity over
`[4,15] x [0,18]`; the workload simulator in `problem.py` is already defined for
real-valued capacity. The fitted quantile endpoints are affine, so their extrema
over every polyhedral region and the final linear-objective problems are solved
exactly by polygon vertex enumeration. There is no staffing grid in RCC. The
Monte Carlo oracle alone searches dedicated FTE on a documented fine grid while
optimizing flexible FTE continuously.

This implementation deliberately reproduces the original adaptive algorithm:
the same calibration sample constructs the partition and its local CQR
corrections; the original delta-based representation test is used; undersized
children inherit their parent's correction and enter the unresolved queue;
surrogate-safe and unresolved regions are optimized separately; and all regions
remaining at the split limit are treated as unresolved. It therefore reproduces
the original computation, including the adaptive-partition statistical caveat,
rather than the independent partition-design variant.

```bash
python experiments/workforce/run_continuous.py --smoke --replications 1
python experiments/workforce/run_continuous.py --replications 30 \
  --output results/corrected/workforce/continuous_rcc_30_seeds.json
```

The initial full-size 30-seed run gives a modest comparison rather than a
showcase:

| Method | Reliable selections | Mean cost | Mean violation probability |
| --- | ---: | ---: | ---: |
| Point PTO | 0/30 | 23.882 | 0.8283 |
| Global CQR PTO | 26/30 | 40.658 | 0.0862 |
| Continuous RCC | 27/30 | 40.864 | 0.0844 |
| Monte Carlo oracle | 16/30 | 39.143 | 0.1001 |

RCC was cheaper than global CQR in 6 runs, tied in 1, and costlier in 23; its
mean paired cost difference was `+0.207`. The oracle is constructed on the
empirical 0.10 boundary, so independent Monte Carlo evaluation places it above
or below 0.10 with roughly equal frequency; its 16/30 classification should not
be read as an oracle failure. These results show that the literal continuous
implementation works and yields sensible fractional solutions, but they do not
yet demonstrate a material RCC advantage over global CQR.

## Continuous-capacity QED call-center experiment

`run_continuous_queue.py` restores the congestion and pooling structure that the
continuous workload proxy lacked. For scenario-specific offered loads (L_i),
the square-root staffing approximation requires

```text
r_i = L_i + beta * sqrt(L_i)
```

capacity in class `i`. Dedicated FTE is used first and a shared cross-trained
pool covers the remaining requirement, giving the continuous response

```text
y(d, f) = sum_i max(r_i - d, 0) / eta - f.
```

A day violates the service requirement when `y(d,f) > 0`. Daily volume, surge
incidence, and class mix are random. The default parameters use `beta=1.6`,
cross-training efficiency `eta=0.9`, and cost `2 d + 1.4 f`. The operational
domain is `[4,12] x [0,9]`; the Monte Carlo oracle is interior, near `(5.5,4.3)`.
Predictive models are trained on incumbent policies in `[4,9] x [1,6]`.

The first ten seeds were used exploratorily to choose a prespecified safety
buffer: both global CQR and RCC are calibrated at 5% miscoverage and evaluated
against a 10% operational violation target. This buffer is essential. On the
30-seed holdout `10--39`, nominal 10% RCC was unsafe in all 30 runs, with mean
risk 0.1283. The frozen buffered specification gave:

| Method | Reliable selections | Mean cost | Mean violation probability |
| --- | ---: | ---: | ---: |
| Point PTO | 0/30 | 12.999 | 0.6371 |
| Buffered global CQR | 30/30 | 22.323 | 0.0069 |
| Buffered RCC | 30/30 | 18.445 | 0.0669 |
| Monte Carlo oracle | boundary estimate | 17.052 | 0.1005 |

The same frozen configuration was then evaluated on 100 untouched seeds
`40--139`:

| Method | Reliable selections | Mean cost | Mean violation probability |
| --- | ---: | ---: | ---: |
| Point PTO | 0/100 | 12.983 | 0.6403 |
| Buffered global CQR | 100/100 | 22.373 | 0.0071 |
| Buffered RCC | 100/100 | 18.444 | 0.0672 |
| Monte Carlo oracle | boundary estimate | 17.058 | 0.1003 |

RCC was cheaper than global CQR in all 100 confirmation runs, by `3.929` on
average, and recovered approximately 74% of the global-to-oracle cost gap. Its
largest independently estimated violation probability was 0.0953. Reproduce the
confirmation with:

```bash
python experiments/workforce/run_continuous_queue.py --buffered \
  --seed 40 --replications 100 \
  --output results/corrected/workforce/continuous_queue_confirmation.json
```

This is promising evidence for the operational value of localization: the
global safety buffer is extremely conservative because it must accommodate the
whole decision domain, while RCC spends the same buffer locally and approaches
the oracle cost. It is not a nominal optimizer-level coverage theorem. Every
selected RCC solution came from an unresolved region, and the unbuffered
negative control demonstrates the adaptive optimizer-selection gap directly.
The model is a standard square-root staffing approximation rather than a full
discrete-event queue, so a later study should test whether the result survives a
higher-fidelity simulator.

### Predictive-capacity stress test

The original call-center comparison uses affine point and quantile regressions.
To determine whether this low-capacity choice creates the RCC advantage,
`run_continuous_queue.py` also supports continuous piecewise-affine models in
dedicated staffing through `--spline-knots`. The flexible-staff effect remains
linear, matching the simulator. RCC intersects each polyhedral region with the
spline strips, so all regional extrema and final optimization problems remain
exact over the continuous domain; this diagnostic does not introduce a staffing
grid.

We fixed capacities at 0, 3, 7, and 15 equally spaced knots and compared two
training designs on the same seeds 140--169. The incumbent design uses only
historical policies in `[4,9] x [1,6]`; the full-support design samples the whole
operational domain `[4,12] x [0,9]`. All runs retain the prespecified 5% conformal
calibration buffer and the 10% operational risk target.

| Training support | Model capacity | Upper pinball loss | Global cost | Global risk | RCC cost | RCC risk | RCC minus global cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Incumbent | 0 | 0.6145 | 22.464 | 0.0062 | 18.515 | 0.0642 | -3.949 |
| Incumbent | 3 | 0.4704 | 21.225 | 0.0115 | 18.532 | 0.0466 | -2.693 |
| Incumbent | 7 | 0.4337 | 20.479 | 0.0262 | 18.448 | 0.0494 | -2.030 |
| Incumbent | 15 | 0.7888 | 20.449 | 0.0342 | 18.334 | 0.0542 | -2.115 |
| Full domain | 0 | 0.1488 | 16.804 | 0.1459 | 16.930 | 0.1374 | +0.126 |
| Full domain | 3 | 0.1312 | 18.175 | 0.0549 | 18.332 | 0.0500 | +0.156 |
| Full domain | 7 | 0.1321 | 18.258 | 0.0530 | 18.361 | 0.0511 | +0.103 |
| Full domain | 15 | 0.1336 | 18.255 | 0.0534 | 18.281 | 0.0538 | +0.026 |
| Incumbent, ReLU | 16 units | 0.1581 | 18.124 | 0.0589 | 18.184 | 0.0562 | +0.061 |
| Full domain, ReLU | 16 units | 0.1343 | 17.828 | 0.0675 | 18.128 | 0.0569 | +0.300 |

With incumbent-only data, added spline capacity improves upper-quantile loss and
shrinks the RCC advantage, but cannot remove the need to extrapolate. The
15-knot loss deterioration confirms that extra parameters alone do not solve
the support problem. With full-domain data, an affine uncertainty model is too
misspecified to meet the risk target, despite its low global conformal
correction. Once three or more knots are used, both uncertainty-aware methods
are reliable and their costs are essentially equal; RCC is slightly more
conservative on average. Thus the large showcase result is best understood as
localized repair of prediction error under limited training support, not a
general advantage over a well-specified conditional-quantile model.

As a separate nonlinear check, the last two rows use a one-hidden-layer neural
network with 16 ReLU units. The point network minimizes squared error. The two
quantiles share a hidden layer and minimize pinball loss; their output heads are
parameterized so that the upper prediction is at least the lower prediction for
every input. This avoids relying on a soft crossing penalty, which can fail when
the optimizer extrapolates beyond the training domain. The fitted networks had
zero crossings on all independent diagnostic points. A one-hidden-layer ReLU
network is piecewise affine. In this two-dimensional example, the implementation
enumerates its activation polygons inside each RCC region, so regional extrema
and final decisions are again continuous and exact rather than grid based.

The neural result reinforces the qualified interpretation. Its upper-quantile
loss under incumbent-only training is much lower than the spline losses, and the
large RCC cost advantage disappears: RCC costs `0.061` more on average for a
small risk reduction. With full-domain training it costs `0.300` more. Global
CQR and RCC satisfy the empirical 10% target in all 30 incumbent runs and in all
30 full-domain runs. The flexible uncertainty model has absorbed most of the
regional structure that RCC previously corrected.

The point-PTO decisions remain unsafe even with splines (mean risks roughly
0.46--0.47 for the nonlinear fits), because improving a conditional-mean model
does not turn it into a chance-constraint model. Reproduce the paired comparison
with, for each `K` in `0 3 7 15`:

```bash
python experiments/workforce/run_continuous_queue.py --buffered \
  --spline-knots K --seed 140 --replications 30 --output incumbent-K.json
python experiments/workforce/run_continuous_queue.py --buffered \
  --full-support-training --spline-knots K --seed 140 --replications 30 \
  --output full-support-K.json
python experiments/workforce/run_continuous_queue.py --buffered \
  --neural-network --seed 140 --replications 30 --output incumbent-nn.json
python experiments/workforce/run_continuous_queue.py --buffered \
  --neural-network --full-support-training --seed 140 --replications 30 \
  --output full-support-nn.json
```

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

The full-size 30-seed runs make the optimizer-selection effect sharper. They can
be reproduced without resimulating the oracle by passing the earlier result as
`--oracle-file`:

```bash
python experiments/workforce/run_queueing.py --replications 30 --max-splits 6 \
  --oracle-file results/corrected/workforce/queueing_30_seeds.json \
  --output results/corrected/workforce/queueing_rcc_30_seeds.json
python experiments/workforce/run_queueing.py --replications 30 --max-splits 16 \
  --oracle-file results/corrected/workforce/queueing_30_seeds.json \
  --output results/corrected/workforce/queueing_rcc_30_seeds_splits16.json
```

| Maximum splits | Reliable selections | Mean cost | Mean selected risk | Mean selected-cell risk | Mean selection gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0/30 | 12.640 | 0.536 | 0.176 | 0.360 |
| 16 | 0/30 | 10.627 | 0.665 | 0.109 | 0.556 |

Thus, deeper recursion brings the average risk of the selected cells closer to
the nominal 0.10 level but makes the selected decisions worse in this experiment.
Finer groups create more locally favorable corrections for the optimizer to
exploit without establishing pointwise validity inside those groups.

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
