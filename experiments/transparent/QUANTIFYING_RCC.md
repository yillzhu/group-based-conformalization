# When should RCC's effect disappear?

The useful question is not simply whether the predictor is good. We should ask
how prediction error creates different calibration corrections in different
regions, how sensitive the optimization is to these corrections, and whether
an apparent saving comes from genuine regional structure or calibration noise.
This note separates these questions. The bounds below are population statements
under explicit assumptions, not new guarantees for the finite-sample RCC code.

## 1. A direct bound in terms of quantile error

Write `L*(x), U*(x)` for the true conditional `alpha/2` and `1-alpha/2`
quantiles. Assume continuous conditional distributions, with positive density
near these quantiles. Suppose both fitted endpoints have uniform error at most
`epsilon`. For a group `G` of positive probability, define

`S = max(Y-U(X), L(X)-Y)`,

`q_G = conditional (1-alpha)-quantile of S given X in G`.

Then `-epsilon <= q_G <= epsilon`, for every such group. To see this, expanding
the fitted interval by `epsilon` contains the true central interval at every
point; shrinking it by `epsilon` is contained in that interval, or is empty.
These inclusions bracket the score quantile. Strict local increase rules out
an earlier quantile on a flat CDF plateau.

Consequently, the global and any local population correction differ by at most
`2 epsilon`. Either corrected upper endpoint differs from `U*` by at most
`2 epsilon`. This applies simultaneously to all positive-mass groups, even if
a group is selected adaptively: the premise is a uniform bound on the fixed
fitted predictor and the corrections here are population quantities.

In particular, exact conditional quantiles give `q_G=0` everywhere. There is
then no population calibration correction for RCC to improve.

Turning an endpoint bound into an objective bound requires optimization
stability. For example, if the fitted upper endpoint is strictly decreasing
with slope magnitude at least `kappa` throughout a scalar capacity domain,
the global and piecewise-locally-corrected minimum feasible capacities differ
by at most `2 epsilon / kappa`. Both roots must exist, and the piecewise rule
must be optimized over the entire domain. This concerns the ideal corrected
feasible sets, not an algorithm that drops regions or uses approximate extrema.
With objective `c*x`, multiply by `c`.

For a general nonconvex feasible set there need not be a useful cost bound:
a tiny endpoint perturbation can remove an isolated cheap feasible component.
That is an optimization-conditioning issue, not necessarily better calibration.

If the conditional response density is bounded above by `M` between `U*` and
the corrected upper endpoint, the pointwise violation probability is at most
`alpha/2 + 2 M epsilon`. This stronger conclusion uses uniform endpoint error;
an average loss or regional coverage statement alone does not imply it.

## 2. A bound using average pinball loss

Uniform approximation is demanding. A second route quantifies how *average*
model error limits regional corrections. Let `R` be the sum of the two expected
pinball losses minus the corresponding oracle losses, conditional on the fitted
model. These are absolute losses, not the relative excess percentages in our
tables.

Assume that, at almost every covariate, the conditional density along the
segments between fitted and true endpoints lies between constants `m>0` and
`M<infinity`. The standard pinball identity gives

`E[(L-L*)^2 + (U-U*)^2] <= 2 R / m`.

Indeed, for one endpoint the excess conditional loss is the integral of
`F_x(t)-tau` between its true and fitted quantile; the density lower bound gives
at least `m/2` times squared endpoint error. The loss-to-quantile relationship
is established more generally in
[Steinwart and Christmann (2011)](https://arxiv.org/abs/1102.2101).

Let `p_G=P(X in G)` and `H_G(t)=P(S<=t | X in G)`. The density upper bound and
Cauchy--Schwarz imply

`|H_G(0) - (1-alpha)| <= 2 M sqrt(R / (m p_G))`.

Now assume the score CDF has slope at least `c0>0` in a neighborhood of zero
large enough to contain the relevant quantile. Inverting that CDF gives

`|q_G| <= (2 M / c0) sqrt(R / (m p_G))`.

Here `delta` is a lower bound on true group probability; the algorithm's
empirical count test does not by itself certify that premise.
For regions of probability at least `delta`, the difference between a local
and global population correction is therefore bounded by

`(2 M / c0) sqrt(R/m) (1 + 1/sqrt(delta))`.

This is a sufficient-condition bound, not necessarily a sharp one. Its message
is useful: as quantile loss approaches oracle loss, population corrections
vanish; smaller groups make localization harder. The assumptions matter. In
particular, the density lower bound can fail for rare-event mixtures, nearly
discrete responses, or predictions outside a bounded conditional support.
The score-CDF inverse can also be badly conditioned. We have not certified
these constants for every fitted model in the experiments.

The bound holds for any group of the stated mass because it follows from a
single global squared-error integral. It does **not** imply that the fitted
endpoint is good at the optimizer's particular point. A predictor can have
small average loss and a severe error on a tiny set that attracts optimization.
Thus disappearing RCC corrections do not by themselves establish decision
safety. This distinction is consistent with the conditional-coverage
limitations discussed by
[Barber et al. (2021)](https://arxiv.org/abs/1903.04684).

## 3. Separate population repair from sampling and selection

For the cell chosen by empirical RCC, calculate the population corrections
using its actual fitted predictor and actual interval. There is an exact
algebraic identity:

`qhat_local - qhat_global = (q_local - q_global) + remainder`,

where

`remainder = (qhat_local-q_local) - (qhat_global-q_global)`.

The first term measures a genuine difference in population corrections at the
selected cell. The remainder includes sampling error, finite-rank adjustment,
and the effects of choosing that cell using the same calibration labels. It
is **not** an independent or mean-zero error term. Even the population term
is evaluated at a sample-selected cell, not a counterfactual population RCC
partition.

`PopulationScoreLaw` computes these population quantities without a simulation
grid. On intervals between true noise boundaries and fitted spline knots,
conditional interval coverage is a clipped affine function of capacity.
Splitting at its clipping and endpoint-crossing locations gives an exact
trapezoidal integral. Scalar root finding then inverts the integrated score
CDF. Negative corrections and empty shrunken intervals are handled explicitly.

On any common affine segment `U(x)=a-b*x`, with both selected constraints active,
the capacity difference is exactly `(qhat_local-qhat_global)/b`. Outside those
conditions corrections still describe a mechanism, but should not be equated
numerically with capacity savings.

## 4. Harder-model stress and competing explanations

The new stress has the same continuous capacity decision and iid full-domain
training support. Outside a quiet operating interval, the noise scale switches
between 0.05 and 0.8 across many short regimes. Both an aligned quiet prefix and
an interior quiet interval are tested. These are synthetic stresses, not new
real-world OR benchmarks, and the discontinuous noise scale is a deliberate
construction.

The learner is allowed up to 96 leaves, selected by independent validation
pinball loss, with at least 40 training observations per leaf. Unlike the
historical model's default setting, the minimum leaf size does not grow as a
fixed fraction of the data. Thus the stress does not impose a fixed 100-leaf
ceiling as sample size increases. No regime boundaries or true scales are
supplied to fitting or optimization. RCC remains the original adaptive rule.

Comparators include the raw learned upper endpoint, ordinary global CQR,
RCC, and two inexpensive controls:

- Scale-normalized global CQR divides each CQR score by its learned interval
  halfwidth, calibrates once globally, and scales the correction back. It uses
  the identical fitted model and calibration observations as RCC.
- Refit global CQR adds half the calibration observations to the original
  training set, selects the model on the original validation set, and calibrates
  on the untouched second half. It uses the same total labeled-data budget as
  RCC. The validation objective is unchanged. This is an experimental comparator,
  not a change to the RCC algorithm or a revival of the shelved refitted-RCC idea.

The exact-response simulator diagnoses conditional-quantile error, population
corrections, and risk only after fitting and decision selection. Neither the
oracle nor the evaluation metrics choose a model or remove a replication.
The construction was first inspected in a 12-replication exploratory pilot;
the saved follow-up uses a separate seed and 100 replications per setting.
This is a development experiment, not a preregistered test or a learning lower
bound against all possible ML models.

## 5. Results: disappearance and a sampling floor

The new study has 100 replications at each setting and a new replication seed,
20261027. Validation and calibration each use 4,000 observations throughout.
The aligned simple case gives:

| Training observations | Full-domain excess pinball loss | RCC saving relative to global cost | Global/RCC decisions meeting 10% risk |
| ---: | ---: | ---: | ---: |
| 100 | 24.315% | 11.482% | 64/79 out of 100 |
| 250 | 3.382% | 0.331% | 97/98 out of 100 |
| 500 | 1.790% | 0.200% | 99/100 out of 100 |
| 2,000 | 0.478% | 0.110% | 100/100 |
| 8,000 | 0.141% | 0.056% | 100/100 |
| 32,000 | 0.036% | 0.051% | 100/100 |

These percentages are ratios of mean cost differences to global mean cost,
not averages of per-run percentages. The large saving at 100 training rows
is accompanied by many operationally unreliable decisions. It is not a
successful fulfillment of the four criteria.

At 32,000 rows, the mean empirical local-minus-global correction is -0.000554.
Its population component is -0.000086, and the sampling/selection remainder
is -0.000468. Thus approximately 85% of this mean correction gap is in the
remainder. This is an exact decomposition in score units, not a claim that
85% of every individual cost difference is caused by sampling alone. Calibration
size is fixed, so the finite-sample adaptive effect need not vanish with training
size. Near-optimum excess pinball loss is also small: 0.044% at 32,000 rows.

## 6. Results: making conditional quantiles harder is not sufficient

The six harder-model settings give the following mean costs. Counts in
parentheses are decisions with exact risk at most 10%, out of 100. Every
method selects a decision in every run.

| Noise pattern | Training observations | Global CQR | RCC | Refit global CQR |
| --- | ---: | ---: | ---: | ---: |
| Complex aligned | 500 | 1.09803 (97) | 1.09619 (95) | 1.09339 (94) |
| Complex aligned | 2,000 | 1.09371 (99) | 1.09080 (95) | 1.09280 (99) |
| Complex aligned | 8,000 | 1.09156 (100) | 1.09258 (100) | 1.09146 (100) |
| Complex interior | 500 | 1.15846 (97) | 1.12430 (97) | 1.09822 (96) |
| Complex interior | 2,000 | 1.10013 (96) | 1.09121 (90) | 1.09303 (97) |
| Complex interior | 8,000 | 1.09288 (100) | 1.09151 (100) | 1.09235 (100) |

For the complex-interior case at 500 training rows, RCC's saving over ordinary
global CQR is 2.949%. Mean exact risks are 1.47% for global, 2.35% for RCC,
and 2.29% for refit. The refit rule is substantially cheaper on average than
RCC but has one additional unreliable replication; these numbers do not
establish strict dominance or matched-risk superiority. Scale-normalized global
CQR has cost 1.13898, mean risk 3.37%, and 96 reliable decisions in that setting.
It captures part of the global-versus-local gain without partitioning.

For the complex-interior case at 8,000 training rows, all methods meet the risk
target in all replications, but RCC's saving over global is only 0.125%.
Full-domain excess pinball loss is still 12.27%, while near-optimum excess
loss is only 0.103%. Thus even appreciable global quantile error need not
translate into a large useful RCC advantage. In the complex-aligned case at
that training size, RCC is actually costlier.

The study provides positive evidence that regional corrections can repair
uncertainty errors of a finite-data learner. It does **not** supply the desired
compelling OR example against strong, equal-information alternatives. In
particular, the favorable low-training comparisons give RCC 4,000 calibration
labels to adapt with, in addition to training and validation labels; it would
be misleading to describe them as successes using only 500 total observations.

The interpretation is that RCC adds a geometric, piecewise-constant residual
correction to a base quantile model. This is itself a restricted way of learning
uncertainty, not an escape from learning it. A promising application would have
decision-relevant residual structure simpler than the entire conditional
quantile function, with feasible-set geometry revealing that structure more
efficiently than the competing learners. It should show a substantial population
correction contrast, persist on new replications, beat simple normalization and
equal-budget refitting, and improve useful decisions without concealing risk
failures. The current synthetic stresses do not yet establish that application.

## Reproduction

```bash
python experiments/transparent/run_quantification.py --replications 100 \
  --output results/transparent/quantification_audit.json
python -m unittest tests.test_transparent_population tests.test_transparent_learned
```

The saved artifact records all seeds, exact regime boundaries and scales,
sample sizes, library versions, summaries, and paired per-replication
decisions/quality/correction diagnostics, including failures. It is separate
from the earlier studies; those artifacts are unchanged.
