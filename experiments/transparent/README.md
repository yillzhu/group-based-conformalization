# A transparent capacity problem

This is a mechanism study with exact answers. It asks when RCC's recursive
partition can improve a single global residual correction, and whether that
benefit survives an uncertainty model that already accounts for local scale.
It is a constructed example, not a claim of empirical or methodological
superiority over current optimization methods.

## Problem and information

Choose continuous capacity `x` in `[0,4]` to minimize cost `x`. A future constraint
response is

```text
Y(x) = 1 - x + sigma(x) Z,       Z ~ Uniform[-1,1].
```

The operational requirement is `P(Y(x)>0) <= 0.10`. Calibration inputs are iid
uniform over the **entire** domain `[0,4]`, independently of `Z`.

Every method receives the exact mean predictor `m(x)=1-x`. This removes mean
prediction error from the experiment. It does not supply a correct conditional
quantile model: the distribution of the residual changes with `sigma(x)`.
There is no fitted ML model, training-support gap, grid search, or Monte Carlo
error in the principal comparison.

For the unscaled conformal methods, both base interval endpoints are set to the
mean, so the CQR score reduces to `|Y-m(x)|`. This is residual conformal prediction,
expressed through the existing CQR implementation. A correction `w` gives the
interval `[1-x-w, 1-x+w]` and the robust surrogate constraint `x >= 1+w`.

The 90% prediction interval and the 10% operational risk target are distinct.
Within a constant-scale region the interval is symmetric, so its upper endpoint
has 5% upper-tail risk. No extra empirical safety buffer is tuned here.

## Exact risk and optimal decision

The true risk is available without simulation:

```text
r(x) = clip((1 - (x-1)/sigma(x))/2, 0, 1).
```

In a region with constant scale `s`, the 10% chance constraint is equivalent to
`x >= 1+0.8s`. Thus the oracle can be solved by inspecting each interval's left
endpoint and this threshold. Its optimal capacity is `1.080` in all three cases.

For a group `G=[a,b]`, the population CQR correction solves

```text
sum_j weight_j(G) min(w/sigma_j, 1) = 0.9,
```

where `weight_j(G)` is the fraction of that interval in regime `j`. The code
inverts this piecewise-linear CDF exactly. RCC's fitted endpoints are affine in
`x`, so extrema, feasibility-boundary cuts, and final optimization are continuous
and analytic as well.

## Three cases and the population comparison

The cases are designed to isolate different mechanisms:

1. **Homogeneous:** `sigma=0.1` throughout the domain.
2. **Aligned:** `sigma=0.1` on `[0,1.2)` and `sigma=0.4` on `[1.2,4]`.
3. **Interior quiet region:** `sigma=0.1` on `[0.9,1.4)` and `sigma=0.4` elsewhere.

Intervals are left closed/right open, except for the domain's right endpoint.
RCC uses `delta=0.1`, at most 64 splits, and the original unresolved-region
constraint fallback. In the population calculation the corrections and group
masses are exact, so the comparison isolates geometry and the mass threshold.

| Case | Global cost / risk | RCC cost / risk | Scale-aware conformal cost / risk |
| --- | --- | --- | --- |
| Homogeneous | 1.090 / 5.00% | 1.090 / 5.00% | 1.090 / 5.00% |
| Aligned | 1.343 / 7.14% | 1.090 / 5.00% | 1.090 / 5.00% |
| Interior quiet region | 1.354 / 0.00% | 1.284 / 0.00% | 1.090 / 5.00% |

In the aligned case, the global residual distribution is a mixture of 30% quiet
and 70% noisy inputs. Its 90th percentile is
`w_global = 0.4(1-0.1/0.7) = 0.342857`. Hence global calibration selects
`x=1.342857`. This decision is in the noisy region and has true risk 7.14%.

RCC's successive left-hand groups illustrate its benefit:

| Group | Population correction | Next fitted boundary |
| --- | ---: | ---: |
| `[0,4]` | 0.342857 | 1.342857 |
| `[0,1.342857]` | 0.097803 | 1.097803 |
| `[0,1.097803]` | 0.090000 | 1.090000 |

The last group is wholly quiet. Its correction is exactly `0.9*0.1`; choosing
`x=1.09` therefore gives a 5% violation probability. In this example localization
reduces cost by 18.8% **and** reduces risk. The calculation does not infer a
pointwise guarantee from average group coverage: it uses the explicit fact that
the residual distribution is identical at every point in the final quiet group.

In the interior case, the true quiet region has probability `0.5/4=0.125`, above
the default `delta=0.1`. Nevertheless, the cuts RCC generates do not isolate a
useful quiet region before the representation rule stops refinement. The method
improves on global calibration but remains well above cost 1.09. A prespecified
population sensitivity calculation gives:

| delta | Interior-case RCC cost |
| ---: | ---: |
| 0.20 | 1.339810 |
| 0.10 | 1.283543 |
| 0.05 | 1.090000 |
| 0.02 | 1.090000 |
| 0.01 | 1.090000 |

This is a geometry/mass tradeoff, not a reason to choose delta after observing
test performance. Smaller groups also make finite-sample calibration harder.

## What the stronger controls know

`known_regions_cqr` calibrates independently within the predefined true scale
intervals. It knows their boundaries but estimates residual corrections from the
same observations as RCC.

`known_scale_cqr` uses the normalized score `|Y-m(x)|/sigma(x)`, pools all
calibration observations, and multiplies the resulting cutoff by `sigma(x)`.
It knows the true relative scale function. Its score distribution is the same
at every decision, and it attains cost 1.09 in all three population cases.

These controls deliberately reveal the structural information that RCC is trying
to recover through localization. They are not trained off-the-shelf ML baselines
and should not be represented as such. They establish that the improvement over
global unscaled calibration is a benefit of accounting for residual scale; it is
not unique to RCC. A correct conditional 90th-percentile model would directly
recover the chance oracle at cost 1.08.

## Finite calibration samples

The saved run uses 200 independent calibration datasets per case, with 4,000 iid
input/response observations in each dataset. Every method uses the same labeled
observations within a replication. All selected decisions are evaluated using
the exact risk formula.

Empirical RCC uses the chapter's count rule, the adjusted finite-sample order
statistic, adaptive recalibration on the same sample, and the inherited-correction
fallback for undersized/pending regions. A test compares its decision with our
existing exact polygon-based continuous RCC solver using the same observations.
It does not inherit the power audit's safe-only modification.

| Case | Method | Mean cost | Mean true risk | Runs meeting 10% target |
| --- | --- | ---: | ---: | ---: |
| Homogeneous | Global | 1.089996 | 5.002% | 200/200 |
| Homogeneous | RCC | 1.089484 | 5.258% | 200/200 |
| Homogeneous | Scale-aware | 1.089996 | 5.002% | 200/200 |
| Aligned | Global | 1.342973 | 7.128% | 200/200 |
| Aligned | RCC | 1.089889 | 5.056% | 200/200 |
| Aligned | Scale-aware | 1.090033 | 4.984% | 200/200 |
| Interior quiet region | Global | 1.354529 | 0.000% | 200/200 |
| Interior quiet region | RCC | 1.236982 | 1.260% | 200/200 |
| Interior quiet region | Scale-aware | 1.090058 | 4.971% | 200/200 |

The finite RCC results need not coincide with the population results: the
empirical representation threshold is below delta, and sampling fluctuations
change the adaptive split sequence. In the homogeneous case, the small empirical
RCC cost reduction also comes with increased risk; the population methods tie.
None of these empirical success rates proves a finite-sample optimizer guarantee
for the original RCC algorithm.

## A sufficient assumption for optimizer-valid calibration

The setting also lets us separate ordinary conformal calibration from a valid
optimizer-level statement. Assume a partition `G_1,...,G_K` is fixed before
calibration and the residual `Y-m(x)` has the **same continuous distribution
F_j at every x in G_j**. The distribution may differ between groups. This
assumption holds in the constructed cases; it is additional structure, not a
consequence of restricted conditional coverage.

For each group let `n_j` be its calibration count. Set `q_j` to the `k_j`-th
smallest **signed** residual, choosing the smallest available rank such that

```text
P(Binomial(n_j, 1-alpha) >= k_j) <= beta/K.
```

If no finite rank is available, use `q_j=+infinity`. By the order-statistic
identity, the probability over calibration data that `F_j(q_j)<1-alpha` is at
most `beta/K`. A union bound gives probability at least `1-beta` that all
groups satisfy this inequality simultaneously.

On that event, **any data-dependent decision** satisfying
`m(x)+q_j <= 0` in group `j` has true violation probability at most alpha.
The residual law is identical throughout the group, so this statement holds
uniformly over its decisions. Simultaneous protection across groups permits
selection of the cheapest feasible group as well.

The included `simultaneous_tolerance` baseline implements this construction
with `alpha=0.10` and `beta=0.05`. Alpha is future operational risk; beta is
the probability of a bad calibration dataset. This is a fixed-partition
certificate under the stated residual-law assumption, not a theorem for the
adaptive RCC partition.

In the saved 200-run experiment it meets the 10% risk target in 187, 198, and
197 runs, respectively. Such a confidence statement permits calibration
failures: it is not a promise that every replication will meet the target.
For the homogeneous case the exact probability of a calibration failure is
0.04715. Observing at least 13 failures in 200 has probability about 0.20 even
at the conservative upper bound 0.05; this run does not contradict the guarantee.

The construction identifies two distinct requirements that ordinary local
conformal calibration does not supply: residual structure within groups and
control of calibration error simultaneously across candidate groups.

## Interpretation and remaining question

This supplies a transparent positive example for the original RCC computation:
it can discover a useful local correction without being given the true noise
boundary. Its success is governed by how the generated regions align with
residual regimes and by the minimum group mass. The additional statistical
structure needed for optimizer validity can also be stated and checked here.

It does not establish an advantage over a strong conditional-quantile model or
other methods that learn residual scale. The next methodological question is
whether RCC can identify useful residual structure more effectively, or with
less modeling effort, than those alternatives while supporting a valid
calibration argument. The current study exposes that question rather than
settling it.

The [learned conditional-quantile follow-up](LEARNED_MODEL.md) addresses model
quality directly. It learns the mean and tail behavior from full-domain noisy
data without supplied scales or regime boundaries. Across 100 seeds at 8,000
training rows, the models have low oracle-relative quantile loss, but the large
RCC benefit disappears: the favorable case's mean saving is about 0.06%, with
higher risk. The supplied-mean mechanism comparison above remains a separate
experiment.

## Reproduce

```bash
python -m pip install -e '.[transparent]'
python experiments/transparent/run.py --replications 200 --calibration-size 4000 \
  --output results/transparent/capacity_audit.json
python -m unittest tests.test_transparent_capacity
```

NumPy is sufficient for the population formulas and original RCC calculation;
SciPy supplies binomial quantiles for the tolerance baseline. No solver license
or ML framework is required. Add `--include-runs` to retain all individual run
diagnostics instead of only the compact summaries and population calculations.
