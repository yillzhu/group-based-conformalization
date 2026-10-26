# Learning conditional quantiles in the capacity example

This follow-up addresses criterion 3: a good ML model learned from full-domain
observations. The original study supplied an exact mean and omitted residual
scale from the unscaled interval model. Here the mean, the locations where
uncertainty changes, and both conditional quantiles are learned from noisy data.
The three mechanisms, continuous domain `[0,4]`, and 10% operational risk target
remain those in [the capacity study](README.md).

## Model and independent data splits

The learner receives only capacity/response pairs `(x,Y)`. It receives neither
the noise scales nor their boundaries, the number of regimes, or the true mean
coefficients. It uses the following model:

1. Fit an affine mean by least squares on training observations.
2. Fit regression trees to squared training residuals, allowing 1, 2, 3, 4, 6,
   8, or 12 leaves. Each leaf contains at least `max(40, ceil(n_train/100))` rows.
3. Estimate the 5th and 95th residual quantiles within each learned leaf.
4. Interpolate across the gap between neighboring training inputs at each learned
   boundary. The resulting quantiles are continuous, piecewise affine, and
   globally noncrossing, and preserve leaf offsets at the observed training inputs.
5. Select tree size by lower-plus-upper pinball loss on independent validation
   observations. Calibration and oracle evaluation quantities do not select models.

The affine mean class is correctly specified for this simple mechanism. The
residual tree learns uncertainty variation without known-scale features. This is
an adequate model for the constructed problem, not a universally optimal learner.

Each replication has independent training, validation, calibration, and diagnostic
streams. Training sizes are 2,000, 8,000, and 32,000; validation and calibration
each contain 4,000 observations. All inputs are iid uniform over the entire domain.
Validation, calibration, and diagnostic datasets are held fixed across training
sizes within each replication. All decision rules share the same fitted model.

The raw rule imposes the learned upper quantile directly. Global CQR and RCC
use identical calibration observations and scores `max(Y-upper, lower-Y)`.
Global calibration uses the same adjusted rank as RCC's root. RCC retains the
original adaptive partition, count rule, and unresolved constraint fallback.
Spline extrema and optimization are continuous and exact. A parity test compares
this reduction with the existing polygon-based continuous RCC implementation.

## Measuring whether the predictor is good

Independent samples of 20,000 covariates each measure full-domain quality and
quality near the optimum, on the fixed interval `[1,1.2]`. The known conditional
law supplies exact expected pinball loss and coverage at each covariate; only
integration over covariates is sampled. Risk at selected decisions is analytic.

The declared diagnostic gate requires expected lower-plus-upper pinball loss
within 10% of oracle loss and mean absolute error in the upper quantile's
conditional CDF value at most 0.025 (2.5 percentage points). These tolerances
are diagnostics, not a general definition of model quality. No run is removed
or retrained because it fails a gate or produces an unfavorable decision.

At 8,000 training rows every model passes both gates, globally and near the
optimum. Actual average errors are much smaller than the tolerances:

| Case | Excess pinball loss, full domain | Excess pinball loss, near optimum | Raw interval coverage, full domain |
| --- | ---: | ---: | ---: |
| Homogeneous | 0.047% | 0.052% | 89.95% |
| Aligned | 0.127% | 0.237% | 89.90% |
| Interior quiet region | 0.340% | 0.127% | 89.90% |

The interval targets 90% coverage; its upper endpoint targets the 95th percentile.
At selected global decisions the average absolute upper-quantile errors are
0.00055, 0.00088, and 0.00117, respectively. Thus the model-quality assessment
includes the optimizer's neighborhood and its selected point, not only pooled
coverage elsewhere. The artifact retains worst quality metrics and learned
tree-size frequencies as well as averages.

## Decision results

The 8,000-row comparison has 100 replications per case:

| Case | Global mean cost | Global mean risk | RCC mean cost | RCC mean risk |
| --- | ---: | ---: | ---: | ---: |
| Homogeneous | 1.090057 | 4.97% | 1.089668 | 5.17% |
| Aligned | 1.090206 | 4.90% | 1.089561 | 5.22% |
| Interior quiet region | 1.090669 | 4.67% | 1.089832 | 5.08% |

Both methods meet the 10% target in all 300 primary runs. In the aligned case,
RCC is cheaper in 70 runs, costlier in 29, and tied in one. Its average saving
is 0.000645, or 0.059% of global cost. This is a small positive average saving,
accompanied by higher risk, rather than the original 18.8% improvement in both
cost and risk.

The raw learned-quantile rule already selects mean capacities of 1.08993,
1.08986, and 1.09022, meeting the 10% target in all 300 primary runs. The model
has absorbed almost all of the structure previously recovered by RCC.

The primary decisions lie in the quiet range where exact risk is `r(x)=5.5-5x`.
The small cost reductions therefore correspond directly to additional risk;
they do not demonstrate an improved cost-risk frontier. The 10% chance oracle
chooses 1.08, while the true conditional 95th-percentile surrogate chooses 1.09.

## Sensitivity to training size

The complete study has 100 replications per case and size: 900 fits. Results for
the aligned case are:

| Training rows | Both quality gates passed | Global mean cost | RCC mean cost | RCC relative saving |
| ---: | ---: | ---: | ---: | ---: |
| 2,000 | 98/100 | 1.090830 | 1.089832 | 0.092% |
| 8,000 | 100/100 | 1.090206 | 1.089561 | 0.059% |
| 32,000 | 100/100 | 1.090122 | 1.089605 | 0.047% |

All cases pass both gates in every run at 8,000 and 32,000 rows. At 2,000 rows,
pass counts are 100, 98, and 94 for the homogeneous, aligned, and interior cases.
One interior-case RCC decision at this smallest size has 10.18% risk; global CQR
stays below 10% in all corresponding runs. These weaker-model runs remain in
the saved results.

Calibration size stays at 4,000, so local calibration noise and adaptive selection
persist as training size increases. These experiments do not imply monotonic
decay of RCC's effect, or establish an optimizer-level coverage theorem.

## Relation to the four criteria

Criterion 3 is now addressed for this transparent setting: an actual learned
conditional-quantile model has full training support and low oracle-relative loss
both globally and near the optimizer. It is not supplied the true noise scale.

The large RCC advantage does not survive this test. There is a small average
cost saving with higher risk; a stronger claim would require a comparison at
matched realized risk and a practically meaningful magnitude. This supports
the interpretation that RCC repairs uncertainty structure omitted by a base
model, without yet establishing superiority over learning conditional quantiles
well. Criterion 1 remains a stylized capacity problem, and criterion 4 remains
unresolved against strong uncertainty modeling.

## Reproduce

```bash
python -m pip install -e '.[transparent]'
python experiments/transparent/run_learned.py \
  --training-sizes 2000 8000 32000 --replications 100 \
  --output results/transparent/learned_quantile_audit.json
python -m unittest tests.test_transparent_learned
```

Add `--include-runs` to save each model's coefficients and knots, validation
losses, diagnostics, and decisions. The default artifact contains compact
summaries, all declared settings, and library versions.
