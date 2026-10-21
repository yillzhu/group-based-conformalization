# Corrected implementation lineage

The checked-in files under `results/enrollment/` are immutable historical
artifacts. They were produced by the pre-correction implementation and remain the
inputs to `notebooks/thesis_results.ipynb`. They must not be presented as outputs
of the corrected code.

New training and experiment entry points write beneath `results/corrected/` so a
new run cannot silently overwrite the historical evidence. Corrected results must
carry their own environment details, split IDs, parameters, and input hashes before
they are compared with the historical tables.

## Implemented corrections

- Group children own independent boundary dictionaries; parent state is unchanged.
- Region-dependent extrema are recomputed after a split instead of applying an
  expression-agnostic cached-value adjustment.
- Exact one-based finite-sample conformal order statistics replace the legacy
  `numpy.quantile(rank / n)` conversion. Unavailable ranks trigger an explicit
  insufficient-calibration route.
- Insufficient children inherit the parent correction rather than silently using
  zero. This is an explicit fallback, not a newly established child-conditional
  guarantee.
- The canonical calibrated interval is `[lower - qhat, upper + qhat]`, including
  enrollment optimization constraints.
- Pending groups at a split cap are sent through the constrained fallback instead
  of being discarded.
- Quadratic RCC forwards `alpha` and `delta` and uses the same quantile network for
  solution selection and evaluation.
- Enrollment preprocessing is fitted on training rows only. Deterministic split IDs
  are saved and reused for RCC calibration.
- PyTorch model initialization is seeded, custom model/scaler paths are validated,
  and applicant comparisons join on `StudentID`.
- Non-optimal solver outcomes are no longer all labeled mathematical infeasibility.
- RCC builds a fresh solver model per region so predictor-embedding auxiliaries do
  not leak between regions, and recorded RCC time uses accumulated solver runtime
  rather than wall-clock construction/training time.
- Quadratic outputs now preserve the solver termination status for every method and
  the true-information solve instead of collapsing all missing solutions into a
  feasibility flag.
- Quadratic sweeps use grid-derived filenames, refuse accidental collisions, and
  validate completed parameter keys before resuming an interrupted run.
- A corrected-schema aggregator validates status/vector/flag consistency, duplicate
  keys, and timings before computing feasibility and recorded-time tables.

## Solver-backed validation completed for enrollment

The corrected enrollment pipeline was trained and run end to end on all 6,000
applicants with Gurobi 13.0.3. All 18,000 solves (basic, global conformal, and RCC)
reported an optimal solution. The corrected mean awards were $1,010.27, $1,057.94,
and $1,053.99 respectively. RCC used a mean 2.40 splits (maximum 22); 2,867 cases
used a certified region and 3,133 used the explicit quantile-constrained fallback.

An independent PyTorch evaluation of every selected RCC solution found no lower
endpoint violation larger than `1e-5`; the largest numerical residual was about
`1.34e-6`. The generated CSVs and manifests live under
`results/corrected/enrollment/` and are intentionally excluded from the historical
artifact lineage.

## Solver-backed validation completed for quadratic examples

The corrected quadratic sweep ran all 60 configurations (10 seeds, dimensions 1--3,
noise standard deviations 0.10 and 0.15, `P=10`, and `alpha=0.1`) using Python 3.12,
TensorFlow 2.18.0, Gurobi 13.0.3, and `gurobi-machinelearning` 1.7.0. The run produced
56 truly feasible instances and four verified truly infeasible instances.

Independent evaluation of every returned vector against its generating quadratic
found the following counts among the 56 truly feasible instances:

| Method | Returned | Truly feasible | Rate over all 56 | Rate when returned |
| --- | ---: | ---: | ---: | ---: |
| Standard | 54 | 43 | 76.79% | 79.63% |
| CQR | 54 | 54 | 96.43% | 100.00% |
| Wrapper | 54 | 50 | 89.29% | 92.59% |
| RCC | 54 | 54 | 96.43% | 100.00% |

The complete grid has no duplicate or missing keys; stored feasibility flags agree
with independently recomputed constraints; all timings are finite and nonnegative;
and the result and source hashes match the run manifest. Two independent smoke runs
also matched exactly in every non-timing output after replacing partial TensorFlow
seeding with unified deterministic Keras seeding.

A license-gated end-to-end test now exercises the complete quadratic smoke case and
asserts optimal structured statuses for all five solves. It is skipped in the
lightweight default test suite and enabled with `RUN_GUROBI_INTEGRATION=1` in a
configured solver environment.

During validation, dimension-3 RCC exposed unstable native matrix-multiplication
warnings in regional membership checks. Split hyperplanes are now unit-normalized,
geometry is explicitly checked for finite values, and membership uses a validated
row-wise contraction. The failing case passes with runtime warnings promoted to
errors after this correction.

Unexercised enrollment solver failure branches remain covered only by defensive
status labeling, not by forced integration cases.

## Methodological limitations not fixed by code

The enrollment interval model still targets predictions from a fitted classifier,
not observed latent enrollment probabilities. Applicant-specific construction of
calibration inputs and optimization over the predicted interval require a separate
theoretical justification before claiming true-probability or pointwise
optimization guarantees. These claims should remain qualified even if the software
passes every test.

The thesis/PDF, raw historical quadratic results, exact historical aggregation
policy, data redistribution permission, project license, and citation metadata are
not present. Those provenance issues require source material or an owner decision.
The new status-aware summarizer intentionally rejects old or ad hoc CSVs that lack
the corrected schema; it does not infer statuses or retrofit historical evidence.
