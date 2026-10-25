# Enrollment example: reassessment and next experiment

## What the current computation establishes

For each of 6,000 synthetic applicants, the experiment minimizes merit aid subject
to a learned lower-bound constraint. The corrected solver run is internally
consistent: all solves were optimal and independent network evaluation confirms
the constraints at the returned decisions. RCC changes the global conformal
correction locally, but its mean saving is only $3.94 per applicant.

The public JANOS paper and replication repository describe both enrollment files as
randomly generated solver-demo records. They do not provide the response-generating
function. The example is therefore neither evidence about a real university nor a
benchmark with a recoverable oracle response probability.

## Three distinct statistical objects

The experiment should keep the following objects separate:

1. The binary enrollment label in the synthetic historical data.
2. The fitted classifier score, trained against that label.
3. The quantile interval, trained against deterministic predictions from the fitted
   classifier rather than against an unknown true response probability.

The reported conformal coverage concerns object 2. Applicant-specific RCC then
holds SAT and GPA fixed, varies merit over calibration-row merit values, and
recomputes both networks. This reveals local approximation geometry between the
two fitted networks. It is not an exchangeable sample of repeated applicant
responses.

## Diagnostics

Run the solver-free audit after training and optimization:

```bash
python experiments/sec46_enrollment/audit.py \
  --output results/corrected/enrollment/audit.json
```

The audit reports held-out Brier score, log loss, AUC, equal-width reliability bins,
merit monotonicity for both networks, and paired RCC/global decision differences.
For the current artifacts, the point model discriminates extremely well but its
score is not calibrated as a literal probability. Its response is nondecreasing
over the tested merit grid, whereas the lower quantile endpoint decreases somewhere
for 2,774 of 6,000 applicants.

## Defensible next experiment

A revised synthetic benchmark needs to publish its data-generating mechanism. It
should generate Bernoulli outcomes from a known monotone response probability
`p(SAT, GPA, merit)`, use disjoint samples for fitting, partition design, conformal
calibration, and evaluation, and retain the oracle probability for evaluating
optimized decisions. Only that design can distinguish these questions:

- Does a returned decision meet the oracle probability requirement?
- Does RCC improve cost at a fixed oracle violation rate?
- Is any improvement due to localized calibration rather than extra labeled data?

The current JANOS data can remain as a software-reproduction case, but should not be
the main statistical validation of RCC. Probability calibration and monotonic
architectures can improve its surrogate, but cannot reconstruct an unreported
oracle after the fact.

## Known-oracle follow-up and negative control

`oracle_benchmark.py` implements the proposed follow-up as a cohort-level tuition
experiment. It publishes a monotone Bernoulli response mechanism, observes repeated
cohort enrollment rates, trains a merit-monotone point model and interval model,
uses independent conformal calibration data, and evaluates optimized decisions
against the oracle. Cohort size is supplied to every predictive method because it
is known and determines sampling variance.

```bash
python experiments/sec46_enrollment/oracle_benchmark.py
python experiments/sec46_enrollment/oracle_stress.py --runs 30 --jobs 4
```

The 30-seed equal-information stress test is negative. Global CQR averaged
$9,718.05 in cohort-weighted merit per student and localized calibration averaged
$9,730.94. Localization was cheaper in 10 runs, costlier in 19, and tied in one;
its mean "saving" was -$12.89, with a range from -$154.32 to $211.02. Global CQR
met the oracle 90% predictive criterion in all 360 segment decisions. Localization
missed once (oracle success probability 0.875).

An intentionally incomplete baseline that omitted known cohort size made
localization appear to save about $484 per student. Giving cohort size to the
quantile model eliminated that apparent advantage. The omitted-feature result is
therefore a diagnostic ablation, not evidence for RCC. This benchmark currently
provides no evidence that localization improves the tuition decision when all
methods receive the same information.

Primary sources:

- Bergman et al., *JANOS: An Integrated Predictive and Prescriptive Modeling
  Framework*, https://arxiv.org/abs/1911.09461
- JANOS replication repository, https://github.com/INFORMSJoC/2020.1023
