# Parked research idea: optimization-guided conformal refitting

## Status

This is a research note, not an implemented or validated contribution. The idea
is intentionally parked on the `codex/optimization-guided-refitting` branch while
the theoretical and computational foundations of the current RCC method are
developed. Do not merge it into the main RCC manuscript merely because the initial
diagnostics are suggestive.

## Motivation

The workforce experiments exposed an information-budget issue. If a method uses
an additional labeled sample to design local conformal groups, a fair baseline can
instead use that sample to refit the predictive model and preserve an untouched
sample for final calibration. In the initial experiments, this refitted-global
baseline matched or beat RCC.

This observation suggests a different use for RCC geometry. The partition may
identify parts of the decision space where prediction quality matters to the
downstream optimization problem. Rather than treating the partition only as the
final uncertainty model, use it to guide data acquisition or model refitting.

## Tentative problem statement

Given a learned constraint, an objective, a labeling or simulation budget, and an
independent final calibration sample, choose where to acquire additional outcomes
so that the final optimized decision has a favorable cost--reliability tradeoff.

The relevant region should combine three properties:

1. competitive objective value;
2. proximity to the learned feasibility boundary; and
3. uncertainty, poor support, or systematic residual error.

A schematic decision-relevant set is

```text
A(epsilon, gamma) = {
    x : f(x) <= v_star + epsilon and abs(h_hat(x)) <= gamma
}.
```

## Candidate procedure

1. Fit an initial point or quantile constraint model on historical data.
2. Use optimization over the fitted model to locate decision-relevant boundary
   regions. This step should be distinguished carefully from inspecting final
   calibration responses.
3. Allocate a fixed labeling or simulation budget among those regions.
4. Refit the predictive model using the acquired outcomes, with any selection or
   importance weighting made explicit.
5. Apply conformal calibration using a fresh, untouched calibration sample.
6. Solve the final optimization problem and evaluate it on independent outcomes.

The RCC geometry is therefore an acquisition or experimental-design mechanism,
not necessarily the final coverage mechanism.

## Necessary comparisons

Any study must hold the labeling and simulation budgets fixed and compare against:

- uniform data acquisition followed by refitting;
- ordinary predictive-uncertainty sampling;
- decision-focused active learning;
- refitting on every available labeled observation;
- RCC used only for final localized calibration;
- fixed, prespecified group calibration;
- global conformalization after refitting; and
- an oracle acquisition policy when one can be simulated.

Comparing against a global model that is denied labeled observations used by the
proposed method is not acceptable.

## Possible theoretical questions

- Final split-conformal validity should survive arbitrary adaptive acquisition and
  refitting when the final calibration sample is fresh and exchangeable. State the
  exact target and conditioning carefully.
- Under what separation and regularity conditions can prediction errors outside a
  near-optimal feasibility boundary be shown not to affect the final decision?
- Can label complexity depend on the complexity of the decision-relevant boundary
  rather than that of the full decision domain?
- Can the acquisition rule guarantee feasibility, optimality regret, or a joint
  cost--reliability bound?
- How should importance weighting address a training sample deliberately shifted
  toward the learned boundary?
- Can cross-fitting recover data efficiency without contaminating final conformal
  calibration?

## Candidate operations-research setting

Simulation optimization is a natural test bed. A detailed queueing or service
system simulator supplies expensive noisy constraint labels, while a learned
surrogate makes repeated optimization inexpensive. The acquisition policy chooses
which staffing plans receive additional simulation replications. The principal
metric is reliable decision quality versus simulation budget, not global prediction
error or interval width.

## Closest literatures to investigate

- decision-focused learning and Smart Predict-then-Optimize;
- active learning for contextual optimization;
- active information acquisition for optimization;
- optimization of data-acquisition decisions;
- adaptive and sequential conformal calibration; and
- simulation optimization and optimal computing-budget allocation.

The initial literature check found closely related work, so novelty must not be
claimed without a systematic review. Particularly relevant starting points are:

- Liu, Grigas, Liu, and Shen, *Active Learning for Contextual Linear
  Optimization: A Margin-Based Approach*;
- Zheng, Waggoner, Liu, and Chen, *Active Information Acquisition for Linear
  Optimization*;
- Persak and Anjos, *OPO: Making Decision-Focused Data Acquisition Decisions*;
  and
- Yang, Liu, Li, and Li, *Sequential Conformal Calibration under Recursive
  Selection*.

## Questions to settle before implementation

1. Are new labels actively obtainable, or is the setting limited to a fixed
   historical dataset?
2. Is the contribution about choosing new design points, allocating repeated
   simulations, weighting existing data, or some combination?
3. What final statistical guarantee is required: predictive coverage, group-average
   feasibility, simultaneous feasibility, or feasibility of the selected optimizer?
4. Why should optimization-guided acquisition outperform refitting on all available
   data under the same total budget?
5. Is this a separate paper or a future-work direction arising from RCC?

Until these questions have convincing answers, this branch should remain parked.
