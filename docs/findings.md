# Findings retained in the refactored companion code

All findings below are **unfixed**. This is a guide to locating and correcting them
in a future change, not a claim that a scientific correction was applied.
Search for `KNOWN-ISSUE` in source to find the affected expressions.

The detailed review below was originally prepared against the `student-enroll`
folder. Its original-file line references remain historical references; each issue
now lists current module/symbol locations. Original docs remain in
the original sibling `student-enroll/docs/` folder. The source PDF is not copied into this repo.

| ID | Problem | Proposed fix / validation |
| --- | --- | --- |
| ISSUE-01 | Shared child boundary dictionaries | Copy parent bounds per child; test sibling and parent independence |
| ISSUE-02 | Lower conformal endpoint sign conflicts | Agree interval convention with thesis; update bound constraints and compare new results separately |
| ISSUE-03 | Cached extrema use wrong child/sign adjustments | Derive expression-specific updates, or recompute extrema; test against fresh solves |
| ISSUE-04 | Last-N vs random calibration and all-data scaler | Persist/reload split IDs and fit scaler on training only; preserve historical outputs |
| ISSUE-05 | Teacher targets and constructed applicant calibration do not justify stated true-probability guarantees | State target/distribution/assumptions; validate claims separately from classifier accuracy |
| ISSUE-06 | Pending groups lost at cap, count boundary mismatch, zero correction fallback | Explicit terminal policies and statuses; tests at split cap and N=10 |
| ISSUE-07 | Alpha/delta not forwarded; rank conversion/small groups | Forward settings and implement intended order statistic; test exact ranks and out-of-range cases |
| ISSUE-08 | RCC timing scope and model identity differ | Record separate phase times and network identity; aggregate comparable timing definitions |
| ISSUE-09 | Entry-point, status, comparison, environment and solver lifecycle defects | Fix independently; test custom paths, missing args, solver cleanup, keyed comparisons |
| ISSUE-10 | Notebook illustrations mix signs and synthetic content with guarantee claims | Align orientation and label conceptual examples; do not treat them as empirical coverage tests |

The new aggregation notebook is independent of the historical optimization code.
It uses StudentID joins/validation, explicit denominators, input hashes, and duplicate
checks for its own reporting. This does not change the original runner's logic.

Labels: **confirmed** means directly established from local source/artifacts or a
small isolated check; **review needed** means the intended mathematics, experiment
provenance, or solver behavior needs reconciliation. No training or full solver
experiments were run. No external literature/API validation was performed.

## Findings that should shape the reorganization

### ISSUE-01. Confirmed: child groups share and overwrite their boundary dictionary

**Current locations:** `src/conformal_cl/groups.py::Group.__init__` and `Group.split`. Package paths without a prefix are under `src/conformal_cl/`.


`src/group.py:62` and `:78` both construct `Group(self.bounds)` without copying.
Each child then assigns the same new hyperplane key with opposite signs. The
parent and both children consequently end up with the second child's cut. The
left child's calibration count can describe a region different from its final bounds.
The constructor also has a mutable dictionary default (`:7`).

An isolated NumPy-only reproduction using 200 points, a cut at zero, and explicitly
supplied extrema confirmed `parent.bounds is left.bounds is right.bounds` is true
and both children end with sign `+1`. This affects both applications. Fix and test
independent region state before presenting the algorithm as a faithful partition.

### ISSUE-02. Confirmed inconsistency: lower conformal endpoint sign

**Current locations:** `enrollment/optimization.py::optimize_merit_with_quantile_constraint`, `enrollment/group_conformal.py::run_enroll_rcc`; compare `enrollment/training.py`. Package paths without a prefix are under `src/conformal_cl/`.


The score is `max(y-upper, lower-y)` in both training pipelines. Enrollment training
evaluates `[lower-qhat, upper+qhat]` (`scripts/train_enroll_data.py:495`). But
`scripts/run_enroll.py::optimize_merit_with_quantile_constraint` imposes
`lower+qhat >= theta`, and `src/enroll_RCC.py` uses that same plus sign.

The thesis itself has both conventions: p. 91 gives `[lower-qhat, upper+qhat]`,
whereas p. 92 uses a plus sign in the enrollment lower-bound constraint. Section
4.4.2, p. 82, also prints plus signs on both interval endpoints. The issue is thus
not just a stale README or code typo; reconcile the notation and intended method.

For the saved `qhat=-0.0025978982`, `lower+qhat` and `lower-qhat` differ by about
0.005196 on the probability scale. The historical Table 4.3 must not be silently
relabeled as output of a corrected-sign implementation.

### ISSUE-03. Confirmed: cached extrema adjustments are not consistently valid

**Current locations:** `src/conformal_cl/groups.py::Group.split`; extrema objectives in both `group_conformal.py` adapters. Package paths without a prefix are under `src/conformal_cl/`.


`src/group.py:84`, `:88`, and `:92` use `G_l.qhat` while updating the right child.
More generally all cached objective values are shifted by `-parent.qhat+child.qhat`.
That matches an expression with `+qhat`, but the quadratic lower extremum uses
`lower-qhat`, and enrollment extrema use `theta-quantile-qhat`. Their adjustment
must have the corresponding opposite sign. First disable cache reuse or derive
the update from an explicit bound/expression type; test against fresh extrema.

### ISSUE-04. Confirmed: calibration rows differ between enrollment training and RCC

**Current locations:** `enrollment/training.py::train_main`, `enrollment/group_conformal.py::load_models_and_data`, `experiments/sec46_enrollment/run.py`. Package paths without a prefix are under `src/conformal_cl/`.


`scripts/train_enroll_data.py::split_train_test_cal` uses random train/test/calibration
splits with `random_state=42`. Both RCC loaders (`src/enroll_RCC.py` and
`scripts/run_enroll.py`) instead select `features[-2000:]`. These are not the same
held-out rows, and the latter selection is not guaranteed disjoint from training.
The basic scaler is also fit on the entire dataset before splitting (`:221`).

Save and reload row IDs for all splits; fit preprocessing on training rows only.
For historical reproduction, record the existing behavior separately rather than
retroactively claiming training-only preprocessing was used.

### ISSUE-05. Review needed: what enrollment calibration actually covers

**Current locations:** `enrollment/training.py::train_quantile`, `enrollment/group_conformal.py::run_enroll_rcc`. Package paths without a prefix are under `src/conformal_cl/`.


Quantile targets are deterministic predictions from the fitted basic classifier,
not observed true enrollment probabilities. `run_enroll_rcc` replaces every
calibration row's SAT/GPA with the current applicant's values, keeps its merit,
and recomputes teacher predictions. The supplied `cal_probs` is unused, and
`qhat_global` is also unused by that function.

Coverage of teacher predictions is not established evidence of coverage of true
enrollment probabilities. Nor does standard marginal coverage alone establish a
guarantee at a merit value selected by optimization. Section 4.3, pp. 78-79,
explicitly distinguishes neighborhood/group coverage from pointwise optimal-decision
coverage. The adaptive partition, chosen group family, and constructed calibration
distribution need an explicit assumptions discussion before stronger claims.

The thesis itself asserts probability calibration from strong predictive metrics
(p. 90). Saved accuracy/MSE/R-squared values do not by themselves demonstrate
probability calibration; add relevant validation or weaken that statement.

### ISSUE-06. Confirmed: split limits and insufficient-data policy need explicit handling

**Current locations:** `groups.py::Group.conformalize`, both `group_conformal.py` loops and final queues. Package paths without a prefix are under `src/conformal_cl/`.


`src/RCC.py` breaks when `count == max_split`; `src/enroll_RCC.py` stops its loop
when `num_splits` reaches the limit. Both then optimize only the certified and
insufficient-data queues, leaving pending candidate groups unprocessed. This can
lose candidates and be reported as infeasibility without proving infeasibility.

Group calibration requires `Nhat > 10`, while queue routing accepts `Nhat >= 10`.
At exactly 10 observations, a group can be queued without receiving its correction.
Insufficient children start with `qhat=0` rather than inheriting a parent/global
value. This is not automatically a conservative fallback.

Decide the intended fallback, handle every terminal region explicitly, and report
`split_limit`, `insufficient_calibration`, and solver failures independently from
true infeasibility. Algorithm 1's insufficient-data route and its final constrained
solve need to be reconciled with this behavior.

### ISSUE-07. Confirmed/review needed: calibration settings and order statistics

**Current locations:** `groups.py::Group.conformalize`, both `calibration.py` modules, `quadratic/models.py::build_quantile_model`, `quadratic/group_conformal.py::run_rcc`. Package paths without a prefix are under `src/conformal_cl/`.


`src/RCC.py:28` and its `G.split(...)` call omit the supplied alpha/delta, so group
calibration uses defaults regardless of `run_rcc` arguments. In
`src/quadratic_sample.py::build_quantile_model`, alpha is accepted but not bound
into the compiled loss, whose default is 0.1. Default thesis settings mask this.

`Group.conformalize` uses `ceil((N+1)*(1-alpha+1/n_cal))/N` as a NumPy quantile
level. This can exceed one for small groups/settings. Moreover, NumPy's
`quantile(..., method='higher')` at `k/N` does not generally select the k-th smallest
sample, which is the rank stated on thesis p. 82. The global calibration routines
use the same rank-to-quantile pattern. Verify the exact intended finite-sample order
statistic and implement explicit small-sample behavior rather than clipping silently.

### ISSUE-08. Confirmed mismatch: quadratic timing does not match the thesis definition

**Current locations:** `quadratic/group_conformal.py::run_rcc`, `quadratic/benchmark.py::run_one`. Package paths without a prefix are under `src/conformal_cl/`.


Thesis p. 87 says Table 4.2 excludes training and embedding construction time.
For standard/CQR/wrapper, `optimize_compare` times calls to `model.optimize()`.
For RCC, `run_rcc` starts the timer after training but before building/embedding
models, partitioning, the extra true-feasibility solve, and final regional solves.
These times are not directly comparable under the stated definition.

Also, `run_one` trains a quantile model, then `RCC.run_rcc` trains another one.
The reported `q_rcc` is evaluated with the first network, not necessarily the one
used to choose the RCC solution. Decide what cross-evaluation is intended and
record model identity, solve-only time, partition overhead, embedding, and training
separately. Raw quadratic results and table aggregation code are absent here.

### ISSUE-09. Confirmed entry-point/portability defects; solver lifecycle needs checking

**Current locations:** `experiments/sec46_enrollment/legacy_sample.py`, `enrollment/training.py::train_quantile`, `groups.py::update_groups`, solver loops, and the enrollment runner. Package paths without a prefix are under `src/conformal_cl/`.


- `src/enroll_RCC.py` standalone entry calls `run_enroll_rcc_batch` without its
  required `basic_model` argument. The runner in `scripts/run_enroll.py` does pass it.
- `train_quantile(basic_model_path=...)` leaves `basic_scaler_path` undefined when
  a custom model path is supplied (`scripts/train_enroll_data.py:364`, `:382`).
- `scripts/run_quadratic_numerical.py` imports a module in sibling `src/` without
  configuring the import path; execution depends on an external PYTHONPATH/setup.
  Its output directory is relative to the process working directory, and import
  immediately starts the experiment and overwrites its output CSV header.
- `environment.yml` omits used dependencies including torch, pandas, matplotlib,
  and tqdm, and includes an absolute environment prefix for another path/name.
- Enrollment initializes PyTorch networks without an explicit torch seed. The split
  random state alone is not a reproducible training seed.
- `Group.update_groups` refers to nonexistent/currently incompatible `split`, `lb`,
  and `ub` attributes and constructor semantics. It has no callers in this checkout.
- **Review needed:** group solvers remove `model.getConstrs()` and repeatedly embed
  neural networks. Verify lifecycle of auxiliary variables and general constraints;
  removing linear constraints alone is not a demonstrated clean predictor removal.
- Non-optimal solver statuses are often labeled `Infeasible`. Time limits/errors
  must not be interpreted as mathematical infeasibility.
- Enrollment RCC comparison uses filtered arrays matched to the first N baseline
  rows instead of a StudentID join. Current all-feasible outputs hide this issue.

## Markdown and notebook corrections

### `docs/RCC_approach_summary.md`

| Location | Finding | Action |
| --- | --- | --- |
| Title and opening | "Risk-Controlling Conformal" does not match the chapter's restricted conditional coverage terminology | Rename and define the coverage property separately from the group algorithm |
| Key Concept / Standard RCC | Uses `lower+qhat` for the lower endpoint; code uses `lower-qhat` | Establish a single interval convention and annotate historical deviations |
| Algorithm Flow / Step A, lines 34-39 | Says a nonnegative **maximum** of `lower+qhat-theta` proves the entire group feasible | Incorrect quantifier: whole-group feasibility requires a nonnegative **minimum**, or max of `theta-lower_bound` nonpositive |
| Steps A-C / Group Class Structure / Gurobi Integration | Extrema do not describe the current enrollment implementation, which maximizes `theta-lower-qhat` and minimizes two transformed expressions | Rewrite from an explicit composite-constraint contract |
| Splitting Strategy, lines 130-131 | Associates `d=-1` with `a*x<=b` and `d=1` with `a*x>=b` | Reverse these labels: actual membership is `d*a*x<=d*b`, so `+1` means `<=` |
| Conformalization per Group | Pseudocode uses ordinary `(1-alpha)` quantile and an informal sufficiency threshold | Include group rank with `1/n_cal`, precise sufficiency rule, and small-sample policy after review |
| Efficiency / pruning | Says small groups are discarded | Current code routes them to a constrained fallback queue |
| Results Interpretation | Describes 10 zero-merit applicants at theta=.6 without tying them to saved evidence | Label as an unverified historical smoke-run claim; current saved output has 6,000 applicants at .17, and current standalone entry is broken |
| "10% of training data" | Last 2,000 are 10% of the full 20,000-row history, not 10% of the 14,000-row training split | Correct wording and explain split mismatch |
| Guarantees throughout | Unqualified whole-group/true-probability/decision guarantees | State assumptions, target, coverage population, and Section 4.3 limitations |

### `docs/model_approach_summary.md`

| Location | Finding | Action |
| --- | --- | --- |
| Optimization results, lines 65-86 | theta=.6 and averages $1,473.80/$1,506.19 describe a different setting from current source, saved CSVs, and Table 4.3 | Preserve as historical only if provenance can be recovered; current thesis setting is .17 with $1,021.17/$1,054.94 |
| Training Results | Accuracy, coverage .90525, width .030603757, and negative qhat approximately match saved metric files | Retain with artifact/run reference; these values are not all wrong |
| Solution line 17 | `logit(output)>=logit(theta)` is ambiguous after sigmoid removal | Define raw network output `z`; impose `z>=log(theta/(1-theta))` |
| Quantile constraint, line 73 | `lower+qhat` conflicts with evaluated interval | Reconcile with ISSUE-02 and thesis rather than editing only prose |
| Quantile Model Approach / Advantages | "true probability quantiles" and model-uncertainty/conservative claims exceed evidence from a deterministic teacher model | Describe intervals targeting teacher outputs, and limit claims accordingly |
| Gurobi support statement | Absolute software capability statement lacks version scope | State the versions and representation used in these experiments; verify capabilities when preparing the environment |

### Markdown inside `scripts/figure.ipynb`

This notebook is illustrative, not a result aggregator or trace of Algorithm 1.
Figure 2 uses hand-authored partitions and offsets. Figure 4 uses constructed
nonnegative residuals, not scores calculated from fitted intervals and held-out
observations; actual CQR scores can be negative, as the saved qhat demonstrates.

Figures 1-2 use the `g<=0` orientation: upper endpoint below zero is feasible.
Figure 3 and summary instead describe lower endpoint above zero as feasible and
upper below zero as infeasible, which reverses that orientation. Align them or
explicitly separate probability-threshold illustrations. Synthetic histograms do not
demonstrate a coverage guarantee. The notebook's figure numbers are not the
chapter's Figure 4.1/Figure 4.2; rename exports as supplemental illustrations.

## Additional prose file: `docs/enrollment_models_experiment_description.txt`

Although not Markdown, this file should also be corrected:

- Section 1 calls these "mixed-integer-free convex optimization problems." The
  learned ReLU embeddings are not justified as convex formulations; describing
  them as Gurobi models with embedded ReLU networks is accurate without that claim.
- Sections 4.3-4.4 give `[lower-qhat, upper+qhat]` and then use `lower+qhat`.
- Claims that the constraint ensures true-probability feasibility with high probability
  exceed what the teacher-target metrics and marginal coverage establish.
- Sections 5.3-5.5 should describe zero default correction, candidate loss at the split
  limit, and intended fallback only after those implementation issues are settled.
- Its theta=.17 and three average merit figures agree with current Table 4.3 artifacts.
  Preserve this validated portion; the file need not be discarded wholesale.

## Thesis reconciliation items

These should be brought back to the chapter as well as fixed in companion docs:

1. Lower-endpoint signs in Section 4.4.2 and enrollment constraint on p. 92 versus
   the interval on p. 91.
2. Section 4.3's explicit neighborhood-coverage limitation versus stronger pointwise
   feasibility language later in Section 4.4.3 and the case study.
3. Section 4.5.1 says "four" formulations but enumerates five including RCC.
4. Table 4.2 solve-only timing definition versus actual RCC timing scope.
5. Probability-calibration assertions require evidence beyond accuracy/MSE/R-squared.
6. The intended guarantee for adaptively chosen partitions and applicant-specific
   constructed calibration inputs needs justification; this review has not established it.

## What is already supported by the saved artifacts

`results/student_enroll/enrollment_result_comparison.csv` records 6,000 optimal
solutions, zero infeasible solutions, and theta=.17 for all three methods:

| Method | Saved average merit, USD | Thesis Table 4.3 |
| --- | ---: | ---: |
| Basic | 1021.1671243308195 | 1,021.17 |
| Quantile | 1054.9396171813096 | 1,054.94 |
| RCC | 1238.306550062892 | 1,238.31 |

The three per-applicant CSVs each contain 6,000 rows. The saved basic metric file
reports test accuracy .95575. The quantile metric file reports test coverage .90525,
mean width .030603757, and qhat -.0025978982. These support statements about the
stored run, not proof that current source exactly regenerates it or that all claimed
statistical guarantees hold.

No `results/quadratic_functions/` output, Table 4.1/4.2 aggregator, or generator of
the thesis's absolute-value/polynomial illustrations was found. The current quadratic
runner configures seeds 0-9, dimensions 1-3, noise .1/.15, P=10, alpha=.1, but the
exact provenance of the reported thesis aggregates cannot be inferred from that alone.

## Suggested priority

1. Preserve historical source, docs, and outputs; correct misleading current guidance.
2. Mechanical package/experiment organization in the companion plan.
3. Independent group state, interval signs, cached extrema, and terminal queue policies.
4. Calibration splits/targets/ranks, scientific claims, and consistent benchmark timing.
5. Recover quadratic evidence, run focused validation, and document historical versus
   corrected results before a public reproducibility claim.
