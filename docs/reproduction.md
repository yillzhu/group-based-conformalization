# Result aggregation and reproduction scope

Run `notebooks/thesis_results.ipynb` in order. It only reads CSVs and exports tables;
it does not train or solve. Exported tables are under `results/aggregated/`, together
with input hashes and reporting-policy metadata in `aggregation_manifest.json`.
Generated tables are ignored by Git; executed notebook outputs document the review.

## Table 4.3

Default input: `results/enrollment/enrollment_result_{basic,quantile,rcc}.csv`.
Validate unique StudentID, identical applicant IDs/features, status values, theta=.17,
and valid merit for optimal rows. Average optimal merit and multiply by 1,000.
Each method has 6,000 optimal rows in the supplied saved results. Compare results to
both the saved summary CSV and thesis rounded values. Hash every source CSV.
New experiment runs overwrite these working CSVs; the original sibling
`student-enroll` folder retains the historical versions.
Old/new or multiple settings must not be mixed silently.

## Tables 4.1–4.2

Default input: `results/quadratic/*.csv`, schema written by the original runner.
The notebook uses the full experimental key (seed, dim, noise_std, P, alpha) and
rejects duplicate keys, including overlapping batch files. Select settings explicitly
using `QUADRATIC_FILTERS`; default alpha=.1, P=10, dim=1–3, noise=.1/.15 matches
the checked-in runner, not a recovered historical aggregation recipe.

`opt_true` is a string-serialized vector such as `[0.1 -0.2]`. A finite vector of the
expected dimension is the available proxy for a true-feasible instance. Missing or
NaN vectors are excluded, matching the thesis instruction to exclude true-infeasible
instances, but old outputs cannot distinguish true infeasibility from solver failure.
Feasibility flags accept 0/1/False/True; legacy -1 and RCC [2] mean no returned
learned solution. Inconsistent flag/vector combinations fail validation.

The historical feasibility denominator is not recoverable without its aggregator.
The notebook therefore displays BOTH rates: feasible / all true-feasible instances,
and feasible / returned learned solutions among true-feasible instances. The chosen
`FEASIBILITY_DENOMINATOR` controls the Table 4.1 comparison/export; its default is
`all_true_feasible`, counting missing learned solutions as failures. Counts accompany
the rates so a different denominator cannot be hidden.

For Table 4.2, `TIMING_POPULATION` defaults to `all_true_feasible`; alternatives are
`returned_solutions` and `all_instances`. The original thesis aggregation filter for
timing is unknown. Method-specific `t_*` values are averaged by dimension. Their
measurement scopes differ (ISSUE-08): this produces a table of recorded times, not
a corrected solve-only benchmark. The notebook labels this and compares recorded
averages to the thesis references without claiming equivalence.

No quadratic input means no computed Tables 4.1–4.2: reference values remain visibly
labeled as thesis references. The notebook never fabricates rows to match them.
Recover the raw runs and exact inclusion rules before calling these tables reproduced.

## Verification limits

The original file/algorithm errors remain. Aggregation of stored numbers does not
establish statistical guarantees or prove current source regenerated those numbers.
Table 4.3 is supported by available applicant rows. Full training/solver execution and
exact historical quadratic reproduction have not been validated.
