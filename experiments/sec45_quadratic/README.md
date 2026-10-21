# §4.5 quadratic experiment

From the repository root, after an editable package install and solver/ML setup:

```bash
python experiments/sec45_quadratic/run.py --smoke
python experiments/sec45_quadratic/run.py
```

The corrected full sweep remains seeds 0–9; dimensions 1,2,3; noise standard
deviations .1,.15; P=10; alpha=.1. It writes beneath
`results/corrected/quadratic/`. The filename includes a stable hash of the complete
parameter grid, so different grids do not collide. Re-execution refuses to replace
an existing file unless `--overwrite` is explicit; use `--resume` to validate and
continue an interrupted compatible grid. Each completed row is appended durably,
and a manifest is written only after the requested grid is complete.

Custom grids are available through `--seeds`, `--dims`, `--noise-stds`,
`--ensemble-sizes`, and `--alphas`. `--output-file` selects an explicit CSV path.
The CSV carries separate structured statuses for the standard, CQR, wrapper, RCC,
and true-information solves. A non-optimal solver termination is not silently
reported as mathematical infeasibility.

Validate and aggregate one or more nonoverlapping corrected batches with:

```bash
python experiments/sec45_quadratic/summarize.py \
  results/corrected/quadratic/quadratic_*.csv
```

The command rejects duplicate parameter keys and inconsistent statuses, vectors,
feasibility flags, or timings. It writes corrected Table 4.1 and recorded-solver-time
Table 4.2 CSVs plus a provenance manifest. These outputs are a corrected lineage;
they do not reconstruct the missing historical inputs.

The historical raw CSVs for thesis Tables 4.1–4.2 are missing. New runs are not
automatically equivalent to those historical experiments. Use the result notebook
only for recovered legacy-schema CSVs with its documented filters. The historical
`legacy_sample.py` remains available as a compatibility entry point, but calls the
corrected RCC implementation.

The default unit tests remain license-free. In a configured Gurobi/TensorFlow
environment, opt in to the solver-backed smoke test with:

```bash
RUN_GUROBI_INTEGRATION=1 python -m unittest tests.test_quadratic_solver_integration
```
