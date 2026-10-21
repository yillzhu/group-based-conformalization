# §4.5 quadratic experiment

From repository root, after an editable package install and solver/ML setup:

```bash
python experiments/sec45_quadratic/run.py
```

The original sweep remains seeds 0–9; dimensions 1,2,3; noise standard deviations
.1,.15; P=10; alpha=.1. It writes `results/quadratic/seed_0_9.csv`. Re-execution
overwrites that filename's header before appending rows. The historical
`legacy_sample.py` runs the original one-instance RCC example with logging.

The historical raw CSVs for thesis Tables 4.1–4.2 are missing. New runs are not
automatically equivalent to those historical experiments. Use the result notebook
to aggregate recovered/new CSVs with documented filters; see ISSUE-08 for timing.
All original logic, including group bugs, remains unchanged.
