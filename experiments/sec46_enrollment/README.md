# §4.6 enrollment experiment

```bash
python experiments/sec46_enrollment/train.py
python experiments/sec46_enrollment/run.py
```

Training and solving require the PyTorch/scikit-learn/Gurobi dependencies. The
comparison entry point keeps theta=.17, alpha=.1, delta=.1, 64 splits, merit [0,25]
in thousands of dollars, and all 6,000 applicants. Solver function defaults remain
theta=.6. Original results/weights/scalers were copied into `results/enrollment/`
for these entry points. The original sibling `student-enroll` folder remains the
historical record. Commands overwrite working outputs; the aggregation notebook
reads `results/enrollment/` and needs neither models nor a solver.

`legacy_sample.py` preserves the original broken missing-`basic_model` call at theta=.6
(ISSUE-09). Formula signs, splits, and other findings are deliberately not fixed.
