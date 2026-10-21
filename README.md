# Group-based conformalization

Companion code for Chapter 4, **Robustness in Constraint Learning with Group-Based
Conformalization**. Two applications share a group representation: the quadratic
benchmark (§4.5) and the student enrollment case study (§4.6).

This is an organization-only refactor of `student-enroll`. Existing formulas,
defaults, randomness, solver logic, return values, and known mistakes are retained.
Module boundaries, imports, repository paths, and runner main guards were reorganized.
`KNOWN-ISSUE[ISSUE-xx]` comments point to [findings and proposed fixes](docs/findings.md).
Proposed fixes in that document have **not** been implemented. Original comments
and docstrings can contain misleading claims; the findings guide takes precedence
as the description of their limitations.

## Start with the saved results

Open [the aggregation notebook](notebooks/thesis_results.ipynb) and run all cells.
It requires only NumPy, pandas, and a Jupyter Python kernel; no Gurobi, TensorFlow,
or PyTorch is needed. The default inputs are the saved CSVs in `results/enrollment/`.

```bash
python -m pip install -e '.[notebook]'
jupyter lab notebooks/thesis_results.ipynb
```

Table 4.3 is recomputed from 18,000 applicant-method rows and checked against the
saved comparison and thesis values. Tables 4.1–4.2 have aggregation code and labeled
reference values, but their raw quadratic CSVs were absent from the original folder.
The notebook reports that gap without substituting thesis values for observations.
If recovered, add those CSVs to `results/quadratic/` and rerun. See
[aggregation rules and limitations](docs/reproduction.md).

## Navigate by purpose

| Location | Purpose |
| --- | --- |
| `src/conformal_cl/groups.py` | Group state, calibration membership, splitting (§4.4) |
| `src/conformal_cl/quadratic/` | Synthetic instances, Keras models/training, optimization, group method (§4.5) |
| `src/conformal_cl/enrollment/` | PyTorch models/training, data split, merit optimization, group adaptation (§4.6) |
| `experiments/sec45_quadratic/` | Original quadratic sweep and sample entry point |
| `experiments/sec46_enrollment/` | Original enrollment training, comparison, and legacy sample |
| `notebooks/thesis_results.ipynb` | Thesis table aggregation, diagnostics, input hashes, CSV exports |
| `examples/illustrations.ipynb` | Original supplemental conceptual drawings, with a findings notice |
| `results/` | Working enrollment copies, new quadratic outputs, aggregated table exports |

[Thesis crosswalk](docs/thesis_map.md) · [Findings](docs/findings.md) ·
[Refactor policy](docs/refactor_policy.md) · [Data](data/README.md)

## Running the original experiments

Install the relevant optional dependencies, or use `environment.yml` as a starting
environment definition. The full ML/solver environment has not been validated here.
An appropriate Gurobi installation/license is required to solve the experiments.

```bash
python -m pip install -e '.[enrollment,quadratic]'
python experiments/sec46_enrollment/train.py
python experiments/sec46_enrollment/run.py
python experiments/sec45_quadratic/run.py
```

The default enrollment comparison uses theta=.17; individual solver function defaults
remain .6 as in the original. Training/experiment commands overwrite their working
output filenames under `results/enrollment/` or `results/quadratic/`. The original sibling `student-enroll` folder remains the historical record. Full runs can be expensive; aggregation does not launch
them. The historical enrollment `legacy_sample.py` still has its missing-argument
defect (ISSUE-09) and is retained for traceability, not recommended as a quick start.

Terminology: RCC denotes **restricted conditional coverage** in the chapter. Legacy
function names such as `run_rcc` are retained; original documents sometimes expand
the acronym differently. No project license, dataset redistribution permission, or
formal citation metadata is inferred by this refactor; settle these before publication.
