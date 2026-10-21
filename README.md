# Group-based conformalization

Companion code for Chapter 4, **Robustness in Constraint Learning with Group-Based
Conformalization**. Two applications share a group representation: the quadratic
benchmark (§4.5) and the student enrollment case study (§4.6).

This began as an organization-only refactor of `student-enroll`. The checked-in
results remain immutable historical artifacts, but the active implementation now
contains a tested correction pass for several confirmed defects. See the
[historical findings](docs/findings.md) and [correction status](docs/corrections.md).
Regenerated corrected results are a separate lineage and must not be represented as
the source of the historical tables.

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

## Running the corrected experiments

Install the relevant optional dependencies, or use `environment.yml` as a starting
environment definition. The enrollment pipeline has been validated end to end with
Gurobi 13.0.3; the TensorFlow-based quadratic pipeline has not. An appropriate
Gurobi installation/license is required to solve either experiment.

```bash
python -m pip install -e '.[enrollment,quadratic]'
python experiments/sec46_enrollment/train.py  # writes results/corrected/enrollment
python experiments/sec46_enrollment/run.py    # writes results/corrected/enrollment
python experiments/sec45_quadratic/run.py     # writes results/corrected/quadratic
```

Use `python experiments/sec45_quadratic/run.py --smoke` for one small quadratic
configuration before launching the full sweep. Both experiment runners accept
explicit output directories; the enrollment runner also accepts applicant limits,
theta, split-cap, and feasibility-tolerance overrides.

The default enrollment comparison uses theta=.17; lower-level solver APIs retain .6
for backward compatibility. Corrected commands write under `results/corrected/`;
checked-in historical
artifacts under `results/enrollment/` are not overwritten. Full runs can be expensive,
and the aggregation notebook does not launch them. The small enrollment sample entry
point is available as a smoke test after corrected models have been trained.

Terminology: RCC denotes **restricted conditional coverage** in the chapter. Legacy
function names such as `run_rcc` are retained; original documents sometimes expand
the acronym differently. No project license, dataset redistribution permission, or
formal citation metadata is inferred by this refactor; settle these before publication.
