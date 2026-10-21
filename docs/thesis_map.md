# Thesis crosswalk and source migration

| Thesis part | New location | Status |
| --- | --- | --- |
| §4.1.1 / Figure 4.1 absolute-value example | No generator recovered | Missing |
| §4.1.2 / Figure 4.2 polynomial wrapper example | No generator recovered | Missing; distinct from neural-network wrapper benchmark |
| §4.2–4.3 interval and coverage interpretation | Calibration and optimization modules; `findings.md` | Theory is not executable evidence of pointwise guarantees |
| §4.4.1 unconstrained group min-max algorithm | No complete implementation identified | Theoretical material only in this checkout |
| §4.4.2 group calibration | `src/conformal_cl/groups.py` | Historical implementation, issues retained |
| §4.4.3 Algorithm 1 | `quadratic/group_conformal.py`, `enrollment/group_conformal.py` | Separate historical adapters, not a rewritten shared engine |
| §4.5.1 generation/training/optimization | `src/conformal_cl/quadratic/` | Original methods and parameters |
| §4.5.2 Tables 4.1–4.2 | `notebooks/thesis_results.ipynb` | New aggregation; raw historical CSVs missing |
| §4.6.1–4.6.2 training and merit solves | `src/conformal_cl/enrollment/` | Teacher-probability targets and original signs retained |
| §4.6.3 Table 4.3 | `notebooks/thesis_results.ipynb` | Re-aggregates preserved 6,000-applicant results per method |
| Supplemental conceptual drawings | `examples/illustrations.ipynb` | Original cells retained; not thesis Figures 4.1–4.2 |

## Every moved function/class

Paths below are relative to repository root. Functions are not renamed so callers,
historical documentation, and saved model architectures stay easy to cross-reference.

| Original symbol | New module |
| --- | --- |
| `src/group.py::Group` | [`src/conformal_cl/groups.py`](../src/conformal_cl/groups.py) |
| `src/group.py::update_groups` | [`src/conformal_cl/groups.py`](../src/conformal_cl/groups.py) |
| `src/quadratic_sample.py::generate_quadratic` | [`src/conformal_cl/quadratic/problem.py`](../src/conformal_cl/quadratic/problem.py) |
| `src/quadratic_sample.py::generate_samples` | [`src/conformal_cl/quadratic/problem.py`](../src/conformal_cl/quadratic/problem.py) |
| `src/quadratic_sample.py::is_feasible` | [`src/conformal_cl/quadratic/problem.py`](../src/conformal_cl/quadratic/problem.py) |
| `src/quadratic_sample.py::build_model` | [`src/conformal_cl/quadratic/models.py`](../src/conformal_cl/quadratic/models.py) |
| `src/quadratic_sample.py::interval_score_loss` | [`src/conformal_cl/quadratic/models.py`](../src/conformal_cl/quadratic/models.py) |
| `src/quadratic_sample.py::build_quantile_model` | [`src/conformal_cl/quadratic/models.py`](../src/conformal_cl/quadratic/models.py) |
| `src/quadratic_sample.py::train_main` | [`src/conformal_cl/quadratic/training.py`](../src/conformal_cl/quadratic/training.py) |
| `src/quadratic_sample.py::train_wrapper` | [`src/conformal_cl/quadratic/training.py`](../src/conformal_cl/quadratic/training.py) |
| `src/quadratic_sample.py::train_quantiles` | [`src/conformal_cl/quadratic/training.py`](../src/conformal_cl/quadratic/training.py) |
| `src/quadratic_sample.py::conformalizing` | [`src/conformal_cl/quadratic/calibration.py`](../src/conformal_cl/quadratic/calibration.py) |
| `src/quadratic_numerical.py::optimize_compare` | [`src/conformal_cl/quadratic/optimization.py`](../src/conformal_cl/quadratic/optimization.py) |
| `src/quadratic_numerical.py::is_feasible` | [`src/conformal_cl/quadratic/optimization.py`](../src/conformal_cl/quadratic/optimization.py) |
| `src/RCC.py::run_rcc` | [`src/conformal_cl/quadratic/group_conformal.py`](../src/conformal_cl/quadratic/group_conformal.py) |
| `src/quadratic_numerical.py::run_one` | [`src/conformal_cl/quadratic/benchmark.py`](../src/conformal_cl/quadratic/benchmark.py) |
| `src/quadratic_numerical.py::run_many` | [`src/conformal_cl/quadratic/benchmark.py`](../src/conformal_cl/quadratic/benchmark.py) |
| `scripts/train_enroll_data.py::create_enrollment_net` | [`src/conformal_cl/enrollment/models.py`](../src/conformal_cl/enrollment/models.py) |
| `scripts/train_enroll_data.py::create_enrollment_net_for_gurobi` | [`src/conformal_cl/enrollment/models.py`](../src/conformal_cl/enrollment/models.py) |
| `scripts/train_enroll_data.py::inverse_sigmoid` | [`src/conformal_cl/enrollment/models.py`](../src/conformal_cl/enrollment/models.py) |
| `scripts/train_enroll_data.py::create_quantile_net` | [`src/conformal_cl/enrollment/models.py`](../src/conformal_cl/enrollment/models.py) |
| `scripts/train_enroll_data.py::interval_score_loss` | [`src/conformal_cl/enrollment/models.py`](../src/conformal_cl/enrollment/models.py) |
| `scripts/train_enroll_data.py::split_train_test_cal` | [`src/conformal_cl/enrollment/data.py`](../src/conformal_cl/enrollment/data.py) |
| `scripts/train_enroll_data.py::conformalizing` | [`src/conformal_cl/enrollment/calibration.py`](../src/conformal_cl/enrollment/calibration.py) |
| `scripts/train_enroll_data.py::train_main` | [`src/conformal_cl/enrollment/training.py`](../src/conformal_cl/enrollment/training.py) |
| `scripts/train_enroll_data.py::train_quantile` | [`src/conformal_cl/enrollment/training.py`](../src/conformal_cl/enrollment/training.py) |
| `scripts/run_enroll.py::load_and_prepare_data` | [`src/conformal_cl/enrollment/optimization.py`](../src/conformal_cl/enrollment/optimization.py) |
| `scripts/run_enroll.py::optimize_merit_with_enrollment_constraint` | [`src/conformal_cl/enrollment/optimization.py`](../src/conformal_cl/enrollment/optimization.py) |
| `scripts/run_enroll.py::optimize_merit_with_quantile_constraint` | [`src/conformal_cl/enrollment/optimization.py`](../src/conformal_cl/enrollment/optimization.py) |
| `src/enroll_RCC.py::load_models_and_data` | [`src/conformal_cl/enrollment/group_conformal.py`](../src/conformal_cl/enrollment/group_conformal.py) |
| `src/enroll_RCC.py::run_enroll_rcc` | [`src/conformal_cl/enrollment/group_conformal.py`](../src/conformal_cl/enrollment/group_conformal.py) |
| `src/enroll_RCC.py::run_enroll_rcc_batch` | [`src/conformal_cl/enrollment/group_conformal.py`](../src/conformal_cl/enrollment/group_conformal.py) |

## Experiment entry points

| Original | New |
| --- | --- |
| `scripts/run_quadratic_numerical.py` | `experiments/sec45_quadratic/run.py` |
| `src/RCC.py` main block | `experiments/sec45_quadratic/legacy_sample.py` |
| `scripts/train_enroll_data.py` main block | `experiments/sec46_enrollment/train.py` |
| `scripts/run_enroll.py` main block | `experiments/sec46_enrollment/run.py` |
| `src/enroll_RCC.py` main block | `experiments/sec46_enrollment/legacy_sample.py` (broken call retained) |

Original summary documents, the prior organization proposal, and first audit remain
only in the original sibling `student-enroll/docs/` folder. Some recommendations in
that earlier proposal involved fixes; the implemented scope here excludes them.
