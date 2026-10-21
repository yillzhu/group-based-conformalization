"""Section 4.5 corrected quadratic sweep with a lightweight smoke mode."""

import argparse
from pathlib import Path

from conformal_cl.paths import PROJECT_ROOT
from conformal_cl.provenance import write_run_manifest
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run only seed=0, dim=1, noise_std=.1, P=2, alpha=.1",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--dims", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--noise-stds", type=float, nargs="+", default=[0.1, 0.15])
    parser.add_argument("--ensemble-sizes", type=int, nargs="+", default=[10])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "corrected" / "quadratic",
    )
    args = parser.parse_args()

    # Importing the benchmark loads TensorFlow. Keeping it after argument parsing
    # lets users inspect --help in the lightweight base environment.
    from conformal_cl.quadratic import benchmark as quadratic_numerical

    results_dir = args.output_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        seeds, dims, noise_stds, Ps, alphas = [0], [1], [0.1], [2], [0.1]
    else:
        seeds = args.seeds
        dims = args.dims
        noise_stds = args.noise_stds
        Ps = args.ensemble_sizes
        alphas = args.alphas

    if not all(dim in (1, 2, 3) for dim in dims):
        parser.error("--dims values must be 1, 2, or 3")
    if not all(P > 0 for P in Ps):
        parser.error("--ensemble-sizes values must be positive")
    if not all(0 < alpha < 1 for alpha in alphas):
        parser.error("--alphas values must be in (0, 1)")

    # generate empty dataframe to store results
    columns = [
        "seed",
        "dim",
        "noise_std",
        "P",
        "alpha",
        "opt_abs",
        "f_abs",
        "qu_abs",
        "emp_abs",
        "t_abs",
        "opt_cqr",
        "f_cqr",
        "qu_cqr",
        "emp_cqr",
        "t_cqr",
        "opt_wrapper",
        "f_wrapper",
        "qu_wrapper",
        "emp_wrapper",
        "t_wrapper",
        "opt_rcc",
        "f_rcc",
        "qu_rcc",
        "emp_rcc",
        "t_rcc",
        "opt_true",
        "t_true",
    ]
    df = pd.DataFrame(columns=columns)

    seed_start = seeds[0]
    seed_end = seeds[-1]
    csv_file = results_dir / f"seed_{seed_start}_{seed_end}.csv"
    df.to_csv(csv_file, index=False)

    quadratic_numerical.run_many(
        csv_file=csv_file,
        seeds=seeds,
        dims=dims,
        noise_stds=noise_stds,
        Ps=Ps,
        alphas=alphas,
    )

    write_run_manifest(
        results_dir / f"seed_{seed_start}_{seed_end}_manifest.json",
        run_kind="corrected_quadratic_smoke" if args.smoke else "corrected_quadratic_sweep",
        parameters={
            "seeds": list(seeds),
            "dims": list(dims),
            "noise_stds": list(noise_stds),
            "ensemble_sizes": list(Ps),
            "alphas": list(alphas),
        },
        inputs=[],
        artifacts=[csv_file],
    )

