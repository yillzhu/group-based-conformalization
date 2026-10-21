"""Section 4.5 corrected quadratic sweep with a lightweight smoke mode."""

import argparse
import hashlib
import json
from itertools import product
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
    parser.add_argument(
        "--output-file",
        type=Path,
        help="explicit CSV path; default name is derived from the full parameter grid",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="validate an existing partial CSV and skip its completed parameter keys",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output instead of refusing a collision",
    )
    args = parser.parse_args()

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
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")

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
        "status_abs",
        "status_cqr",
        "status_wrapper",
        "status_rcc",
        "status_true",
    ]
    parameters = {
        "seeds": list(seeds),
        "dims": list(dims),
        "noise_stds": list(noise_stds),
        "ensemble_sizes": list(Ps),
        "alphas": list(alphas),
    }
    run_id = hashlib.sha256(
        json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    csv_file = args.output_file or results_dir / f"quadratic_{run_id}.csv"
    if not csv_file.is_absolute():
        csv_file = (PROJECT_ROOT / csv_file).resolve()
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = csv_file.with_suffix(".manifest.json")

    requested_keys = set(product(seeds, dims, noise_stds, Ps, alphas))
    completed_keys = set()
    if csv_file.exists() and not args.overwrite:
        if not args.resume:
            parser.error(
                f"output already exists: {csv_file}; use --resume or --overwrite"
            )
        existing = pd.read_csv(csv_file)
        if list(existing.columns) != columns:
            parser.error("existing output schema does not match the current runner")
        key_columns = ["seed", "dim", "noise_std", "P", "alpha"]
        if existing.duplicated(key_columns).any():
            parser.error("existing output contains duplicate parameter keys")
        completed_keys = {
            tuple(row) for row in existing[key_columns].itertuples(index=False, name=None)
        }
        unexpected = completed_keys - requested_keys
        if unexpected:
            parser.error(f"existing output contains keys outside this grid: {unexpected}")
        if len(existing):
            from conformal_cl.quadratic.summary import load_corrected_results

            try:
                load_corrected_results([csv_file])
            except ValueError as error:
                parser.error(f"existing output failed validation: {error}")
    else:
        pd.DataFrame(columns=columns).to_csv(csv_file, index=False)
        if manifest_path.exists():
            manifest_path.unlink()

    if completed_keys != requested_keys:
        # Importing the benchmark loads TensorFlow. Defer it until a solve is
        # actually needed, after lightweight collision and resume validation.
        from conformal_cl.quadratic import benchmark as quadratic_numerical

        completed_keys = quadratic_numerical.run_many(
            csv_file=csv_file,
            seeds=seeds,
            dims=dims,
            noise_stds=noise_stds,
            Ps=Ps,
            alphas=alphas,
            skip_keys=completed_keys,
        )
    if completed_keys != requested_keys:
        raise RuntimeError(
            f"incomplete sweep: completed {len(completed_keys)} of {len(requested_keys)} keys"
        )

    write_run_manifest(
        manifest_path,
        run_kind="corrected_quadratic_smoke" if args.smoke else "corrected_quadratic_sweep",
        parameters={
            **parameters,
            "run_id": run_id,
            "resume_supported": True,
        },
        inputs=[],
        artifacts=[csv_file],
    )

