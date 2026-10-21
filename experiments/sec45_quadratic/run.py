"""Section 4.5: original sweep parameters and append-row CSV output.

Only module imports and filesystem wiring have been reorganized.
"""

from conformal_cl.paths import PROJECT_ROOT
from conformal_cl.quadratic import benchmark as quadratic_numerical
import pandas as pd
import os

if __name__ == "__main__":
    # Create results directory
    results_dir = str(PROJECT_ROOT / "results" / "quadratic")
    os.makedirs(results_dir, exist_ok=True)

    # set parameters
    seeds = range(10)
    dims = [1, 2, 3]
    noise_stds = [0.1, 0.15]
    Ps = [10]
    alphas = [0.1]

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
    csv_file = f"{results_dir}/seed_{seed_start}_{seed_end}.csv"
    df.to_csv(csv_file, index=False)

    quadratic_numerical.run_many(
        csv_file=csv_file,
        seeds=seeds,
        dims=dims,
        noise_stds=noise_stds,
        Ps=Ps,
        alphas=alphas,
    )

