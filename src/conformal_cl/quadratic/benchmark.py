"""Sections 4.5.1-4.5.2: benchmark orchestration and legacy CSV schema."""

import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
from . import problem, training, calibration
from . import group_conformal as RCC
from .optimization import optimize_compare, is_feasible


def run_one(seed, dim, noise_std, P, alpha):
    samples = problem.generate_samples(seed=seed, noise_std=noise_std, dim=dim)

    (X, y, cal_X, cal_y, A_f, A_g, b_f, b_g, c_f, c_g) = samples

    main_model = training.train_main(dim, X, y)
    quantile = training.train_quantiles(dim, X, y, alpha)
    wrapper = training.train_wrapper(dim, X, y, P, seed)
    qhat = calibration.conformalizing(quantile, cal_X, cal_y, alpha)
    opt_abs, opt_cqr, opt_wrapper, opt_true, t_abs, t_cqr, t_wrapper, t_true = (
        optimize_compare(
            dim,
            main_model,
            quantile,
            wrapper,
            P,
            alpha,
            qhat,
            A_f,
            A_g,
            b_f,
            b_g,
            c_f,
            c_g,
        )
    )

    f_abs = is_feasible(A_g, b_g, c_g, opt_abs)
    f_cqr = is_feasible(A_g, b_g, c_g, opt_cqr)
    f_wrapper = is_feasible(A_g, b_g, c_g, opt_wrapper)

    f_emp = np.zeros((3, 1))
    try:
        opts = np.stack([opt_abs, opt_cqr, opt_wrapper])
        opt_q = quantile.predict(opts, verbose=0)
        for i in range(P):
            pred = wrapper[i].predict(opts, verbose=0)
            f_emp += pred <= 0.01
        f_emp /= P
    except ValueError:
        opt_q = np.full((3, 2), np.nan)

    # Use the same quantile network for selection and evaluation.
    opt_rcc, f_rcc, t_rcc, *_ = RCC.run_rcc(
        samples, alpha, delta=0.1, quantile=quantile
    )
    emp_rcc = np.array([[0]], dtype=np.float64)
    if not np.isnan(opt_rcc).any():
        q_rcc = quantile.predict(opt_rcc.reshape(1, -1), verbose=0)[0, 1]
        for i in range(P):
            pred = wrapper[i].predict(opt_rcc.reshape(1, -1), verbose=0)
            emp_rcc += pred <= 0.01
        emp_rcc /= P
    else:
        q_rcc = np.nan

    return (
        opt_abs,
        f_abs,
        opt_q[0, 1],
        f_emp[0, 0],
        t_abs,
        opt_cqr,
        f_cqr,
        opt_q[1, 1],
        f_emp[1, 0],
        t_cqr,
        opt_wrapper,
        f_wrapper,
        opt_q[2, 1],
        f_emp[2, 0],
        t_wrapper,
        opt_rcc,
        f_rcc,
        q_rcc,
        emp_rcc[0, 0],
        t_rcc,
        opt_true,
        t_true,
    )


def run_many(csv_file, seeds, dims, noise_stds, Ps, alphas):
    for seed in seeds:
        for dim in dims:
            for noise_std in noise_stds:
                for P in Ps:
                    for alpha in alphas:
                        (
                            opt_abs,
                            f_abs,
                            qu_abs,
                            emp_abs,
                            t_abs,
                            opt_cqr,
                            f_cqr,
                            qu_cqr,
                            emp_cqr,
                            t_cqr,
                            opt_wrapper,
                            f_wrapper,
                            qu_wrapper,
                            emp_wrapper,
                            t_wrapper,
                            opt_rcc,
                            f_rcc,
                            qu_rcc,
                            emp_rcc,
                            t_rcc,
                            opt_true,
                            t_true,
                        ) = run_one(seed, dim, noise_std, P, alpha)

                        row = [
                            seed,
                            dim,
                            noise_std,
                            P,
                            alpha,
                            opt_abs,
                            f_abs,
                            qu_abs,
                            emp_abs,
                            t_abs,
                            opt_cqr,
                            f_cqr,
                            qu_cqr,
                            emp_cqr,
                            t_cqr,
                            opt_wrapper,
                            f_wrapper,
                            qu_wrapper,
                            emp_wrapper,
                            t_wrapper,
                            opt_rcc,
                            f_rcc,
                            qu_rcc,
                            emp_rcc,
                            t_rcc,
                            opt_true,
                            t_true,
                        ]

                        row_df = pd.DataFrame([row])
                        row_df.to_csv(csv_file, mode="a", header=False, index=False)
                        K.clear_session()
    return row
