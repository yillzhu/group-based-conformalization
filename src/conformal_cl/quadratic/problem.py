"""Section 4.5.1: quadratic instances, noisy data, true feasibility."""

import numpy as np
import tensorflow as tf


def generate_quadratic(dim):
    A_f = np.random.normal(0, 5, (dim, dim))
    A_g = np.random.normal(0, 5, (dim, dim))
    b_f = np.random.normal(0, 1, (dim, 1))
    b_g = np.random.normal(0, 1, (dim, 1))
    c_f = np.random.normal(0, 1)
    c_g = np.random.normal(0, 1)
    return A_f, A_g, b_f, b_g, c_f, c_g


def generate_samples(seed, noise_std, dim):
    # Keras 3 has RNG state beyond ``tf.random.set_seed``.  Seed Python, NumPy,
    # and TensorFlow together and request deterministic kernels so a parameter
    # tuple produces the same trained models both alone and inside a sweep.
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    A_f, A_g, b_f, b_g, c_f, c_g = generate_quadratic(dim)

    if dim == 1:
        sample_sz = 100
    elif dim == 2:
        sample_sz = 500
    elif dim == 3:
        sample_sz = 2000

    sz = int(2 * sample_sz)
    X = np.random.uniform(low=-1, high=1, size=(sz, dim))
    noise = noise_std * np.random.normal(loc=0, scale=1, size=(sz, 1))
    y = np.zeros((sz, 1))
    for i in range(sz):
        X_i = X[i, :]
        y[i] = X_i @ A_g @ X_i.T + b_g.T @ X_i.T + c_g
    y = y + noise * np.std(y)
    return (
        X[:sample_sz, :],
        y[:sample_sz, :],
        X[sample_sz:, :],
        y[sample_sz:, :],
        A_f,
        A_g,
        b_f,
        b_g,
        c_f,
        c_g,
    )


def is_feasible(A_g, b_g, c_g, xhat):
    if np.any(np.isnan(xhat)):
        return -1
    x = xhat.reshape(-1, 1)
    return int((x.T @ A_g @ x + b_g.T @ x + c_g <= 0).item())
