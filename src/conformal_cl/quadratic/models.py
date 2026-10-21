"""Section 4.5.1: Keras point and interval networks."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(dim):
    if dim == 1:
        l1 = 10
        l2 = 5
    elif dim == 2:
        l1 = 20
        l2 = 10
    elif dim == 3:
        l1 = 40
        l2 = 20

    model = keras.Sequential(
        [
            layers.InputLayer((dim,)),
            layers.Dense(l1, activation="relu"),
            layers.Dense(l2, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def interval_score_loss(y_true, y_pred, alpha=0.1):
    ql = tf.slice(y_pred, [0, 0], [-1, 1])
    qu = tf.slice(y_pred, [0, 1], [-1, 1])
    interval_width = qu - ql

    lower_penalty = (ql - y_true) * tf.cast(y_true < ql, tf.float32)
    upper_penalty = (y_true - qu) * tf.cast(y_true > qu, tf.float32)

    interval_score = interval_width + (2 / alpha) * (lower_penalty + upper_penalty)

    return tf.reduce_mean(interval_score)


def build_quantile_model(dim, alpha):
    if dim == 1:
        l1 = 10
        l2 = 5
    elif dim == 2:
        l1 = 20
        l2 = 10
    elif dim == 3:
        l1 = 40
        l2 = 20

    model = keras.Sequential(
        [
            layers.InputLayer((dim,)),
            layers.Dense(l1, activation="relu"),
            layers.Dense(l2, activation="relu"),
            layers.Dense(2),
        ]
    )
    def configured_interval_score_loss(y_true, y_pred):
        return interval_score_loss(y_true, y_pred, alpha=alpha)

    model.compile(optimizer="adam", loss=configured_interval_score_loss)
    return model
