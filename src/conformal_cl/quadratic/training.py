"""Section 4.5.1: point, wrapper-ensemble, and quantile training."""

from sklearn.model_selection import train_test_split
from .models import build_model, build_quantile_model


def train_main(dim, X, y):
    # Train the main model to predict y
    main_model = build_model(dim)
    main_model.fit(X, y, epochs=100, batch_size=32, verbose=0)
    return main_model


def train_wrapper(dim, X, y, P, seed):
    wrapper = []
    for i in range(P):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed * P + i
        )
        model = train_main(dim, X_train, y_train)
        wrapper.append(model)
    return wrapper


def train_quantiles(dim, X, y, alpha):
    quantile = build_quantile_model(dim, alpha)
    quantile.fit(X, y, epochs=1000, batch_size=32, verbose=0)
    return quantile
