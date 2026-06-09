"""
models.py
---------
Builds sklearn-compatible pipelines for all regression models:

  LinearRegression – Ordinary Least Squares (no regularisation).
  Ridge            – Ridge Regression (L2-penalised linear model).
  MLP              – Multi-layer Perceptron (sklearn MLPRegressor).
  SVR              – Support Vector Regression (RBF kernel).
  GradientBoosting – Gradient Boosted Trees.

All pipelines include StandardScaler as the first step so that raw
feature magnitudes are normalised before the estimator sees them.
"""

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
_MLP_HIDDEN  = (256, 128)
_RIDGE_ALPHA = 1.0


# ── Pipeline factories ────────────────────────────────────────────────────────

def make_linear() -> Pipeline:
    """StandardScaler + Ordinary Least Squares regression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()),
    ])


def make_ridge(alpha: float = _RIDGE_ALPHA) -> Pipeline:
    """StandardScaler + Ridge regression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=alpha)),
    ])


def make_mlp(hidden_layer_sizes: tuple = _MLP_HIDDEN) -> Pipeline:
    """StandardScaler + sklearn MLP regressor."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=1e-2,
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=500,
            random_state=RANDOM_STATE,
        )),
    ])


def make_svr(C: float = 1.0, epsilon: float = 0.1) -> Pipeline:
    """StandardScaler + Support Vector Regression (RBF kernel)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  SVR(kernel="rbf", C=C, epsilon=epsilon)),
    ])


def make_gb(n_estimators: int = 100, max_depth: int = 3) -> Pipeline:
    """StandardScaler + Gradient Boosted Trees."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=RANDOM_STATE,
        )),
    ])


# ── Registry ──────────────────────────────────────────────────────────────────

MODEL_FACTORIES = {
    "LinearRegression": make_linear,
    "Ridge":            make_ridge,
    "MLP":              make_mlp,
    "SVR":              make_svr,
    "GradientBoosting": make_gb,
}

AVAILABLE_MODELS = list(MODEL_FACTORIES.keys())


def get_pipeline(model_name: str) -> Pipeline:
    """Returns a fresh (unfitted) pipeline for the given model name."""
    if model_name not in MODEL_FACTORIES:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from {AVAILABLE_MODELS}"
        )
    return MODEL_FACTORIES[model_name]()
