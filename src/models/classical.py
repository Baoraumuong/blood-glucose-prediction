"""
models/classical.py
-------------------
Classical machine-learning models with a unified train-evaluate interface.
Each model runner takes pre-scaled 2-D arrays and a ResultStore, then
performs KFold CV + hold-out test evaluation, writing results to the store.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

from src.evaluation.metrics import ResultStore, cv_evaluate, test_evaluate
from src.models.persistence import save_pickle_artifact

logger = logging.getLogger(__name__)


def _rmse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true.ravel() - y_pred.ravel()) ** 2)))


def _maybe_prefix_grid(
    param_grid: dict[str, list[Any]],
    prefix: str | None,
) -> dict[str, list[Any]]:
    if not prefix:
        return param_grid
    return {
        key if "__" in key else f"{prefix}__{key}": values
        for key, values in param_grid.items()
    }


def _tune_model(
    model: Any,
    name: str,
    feature_tag: str,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: dict[str, list[Any]] | None,
    cfg: dict[str, Any],
    n_folds: int,
    shuffle: bool,
    grid_prefix: str | None = None,
) -> tuple[Any, dict[str, Any] | None]:
    """Run optional GridSearchCV and return the best estimator."""
    if not param_grid:
        return model, None

    tuning_cfg = cfg.get("evaluation", {}).get("grid_search", {})
    if not tuning_cfg.get("enabled", False):
        return model, None

    tune_X, tune_y = X, y
    subsample = tuning_cfg.get("subsample")
    if subsample and len(X) > subsample:
        rng = np.random.default_rng(cfg.get("seed", 42))
        idx = rng.choice(len(X), subsample, replace=False)
        tune_X, tune_y = X[idx], y[idx]
        logger.info(
            "Grid search for %s (%s) using %d/%d training samples",
            name,
            feature_tag,
            len(tune_X),
            len(X),
        )

    grid = _maybe_prefix_grid(param_grid, grid_prefix)
    cv = KFold(n_splits=n_folds, shuffle=shuffle)
    search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        scoring=make_scorer(_rmse_loss, greater_is_better=False),
        cv=cv,
        n_jobs=tuning_cfg.get("n_jobs", 1),
        refit=True,
        verbose=tuning_cfg.get("verbose", 0),
    )
    logger.info(
        "Grid search for %s (%s): %d parameter(s)",
        name,
        feature_tag,
        len(grid),
    )
    search.fit(tune_X, tune_y)
    best_params = {
        key.removeprefix(f"{grid_prefix}__") if grid_prefix else key: value
        for key, value in search.best_params_.items()
    }
    logger.info(
        "Best %s (%s) params: %s | CV RMSE=%.3f",
        name,
        feature_tag,
        best_params,
        -search.best_score_,
    )
    return search.best_estimator_, best_params


def _run_model(
    model: Any,
    name: str,
    feature_tag: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    store:   ResultStore,
    cfg: dict[str, Any],
    n_folds: int = 5,
    svr_subsample: int | None = None,
    train_subsample: int | None = None,
    seed: int = 42,
    shuffle: bool = False,
    feature_names: list[str] | None = None,
    scaler: Any | None = None,
    feature_cols: list[str] | None = None,
    param_grid: dict[str, list[Any]] | None = None,
    grid_prefix: str | None = None,
) -> None:
    """Shared CV + test routine for classical models."""
    logger.info("Training %s (%s) …", name, feature_tag)

    # Cross-validation (optionally sub-sampled for slow models like SVR)
    Xcv, ycv = X_train, y_train
    if svr_subsample and len(X_train) > svr_subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_train), svr_subsample, replace=False)
        Xcv, ycv = X_train[idx], y_train[idx]

    Xfit, yfit = X_train, y_train
    if train_subsample and len(X_train) > train_subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_train), train_subsample, replace=False)
        Xfit, yfit = X_train[idx], y_train[idx]

    model, best_params = _tune_model(
        model=model,
        name=name,
        feature_tag=feature_tag,
        X=Xcv,
        y=ycv,
        param_grid=param_grid,
        cfg=cfg,
        n_folds=n_folds,
        shuffle=shuffle,
        grid_prefix=grid_prefix,
    )
    cv_metrics  = cv_evaluate(model, Xcv, ycv, n_folds=n_folds, shuffle=shuffle)
    preds, test_metrics = test_evaluate(model, Xfit, yfit, X_test, y_test)
    artifact_path = save_pickle_artifact(
        cfg=cfg,
        model_name=name,
        feature_tag=feature_tag,
        filename="model.pkl",
        payload={
            "model": model,
            "model_name": name,
            "feature_tag": feature_tag,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "feature_names": feature_names,
            "best_params": best_params,
        },
    )
    logger.info("Saved %s (%s) artifact: %s", name, feature_tag, artifact_path)
    store.add(name, feature_tag, cv_metrics, test_metrics, preds)
    _log_top_feature_importances(model, name, feature_tag, feature_names)


def _model_importances(model: Any) -> np.ndarray | None:
    """Extract aggregate feature importances or absolute coefficients."""
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        return np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    if hasattr(model, "estimators_"):
        values = [_model_importances(est) for est in model.estimators_]
        values = [v for v in values if v is not None]
        if values:
            return np.mean(values, axis=0)
    return None


def _log_top_feature_importances(
    model: Any,
    name: str,
    feature_tag: str,
    feature_names: list[str] | None,
    top_n: int = 5,
) -> None:
    if not feature_names:
        return
    importances = _model_importances(model)
    if importances is None or len(importances) != len(feature_names):
        return
    order = np.argsort(importances)[::-1][:top_n]
    formatted = ", ".join(
        f"{feature_names[i]}={importances[i]:.4f}" for i in order
    )
    logger.info("Top %d features for %s (%s): %s", top_n, name, feature_tag, formatted)


def _iter_classical_variants(datasets: dict[str, Any]) -> list[dict[str, Any]]:
    if "classical_variants" in datasets:
        return datasets["classical_variants"]
    return [
        dict(
            tag="with features",
            X_train_2d=datasets["X_train_2d_full_s"],
            X_test_2d=datasets["X_test_2d_full_s"],
            y_train=datasets["y_train"],
            y_test=datasets["y_test"],
            flat_feature_names=datasets["full_flat_feature_names"],
        ),
        dict(
            tag="baseline",
            X_train_2d=datasets["X_train_2d_base_s"],
            X_test_2d=datasets["X_test_2d_base_s"],
            y_train=datasets["y_train_base"],
            y_test=datasets["y_test_base"],
            flat_feature_names=datasets["base_flat_feature_names"],
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Individual model runners
# ─────────────────────────────────────────────────────────────────────────────

def run_linear_regression(
    datasets: dict[str, Any],
    store: ResultStore,
    cfg: dict[str, Any],
) -> None:
    eval_cfg = cfg.get("evaluation", {})
    n_folds  = eval_cfg.get("n_folds", 5)
    shuffle  = eval_cfg.get("kfold_shuffle", False)

    for variant in _iter_classical_variants(datasets):
        tag = variant["tag"]
        _run_model(LinearRegression(), "LinearRegression", tag,
                   variant["X_train_2d"], variant["y_train"],
                   variant["X_test_2d"], variant["y_test"], store, cfg, n_folds,
                   shuffle=shuffle,
                   feature_names=variant["flat_feature_names"],
                   scaler=variant.get("scaler"),
                   feature_cols=variant.get("feature_cols"))


def run_svr(
    datasets: dict[str, Any],
    store: ResultStore,
    cfg: dict[str, Any],
) -> None:
    eval_cfg  = cfg.get("evaluation", {})
    model_cfg = cfg["models"].get("svr", {})
    n_folds   = eval_cfg.get("n_folds", 5)
    shuffle   = eval_cfg.get("kfold_shuffle", False)
    subsample = eval_cfg.get("svr_subsample", 5000)
    train_subsample = eval_cfg.get("svr_train_subsample", subsample)
    seed = cfg.get("seed", 42)

    base_svr = SVR(
        kernel=model_cfg.get("kernel", "rbf"),
        C=model_cfg.get("C", 10),
        epsilon=model_cfg.get("epsilon", 0.1),
        cache_size=model_cfg.get("cache_size", 1000),
        shrinking=model_cfg.get("shrinking", True),
    )

    for variant in _iter_classical_variants(datasets):
        tag = variant["tag"]
        model = MultiOutputRegressor(base_svr, n_jobs=-1)
        _run_model(model, "SVR", tag,
                   variant["X_train_2d"], variant["y_train"],
                   variant["X_test_2d"], variant["y_test"], store,
                   cfg, n_folds, svr_subsample=subsample,
                   train_subsample=train_subsample, seed=seed, shuffle=shuffle,
                   feature_names=variant["flat_feature_names"],
                   scaler=variant.get("scaler"),
                   feature_cols=variant.get("feature_cols"),
                   param_grid=model_cfg.get("param_grid"),
                   grid_prefix="estimator")


def run_xgboost(
    datasets: dict[str, Any],
    store: ResultStore,
    cfg: dict[str, Any],
) -> None:
    eval_cfg  = cfg.get("evaluation", {})
    model_cfg = cfg["models"].get("xgboost", {})
    n_folds   = eval_cfg.get("n_folds", 5)
    shuffle   = eval_cfg.get("kfold_shuffle", False)
    seed      = cfg.get("seed", 42)

    base_xgb = xgb.XGBRegressor(
        n_estimators=model_cfg.get("n_estimators", 300),
        max_depth=model_cfg.get("max_depth", 5),
        learning_rate=model_cfg.get("learning_rate", 0.05),
        subsample=model_cfg.get("subsample", 0.8),
        colsample_bytree=model_cfg.get("colsample_bytree", 0.8),
        random_state=seed,
        verbosity=0,
        n_jobs=-1,
    )

    for variant in _iter_classical_variants(datasets):
        tag = variant["tag"]
        model = MultiOutputRegressor(base_xgb, n_jobs=-1)
        _run_model(model, "XGBoost", tag,
                   variant["X_train_2d"], variant["y_train"],
                   variant["X_test_2d"], variant["y_test"], store, cfg, n_folds,
                   shuffle=shuffle,
                   feature_names=variant["flat_feature_names"],
                   scaler=variant.get("scaler"),
                   feature_cols=variant.get("feature_cols"),
                   param_grid=model_cfg.get("param_grid"),
                   grid_prefix="estimator")


def run_decision_tree(
    datasets: dict[str, Any],
    store: ResultStore,
    cfg: dict[str, Any],
) -> None:
    eval_cfg  = cfg.get("evaluation", {})
    model_cfg = cfg["models"].get("decision_tree", {})
    n_folds   = eval_cfg.get("n_folds", 5)
    shuffle   = eval_cfg.get("kfold_shuffle", False)
    seed      = cfg.get("seed", 42)

    for variant in _iter_classical_variants(datasets):
        tag = variant["tag"]
        model = DecisionTreeRegressor(
            max_depth=model_cfg.get("max_depth", 8),
            min_samples_leaf=model_cfg.get("min_samples_leaf", 10),
            random_state=seed,
        )
        _run_model(model, "DecisionTree", tag,
                   variant["X_train_2d"], variant["y_train"],
                   variant["X_test_2d"], variant["y_test"], store, cfg, n_folds,
                   shuffle=shuffle,
                   feature_names=variant["flat_feature_names"],
                   scaler=variant.get("scaler"),
                   feature_cols=variant.get("feature_cols"),
                   param_grid=model_cfg.get("param_grid"))


def run_random_forest(
    datasets: dict[str, Any],
    store: ResultStore,
    cfg: dict[str, Any],
) -> None:
    eval_cfg  = cfg.get("evaluation", {})
    model_cfg = cfg["models"].get("random_forest", {})
    n_folds   = eval_cfg.get("n_folds", 5)
    shuffle   = eval_cfg.get("kfold_shuffle", False)
    seed      = cfg.get("seed", 42)

    for variant in _iter_classical_variants(datasets):
        tag = variant["tag"]
        model = RandomForestRegressor(
            n_estimators=model_cfg.get("n_estimators", 200),
            max_depth=model_cfg.get("max_depth", 10),
            min_samples_leaf=model_cfg.get("min_samples_leaf", 5),
            n_jobs=-1,
            random_state=seed,
        )
        _run_model(model, "RandomForest", tag,
                   variant["X_train_2d"], variant["y_train"],
                   variant["X_test_2d"], variant["y_test"], store, cfg, n_folds,
                   shuffle=shuffle,
                   feature_names=variant["flat_feature_names"],
                   scaler=variant.get("scaler"),
                   feature_cols=variant.get("feature_cols"),
                   param_grid=model_cfg.get("param_grid"))


def run_knn(
    datasets: dict[str, Any],
    store: ResultStore,
    cfg: dict[str, Any],
) -> None:
    eval_cfg  = cfg.get("evaluation", {})
    model_cfg = cfg["models"].get("knn", {})
    n_folds   = eval_cfg.get("n_folds", 5)
    shuffle   = eval_cfg.get("kfold_shuffle", False)

    for variant in _iter_classical_variants(datasets):
        tag = variant["tag"]
        model = KNeighborsRegressor(
            n_neighbors=model_cfg.get("n_neighbors", 10),
            weights=model_cfg.get("weights", "distance"),
            n_jobs=-1,
        )
        _run_model(model, "KNN", tag,
                   variant["X_train_2d"], variant["y_train"],
                   variant["X_test_2d"], variant["y_test"], store, cfg, n_folds,
                   shuffle=shuffle,
                   feature_names=variant["flat_feature_names"],
                   scaler=variant.get("scaler"),
                   feature_cols=variant.get("feature_cols"),
                   param_grid=model_cfg.get("param_grid"))
