"""Leakage-aware model comparison and sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupShuffleSplit, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)


@dataclass
class EvaluationResult:
    comparison: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    sensitivity: pd.DataFrame
    best_model_name: str
    best_estimator: object
    split_summary: dict[str, int | float | str]


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _target_transform(regressor: object) -> TransformedTargetRegressor:
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("regressor", regressor),
        ]
    )
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def model_catalog(random_state: int = 42) -> dict[str, object]:
    """Return transparent baseline and nonlinear candidate pipelines."""

    return {
        "Medyan Temel Model": _target_transform(DummyRegressor(strategy="median")),
        "Ridge Regresyon": _target_transform(Ridge(alpha=10.0)),
        "Random Forest": _target_transform(
            RandomForestRegressor(
                n_estimators=250,
                max_features=0.75,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=random_state,
            )
        ),
        "Extra Trees": _target_transform(
            ExtraTreesRegressor(
                n_estimators=250,
                max_features=0.85,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=random_state,
            )
        ),
        "Histogram Gradient Boosting": _target_transform(
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=random_state,
            )
        ),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def _stratification_bins(target: pd.Series) -> pd.Series | None:
    try:
        bins = pd.qcut(target, q=10, labels=False, duplicates="drop")
    except ValueError:
        return None
    counts = bins.value_counts(dropna=False)
    return bins if len(counts) >= 2 and counts.min() >= 2 else None


def _bootstrap_mae_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    random_state: int,
    iterations: int = 2_000,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    errors = np.abs(y_true - y_pred)
    samples = rng.integers(0, len(errors), size=(iterations, len(errors)))
    estimates = errors[samples].mean(axis=1)
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def _chronological_sensitivity(
    estimator: object,
    data: pd.DataFrame,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    ordered = data.sort_values("İlan Tarihi Parsed").reset_index(drop=True)
    candidate_index = int(len(ordered) * 0.80)
    cutoff = ordered.loc[candidate_index, "İlan Tarihi Parsed"]
    train = ordered.loc[ordered["İlan Tarihi Parsed"] < cutoff]
    test = ordered.loc[ordered["İlan Tarihi Parsed"] >= cutoff]

    if len(train) < 100 or len(test) < 50:
        train = ordered.iloc[:candidate_index]
        test = ordered.iloc[candidate_index:]
        cutoff = test["İlan Tarihi Parsed"].min()

    fitted = clone(estimator).fit(train[MODEL_FEATURES], train[TARGET])
    prediction = fitted.predict(test[MODEL_FEATURES])
    metrics = regression_metrics(test[TARGET].to_numpy(), prediction)
    metrics.update(
        {
            "protocol": "Kronolojik sağlamlık kontrolü",
            "train_rows": len(train),
            "test_rows": len(test),
            "cutoff_date": cutoff.date().isoformat(),
        }
    )
    predictions = test[["İlan Kimliği", "İlan Tarihi Parsed", TARGET]].copy()
    predictions["prediction"] = prediction
    predictions["protocol"] = "chronological"
    return metrics, predictions


def _snapshot_protocol_sensitivity(
    estimator: object,
    snapshots: pd.DataFrame,
    *,
    random_state: int,
) -> list[dict[str, float | int | str]]:
    features = snapshots[MODEL_FEATURES]
    target = snapshots[TARGET]

    naive_train, naive_test = train_test_split(
        np.arange(len(snapshots)),
        test_size=0.20,
        random_state=random_state,
        stratify=_stratification_bins(target),
    )
    naive_model = clone(estimator).fit(features.iloc[naive_train], target.iloc[naive_train])
    naive_pred = naive_model.predict(features.iloc[naive_test])
    naive_metrics = regression_metrics(target.iloc[naive_test].to_numpy(), naive_pred)
    naive_metrics.update(
        {
            "protocol": "Satır bazlı rastgele bölme (sızıntı riski)",
            "train_rows": len(naive_train),
            "test_rows": len(naive_test),
            "cutoff_date": "-",
        }
    )

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    group_train, group_test = next(
        splitter.split(features, target, groups=snapshots["İlan Kimliği"])
    )
    group_model = clone(estimator).fit(features.iloc[group_train], target.iloc[group_train])
    group_pred = group_model.predict(features.iloc[group_test])
    group_metrics = regression_metrics(target.iloc[group_test].to_numpy(), group_pred)
    group_metrics.update(
        {
            "protocol": "İlan kimliği gruplu bölme",
            "train_rows": len(group_train),
            "test_rows": len(group_test),
            "cutoff_date": "-",
        }
    )
    return [naive_metrics, group_metrics]


def evaluate_models(
    latest: pd.DataFrame,
    snapshots: pd.DataFrame,
    *,
    random_state: int = 42,
) -> EvaluationResult:
    """Select a model using CV on training data, then evaluate a locked holdout."""

    x = latest[MODEL_FEATURES]
    y = latest[TARGET]
    train_index, test_index = train_test_split(
        np.arange(len(latest)),
        test_size=0.20,
        random_state=random_state,
        stratify=_stratification_bins(y),
    )
    x_train = x.iloc[train_index]
    x_test = x.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    folds = KFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    fitted_models: dict[str, object] = {}
    rows: list[dict[str, float | str]] = []

    for name, estimator in model_catalog(random_state).items():
        scores = cross_validate(
            estimator,
            x_train,
            y_train,
            cv=folds,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        fitted = clone(estimator).fit(x_train, y_train)
        prediction = fitted.predict(x_test)
        test_metrics = regression_metrics(y_test.to_numpy(), prediction)
        rows.append(
            {
                "model": name,
                "cv_mae_mean": float(-scores["test_mae"].mean()),
                "cv_mae_std": float(scores["test_mae"].std()),
                "cv_rmse_mean": float(-scores["test_rmse"].mean()),
                "cv_r2_mean": float(scores["test_r2"].mean()),
                "cv_r2_std": float(scores["test_r2"].std()),
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "test_mape": test_metrics["mape"],
            }
        )
        fitted_models[name] = fitted

    comparison = pd.DataFrame(rows).sort_values("cv_mae_mean").reset_index(drop=True)
    best_name = str(comparison.iloc[0]["model"])
    best_estimator = fitted_models[best_name]
    best_prediction = best_estimator.predict(x_test)
    interval_low, interval_high = _bootstrap_mae_interval(
        y_test.to_numpy(), best_prediction, random_state=random_state
    )

    predictions = latest.iloc[test_index][
        ["İlan Kimliği", "İlan Tarihi Parsed", "Mahalle", TARGET, "Brüt m²"]
    ].copy()
    predictions["prediction"] = best_prediction
    predictions["residual"] = predictions[TARGET] - predictions["prediction"]
    predictions["absolute_error"] = predictions["residual"].abs()
    predictions["protocol"] = "locked_random_holdout"

    importance = permutation_importance(
        best_estimator,
        x_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=15,
        random_state=random_state,
        n_jobs=-1,
    )
    feature_importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "mae_increase_mean": importance.importances_mean,
            "mae_increase_std": importance.importances_std,
        }
    ).sort_values("mae_increase_mean", ascending=False)

    chronological_metrics, chronological_predictions = _chronological_sensitivity(
        best_estimator, latest
    )
    snapshot_metrics = _snapshot_protocol_sensitivity(
        best_estimator, snapshots, random_state=random_state
    )
    locked_metrics = regression_metrics(y_test.to_numpy(), best_prediction)
    locked_metrics.update(
        {
            "protocol": "Tekil ilanlar: kilitli rastgele test",
            "train_rows": len(train_index),
            "test_rows": len(test_index),
            "cutoff_date": "-",
        }
    )
    sensitivity = pd.DataFrame([locked_metrics, chronological_metrics, *snapshot_metrics])
    predictions = pd.concat([predictions, chronological_predictions], ignore_index=True, sort=False)

    split_summary = {
        "random_state": random_state,
        "train_rows": len(train_index),
        "test_rows": len(test_index),
        "cross_validation_folds": 5,
        "mae_ci_95_low": float(interval_low),
        "mae_ci_95_high": float(interval_high),
        "best_model": best_name,
    }
    return EvaluationResult(
        comparison=comparison,
        predictions=predictions,
        feature_importance=feature_importance.reset_index(drop=True),
        sensitivity=sensitivity,
        best_model_name=best_name,
        best_estimator=best_estimator,
        split_summary=split_summary,
    )


def fit_dashboard_model(data: pd.DataFrame, random_state: int = 42) -> object:
    """Fit the selected portfolio-friendly model on all latest listings."""

    estimator = model_catalog(random_state)["Histogram Gradient Boosting"]
    return estimator.fit(data[MODEL_FEATURES], data[TARGET])
