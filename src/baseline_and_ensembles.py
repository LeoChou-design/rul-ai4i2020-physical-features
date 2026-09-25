"""Train Linear Regression (baseline), Random Forest, and Gradient Boosting
pipelines (preprocessing + regressor) on the AI4I 2020 RUL label, evaluate
with the asymmetric scoring function, and save the fitted pipelines."""

import os

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from data_features import build_dataset, make_preprocessor, stratified_split

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "models")


def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n=== {model_name} ===")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def rul_scoring_function(y_true, y_pred):
    """Asymmetric score: overestimating remaining life (diff >= 0, a late
    warning) is penalized more heavily than underestimating it (diff < 0, an
    early / safer warning)."""
    diff = np.asarray(y_pred) - np.asarray(y_true)
    scores = np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)
    return float(np.mean(scores))


def evaluate_rul_model(y_true, y_pred, model_name):
    results = evaluate_model(y_true, y_pred, model_name)
    results["asym_score"] = rul_scoring_function(y_true, y_pred)
    print(f"Asymmetric Score: {results['asym_score']:.2f}")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    early_mask = y_true > 100
    late_mask = y_true <= 100

    results["early_mae"] = (
        float(mean_absolute_error(y_true[early_mask], y_pred[early_mask]))
        if early_mask.sum() > 0 else None
    )
    results["late_mae"] = (
        float(mean_absolute_error(y_true[late_mask], y_pred[late_mask]))
        if late_mask.sum() > 0 else None
    )
    if results["early_mae"] is not None:
        print(f"Early Stage MAE (RUL>100): {results['early_mae']:.2f}")
    if results["late_mae"] is not None:
        print(f"Late Stage MAE (RUL<=100): {results['late_mae']:.2f}")
    return results


def build_pipeline(regressor, X_train):
    return Pipeline(steps=[
        ("preprocessor", make_preprocessor(X_train)),
        ("regressor", regressor),
    ])


def main():
    df = build_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    baseline = build_pipeline(LinearRegression(), X_train)
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)

    rf_pipeline = build_pipeline(
        RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_split=10,
            min_samples_leaf=4, random_state=42, n_jobs=-1,
        ),
        X_train,
    )
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    joblib.dump(rf_pipeline, os.path.join(MODELS_DIR, "rf_model_pipeline.joblib"))

    gb_pipeline = build_pipeline(
        GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42,
        ),
        X_train,
    )
    gb_pipeline.fit(X_train, y_train)
    y_pred_gb = gb_pipeline.predict(X_test)
    joblib.dump(gb_pipeline, os.path.join(MODELS_DIR, "gb_model_pipeline.joblib"))

    results = {
        "Linear Regression": evaluate_rul_model(y_test, y_pred_baseline, "Linear Regression (baseline)"),
        "Random Forest": evaluate_rul_model(y_test, y_pred_rf, "Random Forest"),
        "Gradient Boosting": evaluate_rul_model(y_test, y_pred_gb, "Gradient Boosting"),
    }

    import pandas as pd
    rows = [{"Model": name, **metrics} for name, metrics in results.items()]
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

    return rf_pipeline, gb_pipeline, X_train, X_val, X_test, y_train, y_val, y_test, y_pred_rf, y_pred_gb


if __name__ == "__main__":
    main()
