"""Generate figures: RF feature importance, actual-vs-predicted for all three
models, residual plots, PFI bar chart, and a 3-model metric comparison."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def fig1_feature_importance(rf_pipeline):
    importances = rf_pipeline["regressor"].feature_importances_
    feature_names = rf_pipeline["preprocessor"].get_feature_names_out()
    order = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[order])
    plt.xticks(range(len(importances)), [feature_names[i] for i in order], rotation=45, ha="right")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_rf_feature_importance.png"), dpi=150)
    plt.close()


def fig2_actual_vs_predicted(y_test, preds_by_model):
    for name, y_pred in preds_by_model.items():
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        lims = [min(np.min(y_test), np.min(y_pred)), max(np.max(y_test), np.max(y_pred))]
        plt.plot(lims, lims, "r--", lw=2)
        plt.xlabel("Actual RUL")
        plt.ylabel("Predicted RUL")
        plt.title(f"{name}: Actual vs. Predicted RUL")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        plt.savefig(os.path.join(FIGURES_DIR, f"fig2_actual_vs_predicted_{safe_name}.png"), dpi=150)
        plt.close()


def fig3_model_comparison():
    comparison = pd.read_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))
    pytorch_metrics = pd.read_csv(os.path.join(RESULTS_DIR, "pytorch_model_metrics.csv"))
    all_models = pd.concat([comparison, pytorch_metrics], ignore_index=True)
    all_models = all_models[all_models["Model"] != "Linear Regression"]

    metrics_to_plot = ["rmse", "mae", "r2", "asym_score"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 5))
    for ax, metric in zip(axes, metrics_to_plot):
        ax.bar(all_models["Model"], all_models[metric])
        ax.set_title(metric.upper())
        ax.tick_params(axis="x", labelrotation=30)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_model_comparison.png"), dpi=150)
    plt.close()


def fig4_pfi():
    pfi_df = pd.read_csv(os.path.join(RESULTS_DIR, "pytorch_pfi.csv"))
    plt.figure(figsize=(10, 7))
    plt.barh(pfi_df["Feature"], pfi_df["PFI_Score"], color="skyblue")
    plt.gca().invert_yaxis()
    plt.xlabel("Permutation Feature Importance (RMSE increase)")
    plt.title("PyTorch Model — Permutation Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_pytorch_pfi.png"), dpi=150)
    plt.close()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    import joblib
    from data_features import build_dataset, stratified_split

    df = build_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df)

    models_dir = os.path.join(RESULTS_DIR, "models")
    rf_pipeline = joblib.load(os.path.join(models_dir, "rf_model_pipeline.joblib"))
    gb_pipeline = joblib.load(os.path.join(models_dir, "gb_model_pipeline.joblib"))

    y_pred_rf = rf_pipeline.predict(X_test)
    y_pred_gb = gb_pipeline.predict(X_test)

    fig1_feature_importance(rf_pipeline)
    fig2_actual_vs_predicted(y_test, {"Random Forest": y_pred_rf, "Gradient Boosting": y_pred_gb})
    fig3_model_comparison()
    fig4_pfi()
    print(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
