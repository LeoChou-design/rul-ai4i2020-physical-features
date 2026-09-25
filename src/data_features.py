"""Load the AI4I 2020 dataset, engineer physically-motivated derived features
(temperature, mechanical power, strain, interaction terms), and compute a
closed-form multi-failure-mode RUL label.

IMPORTANT — data leakage by design: the RUL label below is computed directly
from the same physical thresholds (per machine type L/M/H) that define the
four failure modes (TWF, HDF, PWF, OSF), and those columns are also used as
model inputs. A model trained on this label is learning to reconstruct a
known closed-form formula, not to generalize from sensor data that is
independent of the label. The resulting RMSE/R2 numbers look very strong
(see results/) precisely because of this; they should be read as "how well
can a model recover a known formula from its own inputs", not as a
production-ready RUL benchmark on genuine run-to-failure data.
"""

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "ai4i2020.csv")

BASE_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Type",
]
DERIVED_FEATURES = [
    "Temp_diff",
    "Temp_ratio",
    "Power",
    "Temp_Speed",
    "Torque_Speed",
    "Temp_diff_norm",
]

RANDOM_STATE = 42


def load_raw():
    return pd.read_csv(CSV_PATH)


def add_derived_features(df):
    """Add physically-motivated derived features to a copy of df."""
    df = df.copy()

    # Temperature: difference and ratio between process and ambient temperature
    df["Temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Temp_ratio"] = df["Process temperature [K]"] / df["Air temperature [K]"]

    # Mechanical power from torque and rotational speed (Watts)
    df["Power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60

    # Strain proxy: tool wear time x torque, used to assess overstrain failure (OSF) risk
    df["Strain"] = df["Tool wear [min]"] * df["Torque [Nm]"]

    # Interaction features
    df["Temp_Speed"] = df["Temp_diff"] * df["Rotational speed [rpm]"]
    df["Torque_Speed"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"]

    # Z-score normalized temperature difference
    df["Temp_diff_norm"] = (df["Temp_diff"] - df["Temp_diff"].mean()) / df["Temp_diff"].std()

    return df


def calculate_rul_multi_failure(row):
    """Closed-form RUL: minutes remaining until the soonest of four failure
    modes (TWF, HDF, PWF, OSF), per machine type. See module docstring for
    the data-leakage caveat this implies."""
    twf_limit = {"L": 200, "M": 220, "H": 240}[row["Type"]]
    rul_twf = twf_limit - row["Tool wear [min]"]

    if row["Temp_diff"] < 8.6 and row["Rotational speed [rpm]"] < 1380:
        rul_hdf = 0
    else:
        temp_margin = max(0, row["Temp_diff"] - 8.6) / 0.1
        speed_margin = max(0, row["Rotational speed [rpm]"] - 1380) / 10
        rul_hdf = min(temp_margin, speed_margin)

    power = row["Power"]
    if power < 3500:
        rul_pwf = (3500 - power) / 50
    elif power > 9000:
        rul_pwf = (power - 9000) / 50
    else:
        rul_pwf = min(abs(power - 3500), abs(power - 9000)) / 50

    strain_threshold = {"L": 11000, "M": 12000, "H": 13000}[row["Type"]]
    rul_osf = max(0, (strain_threshold - row["Strain"]) / 100)

    rul = min(rul_twf, rul_hdf, rul_pwf, rul_osf)
    return max(0, rul)


def build_dataset():
    """Return the full engineered DataFrame with RUL label attached."""
    df = load_raw()
    df = add_derived_features(df)
    df["RUL"] = df.apply(calculate_rul_multi_failure, axis=1)
    return df


def stratified_split(df):
    """60/20/20 train/val/test split, stratified on binned RUL so every
    split sees the full RUL range."""
    df = df.copy()
    df["RUL_bin"] = pd.cut(df["RUL"], bins=[0, 50, 100, 150, 200, 250], labels=False)
    df_filtered = df.dropna(subset=["RUL_bin"])

    X = df_filtered[BASE_FEATURES + DERIVED_FEATURES]
    y = df_filtered["RUL"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=df_filtered["RUL_bin"], random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5,
        stratify=df_filtered.loc[y_temp.index, "RUL_bin"], random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_preprocessor(X_train):
    numerical_features = [c for c in X_train.columns if X_train[c].dtype != "object"]
    categorical_features = [c for c in X_train.columns if X_train[c].dtype == "object"]
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])


if __name__ == "__main__":
    df = build_dataset()
    print(f"Rows: {len(df)}, RUL range: [{df['RUL'].min():.1f}, {df['RUL'].max():.1f}]")
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df)
    print(f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
