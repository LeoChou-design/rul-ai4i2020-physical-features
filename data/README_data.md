# Data: AI4I 2020 Predictive Maintenance Dataset

10,000 synthetic CNC machine records with five failure modes (TWF, HDF, PWF, OSF, RNF).

| File | Content | Size |
|---|---|---|
| `ai4i2020.csv` | Raw dataset as released by the UCI Machine Learning Repository | 10,000 rows × 14 columns |

Run `src/data_features.py` to see row counts and the derived-feature / RUL summary.

## Columns (raw)

`UDI`, `Product ID`, `Type` (L/M/H), `Air temperature [K]`, `Process temperature [K]`,
`Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`, `Machine failure`,
and five binary failure-mode flags: `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.

## Derived features (this study)

Computed in `src/data_features.py::add_derived_features`:

| Feature | Formula |
|---|---|
| `Temp_diff` | Process temperature − Air temperature |
| `Temp_ratio` | Process temperature / Air temperature |
| `Power` | Torque × Rotational speed × 2π / 60 (mechanical power, W) |
| `Strain` | Tool wear × Torque (used only to compute the RUL label, not as a model input) |
| `Temp_Speed` | Temp_diff × Rotational speed |
| `Torque_Speed` | Torque × Rotational speed |
| `Temp_diff_norm` | Z-score normalized Temp_diff |

## RUL label — read this before treating it as a benchmark

The RUL label is a **closed-form function of the same physical thresholds**
that define TWF/HDF/PWF/OSF (per machine type L/M/H), and those same columns
are also used as model inputs. See the caveat in `src/data_features.py` —
this label is best understood as "how well can a model reconstruct a known
formula", not a real run-to-failure benchmark.

## Source

- Dataset page: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- Citation: Matzka, S. (2020). Explainable Artificial Intelligence for
  Predictive Maintenance Applications. In *2020 Third International
  Conference on Artificial Intelligence for Industries (AI4I)*, pp. 69–74.
  https://doi.org/10.1109/AI4I49448.2020.00023
- License: CC BY 4.0 (UCI Machine Learning Repository standard license).
