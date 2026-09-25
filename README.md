# Research on Remaining Useful Life Prediction of Mechanical Equipment Based on Derived Physical Features and Machine Learning

Data processing, model training, and evaluation code for a conference paper on physics-informed RUL prediction with an asymmetric, cost-aware scoring function.

## 1. Paper & Conference

| Item | Detail |
|---|---|
| Title | Research on Remaining Useful Life Prediction of Mechanical Equipment Based on Derived Physical Features and Machine Learning |
| Paper code | DLT2026-SS30-04 |
| Authors | Li-Yang Chou¹ (周理陽), Li-Fu Hsu² |
| Affiliations | ¹Department of Mechanical Engineering, National Central University; ²National Taiwan University of Science and Technology |
| Keywords | RUL, Physical Feature Engineering, Asymmetric Score Function, Predictive Maintenance |

| Conference | Detail |
|---|---|
| Full name | 第26屆數位生活科技研討會 — 26th Symposium on Digital Life Technologies (DLT 2026) |
| Organizers | National Kaohsiung University of Science and Technology, Chia Nan University of Pharmacy and Science, National University of Kaohsiung |
| Date | 22–23 May 2026 |
| Venue | Chia Nan University of Pharmacy and Science, Tainan, Taiwan |
| Official website | https://2026dlt.wixsite.com/home |
| Proof of acceptance | Full paper, abstract, and signed copyright authorization letter (DLT2026-SS30-04) on file with the conference organizer |

### Abstract

Traditional data-driven RUL prediction models often lack engineering interpretability due to a deficiency in physical constraints. This study proposes machine learning models incorporating physical knowledge to enhance prediction accuracy and practicality across multiple failure modes. Using the AI4I 2020 Predictive Maintenance Dataset, raw data is transformed into physically significant derived features, such as mechanical power, strain, and temperature differentials. The research constructs prediction models based on Random Forest and Feed-forward Neural Networks, incorporating an Asymmetric Score function to penalize costly "Late Prediction" risks. Experimental results demonstrate that compared to baseline models, those integrated with physical knowledge not only reduce overall RMSE but also exhibit superior predictive stability during critical end-of-life stages. This approach provides a cost-effective and interpretable maintenance decision-making tool for smart factories.

## 2. Data

UCI Machine Learning Repository — **AI4I 2020 Predictive Maintenance Dataset**: 10,000 synthetic CNC machine records with five documented failure modes (TWF, HDF, PWF, OSF, RNF).

Details, column definitions, and citation: [`data/README_data.md`](data/README_data.md).

## 3. Method

- **Physical feature engineering** (`src/data_features.py`): from the four raw sensor columns (air/process temperature, rotational speed, torque) plus tool wear, derive `Temp_diff`, `Temp_ratio`, `Power` (mechanical power from torque × speed), `Strain` (wear × torque), and two interaction terms.
- **RUL label**: a closed-form, per-machine-type (L/M/H) function of the same physical thresholds that define the four failure modes. **This is a documented data-leakage-by-design label** (see the caveat in `src/data_features.py` and `data/README_data.md`) — it measures how well a model reconstructs a known formula from its own inputs, not generalization to independent run-to-failure sensor data. This is disclosed rather than hidden because it is a genuine property of the dataset/label design, not a bug.
- **Split**: 60/20/20 train/val/test, stratified on binned RUL.
- **Models**: Linear Regression (baseline), Random Forest, Gradient Boosting (`src/baseline_and_ensembles.py`, scikit-learn pipelines with shared preprocessing), and a 3-hidden-layer PyTorch feed-forward network (`src/rul_predictor_nn.py`).
- **Asymmetric Score function**: penalizes "late prediction" (predicted RUL > actual, i.e. a dangerously late warning) more heavily than "early prediction" (predicted RUL < actual, a safer early warning) — `exp(diff/10)-1` for overestimates vs. `exp(-diff/13)-1` for underestimates.
- **Evaluation**: RMSE, MAE, R², the Asymmetric Score, and MAE split by early-stage (RUL > 100) vs. late-stage (RUL ≤ 100) samples, since late-stage accuracy matters most for maintenance decisions.
- **Interpretability**: built-in Random Forest feature importance, and permutation feature importance (PFI) for the neural network (`src/rul_predictor_nn.py`).

## 4. Results

Test-set performance (`results/model_comparison.csv`, `results/pytorch_model_metrics.csv`):

| Model | RMSE | MAE | R² | Asymmetric Score |
|---|---|---|---|---|
| Linear Regression | 3.66 | 2.51 | 0.717 | 0.302 |
| Random Forest | 1.85 | 0.57 | 0.927 | 0.079 |
| Gradient Boosting | 1.91 | 0.63 | 0.923 | 0.087 |
| PyTorch Feed-Forward NN | 1.95 | 0.67 | 0.919 | 0.093 |

Random Forest is marginally the strongest model on every metric in this run; Gradient Boosting and the PyTorch MLP are close behind. All three physics-feature-informed models dramatically outperform the Linear Regression baseline, and every RUL value in this test split is ≤ 100, so "late-stage" MAE equals the overall MAE (a property of this split, not a bug).

Feature importance (both Random Forest's native importances and the neural network's permutation importance, `results/pytorch_pfi.csv`) consistently ranks **Torque**, **Power**, and **Rotational speed** as the most influential features — consistent with the paper's physical-feature framing.

Figures: `figures/fig1_rf_feature_importance.png`, `fig2_actual_vs_predicted_{random_forest,gradient_boosting}.png`, `fig3_model_comparison.png`, `fig4_pytorch_pfi.png`.

## 5. File Structure

```
rul-ai4i2020-physical-features/
├─ data/
│  ├─ ai4i2020.csv              Raw AI4I 2020 dataset
│  └─ README_data.md            Column definitions, derived features, source, license
├─ src/
│  ├─ data_features.py          Physical feature engineering, RUL label, stratified split
│  ├─ baseline_and_ensembles.py Linear Regression / Random Forest / Gradient Boosting + asymmetric scoring
│  ├─ rul_predictor_nn.py       PyTorch feed-forward NN + permutation feature importance
│  └─ make_figures.py           All figures
├─ results/
│  ├─ models/                   Saved joblib pipelines + PyTorch state dict
│  ├─ model_comparison.csv      Linear/RF/GB metrics
│  ├─ pytorch_model_metrics.csv PyTorch NN metrics
│  └─ pytorch_pfi.csv           Permutation feature importance
├─ figures/                     fig1–fig4
├─ references/                  Bibliography (see references/README.md)
└─ requirements.txt
```

## 6. How to Run

```bash
pip install -r requirements.txt

python src/data_features.py           # inspect engineered features + RUL distribution
python src/baseline_and_ensembles.py  # train Linear/RF/GB, write results/model_comparison.csv
python src/rul_predictor_nn.py        # train the PyTorch MLP + permutation feature importance
python src/make_figures.py            # write all figures to figures/
```

## 7. References

See [`references/README.md`](references/README.md) for the full bibliography and licensing notes. Full-text PDFs of third-party papers are not redistributed in this repository.

## 8. License

Code and documentation authored for this project (`src/`, this README) are released under the MIT License — see [`LICENSE`](LICENSE).

The following are **not** covered by that license and remain under their own terms:

- **Dataset** (`data/`): UCI Machine Learning Repository, AI4I 2020 Predictive Maintenance Dataset, CC BY 4.0 — cite Matzka (2020), see `data/README_data.md`.
- **References** (`references/`): copyright of the original authors/publishers — see `references/README.md`.
