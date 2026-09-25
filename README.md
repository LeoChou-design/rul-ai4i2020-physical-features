<a id="zh"></a>

**中文** | [English](#english)

# 以衍生物理特徵與機器學習預測機械設備剩餘壽命

本專案為一篇研討會論文的資料處理、模型訓練與評估程式，主題是結合物理知識的 RUL 預測與「非對稱、考量成本」的評分函數。

## 一、論文與研討會資料

| 項目 | 內容 |
|---|---|
| 論文題目 | Research on Remaining Useful Life Prediction of Mechanical Equipment Based on Derived Physical Features and Machine Learning（以衍生物理特徵與機器學習預測機械設備剩餘壽命，論文以英文撰寫） |
| 稿號 | DLT2026-SS30-04 |
| 作者 | 周理陽¹（Li-Yang Chou）、Li-Fu Hsu² |
| 單位 | ¹國立中央大學機械工程學系；²國立臺灣科技大學 |
| 關鍵字 | RUL、物理特徵工程、非對稱評分函數、預測性維護 |

| 研討會 | 內容 |
|---|---|
| 全名 | 第26屆數位生活科技研討會（26th Symposium on Digital Life Technologies, DLT 2026） |
| 主辦 | 國立高雄科技大學、嘉南藥理大學、國立高雄大學 |
| 日期 | 2026 年 5 月 22 日至 23 日 |
| 地點 | 嘉南藥理大學（臺南市） |
| 官方網站 | https://2026dlt.wixsite.com/home |
| 錄取證明 | 全文、摘要與簽署之著作權授權書（DLT2026-SS30-04）已送交大會 |

### 論文摘要（中文為譯文，原文見下方英文部分）

傳統資料驅動的 RUL 預測模型常因缺乏物理限制而欠缺工程可解釋性。本研究提出結合物理知識的機器學習模型，以提升多種故障模式下的預測準確度與實用性。以 AI4I 2020 預測性維護資料集為對象，將原始資料轉換為具物理意義的衍生特徵，例如機械功率、應變與溫差。研究以隨機森林與前饋神經網路建立預測模型，並納入非對稱評分函數，以懲罰代價高昂的「過晚預測」風險。實驗結果顯示，相較於基準模型，融入物理知識的模型不僅降低整體 RMSE，在關鍵的壽命末期階段也展現更佳的預測穩定性。此方法為智慧工廠提供兼具成本效益與可解釋性的維護決策工具。

## 二、資料

UCI Machine Learning Repository 的 **AI4I 2020 預測性維護資料集**：10,000 筆合成 CNC 機台紀錄，含五種故障模式（TWF、HDF、PWF、OSF、RNF）。

欄位定義與引用方式見 [`data/README_data.md`](data/README_data.md)。

## 三、方法

- **物理特徵工程**（`src/data_features.py`）：由空氣／製程溫度、轉速、扭力與刀具磨損，衍生 `Temp_diff`、`Temp_ratio`、`Power`（扭力 × 轉速換算的機械功率）、`Strain`（磨損 × 扭力）與兩個交互項。
- **RUL 標籤**：依機型（L/M/H）以與 TWF/HDF/PWF/OSF 相同的物理門檻算出的封閉解。**這是刻意標註的「設計上的資料洩漏」標籤**（見 `src/data_features.py` 與 `data/README_data.md` 的說明）——它衡量的是模型從自身輸入還原已知公式的能力，而非對獨立 run-to-failure 感測資料的泛化能力。此點如實揭露，因為這是資料集與標籤設計的固有性質，並非程式錯誤。
- **切分**：60/20/20 訓練／驗證／測試，依分箱後的 RUL 分層。
- **模型**：線性迴歸（基準）、隨機森林、梯度提升（`src/baseline_and_ensembles.py`，共用前處理的 scikit-learn pipeline），以及三層隱藏層的 PyTorch 前饋網路（`src/rul_predictor_nn.py`）。
- **非對稱評分函數**：「過晚預測」（預測 RUL > 實際，警示危險地偏晚）的懲罰比「過早預測」（預測 RUL < 實際，較安全的提早警示）更重——高估用 `exp(diff/10)-1`，低估用 `exp(-diff/13)-1`。
- **評估**：RMSE、MAE、R²、非對稱分數，以及依早期（RUL > 100）／晚期（RUL ≤ 100）分段的 MAE，因為晚期準確度對維護決策最重要。
- **可解釋性**：隨機森林內建特徵重要性，以及神經網路的排列特徵重要性（PFI，`src/rul_predictor_nn.py`）。

## 四、執行結果

測試集表現（`results/model_comparison.csv`、`results/pytorch_model_metrics.csv`）：

| 模型 | RMSE | MAE | R² | 非對稱分數 |
|---|---|---|---|---|
| Linear Regression | 3.66 | 2.51 | 0.717 | 0.302 |
| Random Forest | 1.85 | 0.57 | 0.927 | 0.079 |
| Gradient Boosting | 1.91 | 0.63 | 0.923 | 0.087 |
| PyTorch 前饋網路 | 1.95 | 0.67 | 0.919 | 0.093 |

隨機森林在每項指標上略勝一籌，梯度提升與 PyTorch 網路緊追在後；三個融入物理特徵的模型都大幅優於線性迴歸基準。本測試切分中所有 RUL 值皆 ≤ 100，因此「晚期 MAE」等於整體 MAE（這是此切分的性質，並非錯誤）。

特徵重要性（隨機森林原生重要性與神經網路的排列重要性 `results/pytorch_pfi.csv`）一致地將 **Torque（扭力）**、**Power（功率）** 與 **Rotational speed（轉速）** 列為最具影響力的特徵，與論文的物理特徵設計相符。

圖表：`figures/fig1_rf_feature_importance.png`、`fig2_actual_vs_predicted_{random_forest,gradient_boosting}.png`、`fig3_model_comparison.png`、`fig4_pytorch_pfi.png`。

## 五、檔案結構

```
rul-ai4i2020-physical-features/
├─ data/
│  ├─ ai4i2020.csv              AI4I 2020 原始資料
│  └─ README_data.md            欄位定義、衍生特徵、來源、授權
├─ src/
│  ├─ data_features.py          物理特徵工程、RUL 標籤、分層切分
│  ├─ baseline_and_ensembles.py 線性／隨機森林／梯度提升 + 非對稱評分
│  ├─ rul_predictor_nn.py       PyTorch 前饋網路 + 排列特徵重要性
│  └─ make_figures.py           所有圖表
├─ results/
│  ├─ models/                   儲存的 joblib pipeline 與 PyTorch 權重
│  ├─ model_comparison.csv      線性／RF／GB 指標
│  ├─ pytorch_model_metrics.csv PyTorch 網路指標
│  └─ pytorch_pfi.csv           排列特徵重要性
├─ figures/                     fig1–fig4
├─ references/                  參考文獻（見 references/README.md）
└─ requirements.txt
```

## 六、如何執行

```bash
pip install -r requirements.txt

python src/data_features.py           # 檢視衍生特徵與 RUL 分布
python src/baseline_and_ensembles.py  # 訓練線性／RF／GB，輸出 results/model_comparison.csv
python src/rul_predictor_nn.py        # 訓練 PyTorch 網路與排列特徵重要性
python src/make_figures.py            # 產生所有圖表到 figures/
```

## 七、參考文獻

完整書目與授權說明見 [`references/README.md`](references/README.md)。第三方論文全文不隨本 repo 散布。

## 八、授權

本專案自行撰寫的程式碼（`src/`）與文件以 MIT License 釋出，詳見 [`LICENSE`](LICENSE)。

以下內容不在本授權範圍內，各自沿用原本的條款：

- **資料集**（`data/`）：UCI Machine Learning Repository 的 AI4I 2020 預測性維護資料集，CC BY 4.0，引用 Matzka (2020)，見 `data/README_data.md`。
- **參考文獻**（`references/`）：著作權歸各作者與出版方所有，見 `references/README.md`。

---

<a id="english"></a>

[中文](#zh) | **English**

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
