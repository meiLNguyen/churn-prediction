# Telco Customer Churn Prediction

End-to-end machine learning project to predict customer churn (customers leaving the service) using XGBoost, with MLflow experiment tracking and decision-threshold tuning.

## 🎯 Project Goal

Build a binary classifier that predicts whether a customer will churn (`Churn` = Yes/No) based on service usage data (7,043 customers, 21 features). The model helps a telco **proactively retain at-risk customers** before they leave. Final result: **AUC 0.84** and — after threshold tuning — **Recall 0.72**, catching ~72% of would-be churners.

## 📊 Dataset

- **Source:** [Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — classic benchmark dataset on Kaggle
- **Size:** 7,043 customers · 21 columns (19 features + `customerID` + target `Churn`)
- **Target distribution:** No = 5,174 (73.5%) · Yes = 1,869 (26.5%) → **imbalanced** — accuracy is misleading, so we evaluate with **AUC / Recall / F1**
- **Feature groups:** customer info (`gender`, `SeniorCitizen`, ...), subscribed services (`InternetService`, `TechSupport`, ...), contract & billing (`Contract`, `MonthlyCharges`, ...)

## 🛠️ Pipeline

```
data/ ──▶ 01_eda.ipynb ──▶ 02_preprocessing+baseline ──▶ 03_xgboost+mlflow ──▶ 04_threshold_tuning
```

1. **EDA** — distribution analysis, feature vs. Churn correlation, data-quality fix (`TotalCharges` contains whitespace-only rows → `pd.to_numeric(errors="coerce")`)
2. **Preprocessing** — binary + one-hot encoding, median-fill missing `TotalCharges` (11 rows, all `tenure == 0`), 80/20 stratified train/test split
3. **Baseline** — `DummyClassifier` (majority class) + `LogisticRegression` (scaled, pipeline)
4. **Model** — XGBoost, 4 configs tuned on a **validation split** (test set touched only once), every run tracked in **MLflow**
5. **Decision rule** — threshold tuned on validation (max F1 with Recall ≥ 0.70) → chosen **0.35** instead of default 0.5
6. **Evaluation** — AUC, F1, Precision, Recall, confusion matrix on the untouched test set (1,409 customers)

## 📈 Results (test set)

| Model | Accuracy | AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Dummy (majority) | 0.7346 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 0.8070 | 0.8416 | 0.6584 | 0.5668 | 0.6092 |
| XGBoost (best) | 0.8062 | 0.8433 | 0.6700 | 0.5321 | 0.5931 |
| **XGBoost + tuned threshold (0.35)** | **0.7771** | **0.8410** | **0.5630** | **0.7166** | **0.6306** |

**Why the tuned row matters:** within the notebook-04 fit (trained on `train` only), the same model catches **71.7% of churners (268/374)** at threshold 0.35 instead of **50.5%** at 0.5 — the gain comes from the decision rule, not a bigger model.

> **Two model fits appear in the table above, on purpose.** The `XGBoost (best)` row is the final model refit on `train + val` (notebook 03). The tuned-threshold row is the notebook-04 fit on `train` only, whose sweep baseline at threshold 0.5 is Recall 0.5053. Threshold tuning is only interpretable against the *same* fit — which is why the pair quoted below is 0.5053 → 0.7166, not 0.5321 → 0.7166.

## 🧠 Key Takeaways

- **`tenure` is the strongest feature** — new customers churn far more than long-standing ones (confirmed by EDA + XGBoost feature importance).
- **Imbalanced data → metric choice matters**: the Dummy baseline "scores" 73.5% accuracy but AUC = 0.5 (random) and Recall = 0. We deliberately used AUC/Recall/F1.
- **A strong baseline is hard to beat**: Logistic Regression already reached AUC 0.84; XGBoost only edged it out (+0.002). The big win came from **threshold tuning** — Recall 0.5053 → 0.7166.
- **ML engineers ship a decision, not a probability**: for churn, missing an at-risk customer (Recall) is costlier than a false alarm (Precision), so threshold 0.35 is the business-aligned choice.
- **Experiments are reproducible**: every config, metric and the final model are logged in `mlruns/` (MLflow).

## 🚀 How to Run

```bash
# 1. Create environment
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt

# 2. Launch Jupyter Lab (Windows — clean PYTHONPATH + correct venv kernel)
./start_jupyter.sh

# 3. Run notebooks in order (kernel: Python (churn-venv))
notebooks/01_eda.ipynb
notebooks/02_preprocessing_baseline.ipynb
notebooks/03_xgboost_mlflow.ipynb
notebooks/04_threshold_tuning.ipynb

# 4. Browse experiment history
.venv/Scripts/mlflow ui
```

## 🗂️ Repo Structure

```
churn-prediction/
├── data/Telco-Customer-Churn.csv          # git-ignored (see README link to download)
├── notebooks/
│   ├── 01_eda.ipynb                        # exploratory data analysis
│   ├── 02_preprocessing_baseline.ipynb     # encoding + Dummy/LR baselines
│   ├── 03_xgboost_mlflow.ipynb             # XGBoost tuning + MLflow tracking
│   └── 04_threshold_tuning.ipynb           # decision-threshold optimization
├── start_jupyter.sh                        # clean-PYTHONPATH launcher (Windows)
├── requirements.txt
├── .gitignore
└── README.md
```

## 📬 Contact

[meiLNguyen](https://github.com/meiLNguyen)
