# Telco Customer Churn Prediction

End-to-end machine learning project to predict customer churn (customers leaving the service) using XGBoost, with MLflow experiment tracking.

<!-- Replace this with your real results after training. This README is the first thing recruiters read! -->

## 🎯 Project Goal

<!-- 3-4 sentences: what problem, why this dataset, what result you achieved -->

Build a binary classifier that predicts whether a customer will churn (`Churn` = Yes/No) based on service usage data. Target: **AUC ≥ 0.85** on the test set — good enough for a company to proactively retain at-risk customers before they leave.

## 📊 Dataset

- **Source:** [Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — classic benchmark dataset on Kaggle
- **Size:** 7,043 customers · 21 columns (19 features + `customerID` + target `Churn`)
- **Target distribution:** No = 5,174 (73.5%) · Yes = 1,869 (26.5%) → slightly imbalanced
- **Feature groups:** customer info (`gender`, `SeniorCitizen`, ...), subscribed services (`InternetService`, `TechSupport`, ...), contract & billing (`Contract`, `MonthlyCharges`, ...)

## 🛠️ Pipeline

```
data/ ──▶ 01_eda.ipynb ──▶ 02_preprocessing ──▶ baseline ──▶ XGBoost ──▶ evaluation
```

1. **EDA** — distribution analysis, feature vs. Churn correlation, missing-value detection (note: `TotalCharges` contains whitespace)
2. **Preprocessing** — categorical encoding, missing-value handling, 80/20 stratified train/test split
3. **Baseline** — `DummyClassifier` (majority class) + `LogisticRegression`
4. **Model** — XGBoost, tuned with GridSearch/RandomSearch, tracked with MLflow
5. **Evaluation** — baseline vs. XGBoost on **AUC, F1, Precision, Recall, confusion matrix**

## 📈 Results

| Model | AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Dummy (majority) | 0.50 | — | — | — |
| Logistic Regression | ? | ? | ? | ? |
| **XGBoost (best)** | **?** | **?** | **?** | **?** |

<!-- Fill in real numbers after training — don't leave the ? marks -->

## 🧠 Key Takeaways

<!-- 3-4 bullets: most interesting / hardest part of this project -->
- *(example)* `tenure` (length of customer relationship) is the strongest feature — new customers churn more
- *(example)* The most important metric for churn is **Recall** — missing a soon-to-churn customer is worse than a false alarm

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/train.py          # train + evaluate the full pipeline
jupyter notebook notebooks/01_eda.ipynb   # step-by-step EDA
```

## 🗂️ Repo Structure

```
churn-prediction/
├── data/Telco-Customer-Churn.csv
├── notebooks/01_eda.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

## 📬 Contact

<!-- Your GitHub username / email -->
