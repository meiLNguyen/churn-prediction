# 🏢 Telco Customer Churn Prediction

> Dự đoán khách hàng rời mạng (churn) bằng XGBoost — project ML end-to-end đầu tiên.
> *(Xóa dòng này và điền nội dung thật sau khi hoàn thành — đây là README bạn sẽ đưa lên GitHub để nhà tuyển dụng đọc!)*

## 🎯 Mục tiêu project
<!-- 3-4 câu: bài toán là gì, vì sao chọn, kết quả đạt được bao nhiêu -->
Xây dựng model phân loại khách hàng có khả năng rời mạng (churn = Yes/No) dựa trên dữ liệu sử dụng dịch vụ.
Mục tiêu: đạt **AUC ≥ 0.85** trên test set — đủ tốt để công ty chủ động giữ chân khách hàng trước khi họ rời đi.

## 📊 Dataset
- Nguồn: Telco Customer Churn (IBM) — dataset kinh điển trên Kaggle
- **7,043 khách hàng · 21 cột** (19 features + customerID + target `Churn`)
- Phân bố target: No = 5,174 (73.5%) · Yes = 1,869 (26.5%) → **hơi mất cân bằng**
- Các nhóm feature: thông tin khách hàng (gender, SeniorCitizen...), dịch vụ đang dùng (InternetService, TechSupport...), hợp đồng & phí (Contract, MonthlyCharges...)

## 🛠️ Quy trình (Pipeline)
```
data/ ──▶ 01_eda.ipynb ──▶ 02_preprocessing ──▶ baseline ──▶ XGBoost ──▶ evaluation
```
1. **EDA** — phân tích phân bố, tương quan feature vs Churn, phát hiện giá trị thiếu (`TotalCharges` có dấu cách)
2. **Preprocessing** — encode categorical, xử lý missing, train/test split (80/20, stratified)
3. **Baseline** — DummyClassifier (majority class) + LogisticRegression
4. **Model** — XGBoost (tune bằng GridSearch/RandomSearch, track bằng MLflow)
5. **Evaluation** — so sánh baseline vs XGBoost trên **AUC, F1, Precision, Recall, confusion matrix**

## 📈 Kết quả
| Model | AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Dummy (majority) | 0.50 | — | — | — |
| Logistic Regression | ? | ? | ? | ? |
| **XGBoost (best)** | **?** | **?** | **?** | **?** |

<!-- Điền số thật sau khi chạy xong — đừng để dấu ? -->

## 🧠 Bài học rút ra
<!-- 3-4 bullet: điều thú vị/khó nhất khi làm project này -->
- *(ví dụ)* `tenure` (thời gian gắn bó) là feature mạnh nhất — khách mới rời nhiều hơn
- *(ví dụ)* Metric quan trọng nhất cho churn là **Recall** — bỏ sót khách sắp rời còn tệ hơn dự đoán nhầm

## 🚀 Cách chạy
```bash
pip install -r requirements.txt
python src/train.py            # train + eval toàn bộ pipeline
jupyter notebook notebooks/01_eda.ipynb   # xem EDA từng bước
```

## 🗂️ Cấu trúc repo
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

## 📬 Liên hệ
<!-- GitHub username của bạn -->
