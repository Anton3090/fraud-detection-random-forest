# 💳 Credit Card Fraud Detection using Random Forest + SMOTE

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.8+-blue)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue?logo=kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 📌 Overview

🚀 This project detects **fraudulent credit card transactions** using a **Random Forest Classifier** enhanced with **SMOTE** (Synthetic Minority Over-sampling Technique).  
📊 Built on the [Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), it includes preprocessing, resampling, model training, threshold tuning, evaluation, and feature importance analysis.

---

## 📁 Dataset Description

📦 **Records**: 284,807 transactions  
🧬 **Features**:
- `V1`–`V28`: PCA-anonymized features
- `Amount`: Transaction amount (scaled)
- `Time`: Seconds since first transaction (dropped)
- `Class`: Target (0 = Legit ✅, 1 = Fraud ❌)

⚠️ **Class Imbalance**:
- Legit: 284,315 🟢
- Fraud: 492 🔴

---

## 📊 Exploratory Data Analysis (EDA)

📉 Visualized:
- Class distribution
- Amount distribution by class
- Correlation heatmaps (features vs `Class`)
- Boxplots (e.g., `V14` vs `Class`)
- Hourly frequency of transactions
- 2D PCA scatter plot

---

## ⚙️ Data Preprocessing & SMOTE

### 🔄 Preprocessing

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df['Amount'] = StandardScaler().fit_transform(df[['Amount']])
X = df.drop(['Time', 'Class'], axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
```

### 🧪 Apply SMOTE

```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
```

---

## 🌲 Model: Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rfc.fit(X_train_resampled, y_train_resampled)
```

🧠 Trained with `class_weight='balanced'` to handle imbalance  
🌐 100 decision trees used

---

## 🎯 Threshold Tuning for Better Recall

```python
from sklearn.metrics import precision_recall_curve

y_proba = rfc.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

optimal_threshold = 0.3
y_pred_adjusted = (y_proba > optimal_threshold).astype(int)
```

🎚️ Adjusted threshold from default (0.5) to improve fraud recall

---

## 🧪 Evaluation Metrics

| 🔍 Metric           | 📈 Value |
|--------------------|----------|
| Accuracy            | 99.75%   |
| ROC AUC Score       | 0.99     |
| Precision (Fraud)   | ~0.87    |
| Recall (Fraud)      | ~0.78    |
| F1-Score (Fraud)    | ~0.82    |

🧾 **Confusion Matrix**:
```
[[85268    27]
 [   32   116]]
```

📋 **Classification Report**:
- Class 0: Precision = 1.00, Recall = 1.00
- Class 1: Precision = 0.87, Recall = 0.78

---

## 🌟 Feature Importance

📌 Most important features by Random Forest:

- `V17`, `V14`, `V12`, `V10`

📊 Visualized with horizontal bar plot

---

## 📂 Project Structure

```
creditcard-fraud-detection/
│
├── notebook.ipynb              # 🔍 Full implementation and analysis
├── README.md                   # 📘 Project overview
├── images/                     # 🖼️ Visuals and plots
└── requirements.txt            # 📦 Python dependencies
```

---

## ✅ Key Takeaways

✔️ Random Forest + SMOTE = Powerful combo for imbalanced fraud detection  
📈 Threshold tuning improves recall for fraud cases  
📊 Features `V14`, `V17`, `V12`, and `V10` are highly informative  
💡 Easy to interpret, scalable, and reproducible

---

## 🛠️ Future Improvements

📌 Try alternative models:
- XGBoost 🌲
- LightGBM ⚡
- Logistic Regression 📈

🧪 Add:
- GridSearchCV for hyperparameter tuning  
- Real-time deployment using Flask / Gradio / Streamlit

---

## 📦 Dependencies

```txt
numpy
pandas
matplotlib
seaborn
scikit-learn
imblearn
```

---

## 📜 License

MIT License © 2025 Anton Atef

---

## 🤝 Contributions

👨‍💻 Feel free to fork, clone, and submit pull requests!  
📬 Suggestions and issues are welcome anytime!

---

## 📬 Contact

📧 Email: [tony.atef.954@gmail.com](mailto:tony.atef.954@gmail.com)

---
