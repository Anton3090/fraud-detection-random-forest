# 💳 Credit Card Fraud Detection using Random Forest

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.8+-blue)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue?logo=kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)


## 📌 Overview

This project applies a **Random Forest Classifier** to detect fraudulent credit card transactions using the **Credit Card Fraud Detection** dataset provided by ULB on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It includes data preprocessing, model training, evaluation, and visualizations.

---

## 📁 Dataset Description

- The dataset contains **284,807** transactions with **30 features**:
  - 28 anonymized PCA components: `V1` to `V28`
  - `Time`: Seconds elapsed between transaction and first transaction
  - `Amount`: Transaction amount
  - `Class`: Target variable (0 = Legit, 1 = Fraud)

- **Highly imbalanced**:
  - Normal: 284,315
  - Fraudulent: 492

---

## 📊 Exploratory Data Analysis (EDA)

### 1. 🔢 Class Distribution
- Strong class imbalance visualized using a bar plot.

### 2. 💰 Amount Distribution
- Separate histograms for fraud and non-fraud transactions.

### 3. 🧊 Correlation Heatmaps
- Correlation between features and target `Class`.
- Full Pearson correlation heatmap for all features.

### 4. 📦 Boxplots
- Boxplots (e.g. `V14` vs `Class`) reveal significant outliers and feature separation.

### 5. 🕒 Time Analysis
- Frequency of transactions per hour to identify patterns.

### 6. 🧬 PCA Visualization
- 2D PCA scatter plot shows visual separability between fraud and normal transactions.

---

## ⚙️ Preprocessing

- Scaled `Amount` using `StandardScaler`.
- Dropped `Time` and `Class` columns from features.
- Performed stratified train-test split to preserve fraud ratio.

```python
X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']
````

---

## 🤖 Model: Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rfc.fit(X_train, y_train)
```

* `class_weight='balanced'` used to handle class imbalance.
* 100 decision trees with default hyperparameters.

---

## 🧪 Model Evaluation

* **Accuracy:** `99.94%`
* **ROC AUC Score:** `0.9376`
* **Precision (Fraud):** `0.97`
* **Recall (Fraud):** `0.70`

### 📉 Confusion Matrix

```
[[85292     3]
 [   44   104]]
```

### 📋 Classification Report

```text
Class 0 - Precision: 1.00, Recall: 1.00
Class 1 - Precision: 0.97, Recall: 0.70
```

---

## 🔍 Feature Importance

Top features identified by Random Forest:

* `V17`, `V14`, `V12`, `V10`

Visualized using a horizontal bar chart.

---

## 📦 Folder Structure

```bash
creditcard-fraud-detection/
│
├── notebook.ipynb              # Complete Kaggle notebook
├── README.md                   # This file
├── images/                     # Visualizations
└── requirements.txt            # Python package dependencies
```

---

## ✅ Key Takeaways

* Random Forest performs well with high-dimensional, imbalanced datasets.
* Features `V14`, `V17`, `V12`, `V10` show strong predictive power.
* Potential improvements: **SMOTE**, **undersampling**, **anomaly detection**.

---

## 🚀 Future Work

* Compare with models: **XGBoost**, **LightGBM**
* Automate with **AutoML pipelines**
* Perform hyperparameter tuning using **GridSearchCV**

---

## 🧪 Dependencies

```txt
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

## 📜 License

MIT License © 2025 Anton Atef

---

## 🤝 Contributions

Feel free to fork this repo and submit pull requests. Suggestions and improvements are welcome!

---

## 📬 Contact

For questions, reach out via GitHub Issues or email: tony.atef.954@gmail.com

---
