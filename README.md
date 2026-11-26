# Machine Learning Assignment-2
## Name: Soorya GS
## BITS-ID: 2026CT05109
# Income Prediction Using Machine Learning Classification Models

## 1. Problem Statement
The objective of this project is to build, evaluate, and compare multiple supervised machine learning classification models to predict whether an individual's annual income exceeds **$50,000** (`>50K`) or is less than or equal to **$50,000** (`<=50K`) based on demographic, educational, and employment attributes. 

Accurate income classification supports socioeconomic policy analysis, financial risk assessment, and targeted resource allocation. Because real-world census data contains missing entries, mixed data types, and class imbalance, this pipeline implements leak-free data preprocessing and evaluates models across six complementary performance metrics—including AUC and Matthews Correlation Coefficient (MCC)—to ensure robust evaluation beyond simple accuracy.

---

## 2. Dataset Description
* **Dataset Name:** Adult Census Income Dataset
* **Source:** UCI Machine Learning Repository / Kaggle (`adult.csv`)
* **Dataset Justification:** Selected because it exceeds all minimum complexity requirements (32,561 raw instances and 14 predictive features), represents a realistic tabular classification problem with mixed numerical and categorical attributes, contains missing values requiring principled imputation, and exhibits natural class imbalance.
* **Instances & Features:**
  * **Raw Instances:** 32,561 rows (32,537 rows remaining after removing 24 exact duplicates)
  * **Predictive Features:** 14 features (exceeding the minimum requirement of 12)
  * **Target Variable:** `income` (Binary classification: `<=50K` mapped to `0`, `>50K` mapped to `1`)

### Feature Types
* **Numerical Features (6):** `age`, `fnlwgt` (final sampling weight), `education.num`, `capital.gain`, `capital.loss`, `hours.per.week`
* **Categorical Features (8):** `workclass`, `education`, `marital.status`, `occupation`, `relationship`, `race`, `sex`, `native.country`

### Missing Values & Class Imbalance Analysis
* **Missing Values:** Missing entries were encoded as `'?'` across three categorical columns: `workclass` (1,836 missing), `occupation` (1,843 missing), and `native.country` (583 missing). These were converted to `NaN` and imputed using the training-set mode.
* **Class Distribution:** The dataset shows a moderate ~3:1 class imbalance:
  * **Class `0` (`<=50K`):** 75.91% (24,698 instances)
  * **Class `1` (`>50K`):** 24.09% (7,839 instances)
  * Because of this imbalance, **MCC**, **F1 Score**, and **AUC** are emphasized alongside Accuracy.

---

## 3. GitHub Repository Link
* **Repository URL:** https://github.com/2026ct05109-svg/ML-Assignment-2

### Repository Structure
```text
project-folder/
├── requirements.txt
├── README.md
├── test_data.csv
├── metrics_comparison_bar.png
├── metrics_comparison_heatmap.png
└── model/
    ├── train_and_evaluate.py
    ├── preprocessor.joblib
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest_ensemble.joblib
    └── evaluation_metrics.csv
