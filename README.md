# Credit Card Fraud Detection 🚨💳

## Overview

This project detects fraudulent credit card transactions using machine learning classification models on a highly imbalanced dataset. The primary goal is to identify frauds accurately while minimizing false positives to protect customers and reduce financial loss.

---

## Dataset

- The dataset contains transactions made by European cardholders in September 2013, including **284,807 transactions** over two days, with **492 fraudulent transactions (0.172%)**.
- All features are numerical, with `V1` to `V28` as PCA-transformed components.
- `Time` (seconds elapsed from the first transaction) and `Amount` are the original, untransformed features.
- `Class` is the target variable, where `1` indicates fraud and `0` indicates legitimate transactions.

**Dataset Link:**

```python
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

## Approach

* Exploratory Data Analysis (EDA): Analyzed class imbalance, transaction amounts, and time distributions.
* Preprocessing: Scaled Amount and Time, handled class imbalance using class weighting and SMOTE.
* Modeling: Trained Random Forest and LightGBM classifiers.
* Evaluation: Used Precision, Recall, F1-score, and AUC-PR for meaningful evaluation under severe class imbalance.

## Results

| Metric    | Legitimate (0) | Fraudulent (1) |
| --------- | -------------- | -------------- |
| Precision | 0.93           | 0.98           |
| Recall    | 0.98           | 0.93           |
| F1-Score  | 0.96           | 0.95           |

* Overall Accuracy: 95%
* The model demonstrates a high ability to detect fraudulent transactions while minimizing false alarms on legitimate transactions.


## Tools & Libraries
* Python (Pandas, NumPy, Scikit-learn)
* LightGBM
* Matplotlib, Seaborn

## Key Learnings

* Handling imbalanced datasets using class weights and oversampling techniques.
* Building and tuning classification models for fraud detection.
* Using meaningful evaluation metrics for imbalanced classification.
* Working with PCA-transformed features in real-world datasets.

## Future Improvements

* Experiment with anomaly detection methods (Isolation Forest, Autoencoders).
* Deploy as a streamlit demo for live transaction testing.
* Create an interactive dashboard (Power BI or Plotly Dash) for visual monitoring.


## Author
Ashok Kumar N

Feel free to connect on LinkedIn for discussions and collaborations!




