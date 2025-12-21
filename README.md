# Fraud Detection using Logistic Regression

## Project Overview
This project demonstrates a **fraud detection pipeline** using machine learning and probability concepts.  
The goal is to **predict whether a transaction is fraudulent** based on features from a dataset, while interpreting results from both a **technical** and **business perspective**.

---

## Motivation
Fraudulent transactions are **rare** (≈0.17%), but costly. Detecting fraud involves:
- Understanding data distributions
- Modeling probabilities of fraud
- Balancing detection vs operational cost

This project simulates a **real-world fraud detection scenario**, similar to what banks or payment processors face.

---

## Dataset
- Source: [creditcard.csv] (downloaded from kaggle and used in Jupyter notebook)
- Features:
  - `Time` – seconds since first transaction
  - `Amount` – transaction amount
  - `V1`–`V28` – anonymized PCA components
  - `Class` – target (0 = Non-Fraud, 1 = Fraud)

---

## key concept used:

**PDF & CDF analysis** to understand feature distribution:
PDF: Shows concentration of data points
CDF: Shows cumulative probability

Observations:
Fraud transactions occur in the tail (high Amount)
Features are skewed → log-transform needed

**Predict_proba()** → probability output, not just 0/1

**Scaling** → stabilizes learning, important for gradient-based models[using standardscaler :(mean=0, std=1)]

**Log-transform** → reduces skew, handles heavy-tailed features

**Threshold tuning** → critical for rare events[we are not using default threshold =0.5 ]

**Precision/Recall/F1** → preferred over accuracy for imbalanced datasets

**ROC-AUC** → independent metric to evaluate ranking ability

**Class_weight**='balanced' → handles rare fraud events
