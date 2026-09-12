"""
Payment Transaction Fraud Detection — Logistic Regression + Probability Analysis
==================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve
)

plt.rcParams['figure.dpi'] = 150
np.random.seed(42)

# =======================================================================

# 1. LOAD & CLEAN

df = pd.read_csv("data/bs140513_032310.csv")
for col in ['customer', 'age', 'gender', 'zipcodeOri', 'merchant', 'zipMerchant', 'category']:
    df[col] = df[col].str.strip("'")

print("Shape:", df.shape)
fraud_rate = df['fraud'].mean() * 100
print(f"Fraud rate: {fraud_rate:.4f}%")
print("Unique zipcodeOri:", df['zipcodeOri'].nunique(), "| Unique zipMerchant:", df['zipMerchant'].nunique())

# =======================================================================
# 2. EDA — fraud rate by category is the standout signal in this dataset

cat_fraud = df.groupby('category')['fraud'].agg(['count', 'mean']).sort_values('mean', ascending=False)
print("\nFraud rate by merchant category:\n", cat_fraud)

plt.figure(figsize=(9, 6))
cat_fraud_sorted = cat_fraud.sort_values('mean')
plt.barh(cat_fraud_sorted.index, cat_fraud_sorted['mean'] * 100, color='#4C72B0')
plt.xlabel("Fraud rate (%)")
plt.title("Fraud Rate by Merchant Category")
plt.tight_layout()
plt.savefig("fraud_rate_by_category.png")
plt.close()
print("Saved: fraud_rate_by_category.png")

plt.figure(figsize=(10, 5))
sns.kdeplot(df[df['fraud'] == 0]['amount'], label='Non-Fraud', fill=True)
sns.kdeplot(df[df['fraud'] == 1]['amount'], label='Fraud', fill=True)
plt.title("PDF of Transaction Amount")
plt.xlabel("Amount")
plt.xlim(0, 500)
plt.legend()
plt.tight_layout()
plt.savefig("pdf_transaction_amount.png")
plt.close()
print("Saved: pdf_transaction_amount.png")

# =======================================================================
# 3. FEATURE ENGINEERING

df['LogAmount'] = np.log(df['amount'] + 1)
X = df.drop(['fraud', 'amount', 'customer', 'merchant', 'zipcodeOri', 'zipMerchant', 'step'], axis=1)
y = df['fraud']
X = pd.get_dummies(X, columns=['category', 'age', 'gender'], drop_first=True)
print("\nFeature count after encoding:", X.shape[1])

# =======================================================================
# 4.TRAIN/TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =======================================================================
# 5. MODEL

model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_test_prob = model.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_test_prob)
print(f"\nROC-AUC: {roc_auc:.4f}")

plt.figure(figsize=(10, 5))
sns.kdeplot(y_test_prob[y_test == 0], label='Non-Fraud', fill=True)
sns.kdeplot(y_test_prob[y_test == 1], label='Fraud', fill=True)
plt.title("PDF of Predicted Fraud Probabilities")
plt.xlabel("Predicted Probability of Fraud")
plt.legend()
plt.tight_layout()
plt.savefig("pdf_predicted_probabilities.png")
plt.close()
print("Saved: pdf_predicted_probabilities.png")

# =======================================================================
# 7. THRESHOLD TUNING (grid search, F1-optimal, same method as the

print("\n--- Default threshold (0.5) ---")
print(classification_report(y_test, (y_test_prob >= 0.5).astype(int), digits=3))

grid = np.arange(0.01, 1.00, 0.01)
grid_results = pd.DataFrame([
    {
        "threshold": t,
        "precision": precision_score(y_test, (y_test_prob >= t).astype(int), zero_division=0),
        "recall": recall_score(y_test, (y_test_prob >= t).astype(int)),
        "f1": f1_score(y_test, (y_test_prob >= t).astype(int)),
    }
    for t in grid
])
best_row = grid_results.loc[grid_results['f1'].idxmax()]
best_threshold = best_row['threshold']

print(f"\n--- Tuned threshold ({best_threshold:.2f}) ---")
y_pred_best = (y_test_prob >= best_threshold).astype(int)
print(classification_report(y_test, y_pred_best, digits=3))
cm = confusion_matrix(y_test, y_pred_best)
print("Confusion matrix:\n", cm)

plt.figure(figsize=(9, 5))
plt.plot(grid_results['threshold'], grid_results['precision'], label='Precision')
plt.plot(grid_results['threshold'], grid_results['recall'], label='Recall')
plt.plot(grid_results['threshold'], grid_results['f1'], label='F1', linewidth=2)
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Chosen threshold ({best_threshold:.2f})')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision / Recall / F1 vs Threshold")
plt.legend()
plt.tight_layout()
plt.savefig("threshold_tuning.png")
plt.close()
print("Saved: threshold_tuning.png")

fpr, tpr, _ = roc_curve(y_test, y_test_prob)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()
print("Saved: roc_curve.png")

# =======================================================================
# 8. BUSINESS IMPACT

tn, fp, fn, tp = cm.ravel()
print(f"\n--- Business impact at threshold {best_threshold:.2f} ---")
print(f"Fraud caught: {tp} / {tp + fn} ({tp/(tp+fn)*100:.1f}%)")
print(f"False alarms: {fp}")
print(f"Fraud missed: {fn}")

results = pd.DataFrame({
    'category': df.loc[X_test.index, 'category'].values,
    'LogAmount': X_test['LogAmount'].values,
    'actual_fraud': y_test.values,
    'fraud_probability': y_test_prob,
    'flagged': y_pred_best
})
results.to_csv("fraud_predictions.csv", index=False)
print("\nSaved: fraud_predictions.csv")
