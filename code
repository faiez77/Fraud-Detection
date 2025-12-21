import numpy as np;
import pandas as pd;
df=pd.read_csv("creditcard.csv.zip")

  # fraud is 0.17 perecent here accuracy alone is meaningless
df['Class'].value_counts(normalize=True)*100

// pdf analysis
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,5))

sns.kdeplot(df[df['Class']==0]['Amount'], label='Non-Fraud', fill=True)
sns.kdeplot(df[df['Class']==1]['Amount'], label='Fraud', fill=True)

plt.title("PDF of Transaction Amount")
plt.legend()
plt.show()


// cdf analysis
amount_sorted = np.sort(df['Amount'])
cdf = np.arange(len(amount_sorted)) / float(len(amount_sorted))

plt.figure(figsize=(8,5))
plt.plot(amount_sorted, cdf)
plt.xlabel("Transaction Amount")
plt.ylabel("CDF")
plt.title("CDF of Transaction Amount")
plt.show()


# Log-transform skewed features
df['LogAmount'] = np.log(df['Amount'] + 1)

# Drop original Amount
X = df.drop(['Class','Amount'], axis=1)
y = df['Class']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.3, random_state=42)

# Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_train_prob = model.predict_proba(X_train_scaled)[:,1]
y_test_prob  = model.predict_proba(X_test_scaled)[:,1]

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,5))
sns.kdeplot(y_test_prob[y_test==0], label='Non-Fraud', fill=True)
sns.kdeplot(y_test_prob[y_test==1], label='Fraud', fill=True)
plt.title("PDF of Predicted Fraud Probabilities")
plt.xlabel("Predicted Probability of Fraud")
plt.ylabel("Density")
plt.legend()
plt.show()

// threshold optimisation

from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score, roc_auc_score

# Precision-Recall vs Threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_test_prob)

# Confusion Matrix at chosen threshold
chosen_thresh = 0.3
y_pred_thresh = (y_test_prob >= chosen_thresh).astype(int)
cm = confusion_matrix(y_test, y_pred_thresh)
f1 = f1_score(y_test, y_pred_thresh)

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_test_prob)
