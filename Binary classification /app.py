# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve)

# Step 2: Load the Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
df = pd.read_excel(url, header=1)  # Skip first row (column names)
df.rename(columns={'default payment next month': 'default'}, inplace=True)  # Rename target column

# Step 3: Visualize Class Distribution
plt.figure(figsize=(5, 4))
sns.countplot(x=df['default'], palette='viridis')
plt.title("Distribution of Target Variable (Default Payment)")
plt.xlabel("Default (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# Step 4: Feature Selection
X = df.drop(columns=['ID', 'default'])  # Remove ID and target variable
y = df['default']  # Target variable

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 6: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Initialize and Train Models

# 1. Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# 2. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 3. XGBoost Classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# 4. Support Vector Machine (SVM)
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Step 8: Predictions
models = {
    "Logistic Regression": logistic_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "SVM": svm_model
}

# Step 9: Evaluate Models
results = []
plt.figure(figsize=(10, 7))

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": report['1']['precision'],
        "Recall": report['1']['recall'],
        "F1-score": report['1']['f1-score']
    })

    # Print Model Performance
    print(f"\n{name} Performance:")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    plt.subplot(2, 2, list(models.keys()).index(name) + 1)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")

plt.tight_layout()
plt.show()

# Step 10: Compare Model Performance
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df)

# Step 11: Visualizing Model Performance
plt.figure(figsize=(10, 5))
sns.barplot(x="Model", y="Accuracy", data=results_df, palette="viridis")
plt.title("Model Comparison - Accuracy")
plt.show()

# Step 12: Feature Importance for Random Forest & XGBoost
plt.figure(figsize=(12, 5))

# Random Forest Feature Importance
plt.subplot(1, 2, 1)
rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
rf_importances[:10].plot(kind='bar', color='teal')
plt.title("Top 10 Feature Importance (Random Forest)")

# XGBoost Feature Importance
plt.subplot(1, 2, 2)
xgb_importances = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
xgb_importances[:10].plot(kind='bar', color='orange')
plt.title("Top 10 Feature Importance (XGBoost)")

plt.tight_layout()
plt.show()

# Step 13: ROC Curve Comparison
plt.figure(figsize=(8, 6))
for name, model in models.items():
    y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# Step 14: Precision-Recall Curve
plt.figure(figsize=(8, 6))
for name, model in models.items():
    y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    plt.plot(recall, precision, label=f"{name}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# Step 15: Distribution of Key Financial Features
key_features = ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'PAY_AMT1', 'PAY_AMT2']
df[key_features].hist(figsize=(12, 8), bins=30, color='skyblue')
plt.suptitle("Distribution of Key Financial Features")
plt.show()
