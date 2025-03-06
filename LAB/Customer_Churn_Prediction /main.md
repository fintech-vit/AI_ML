# **Customer Churn Prediction: A Step-by-Step Guide**  

## **📌 Introduction**
Customer churn is a major concern for businesses, especially in subscription-based models. In this guide, we will:
✅ Load and explore customer data  
✅ Analyze churn behavior using **data visualization**  
✅ Train a **Machine Learning model** to predict churn  
✅ Evaluate the model's performance  

---

## **📌 Step 1: Import Required Libraries**
We start by importing essential Python libraries.

```python
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler, LabelEncoder  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  
```

💡 **Why these libraries?**
- **NumPy & Pandas** → Data manipulation  
- **Matplotlib & Seaborn** → Data visualization  
- **Scikit-learn (sklearn)** → Machine learning tools  

---

## **📌 Step 2: Load and Explore the Dataset**
We will use the **Telco Customer Churn dataset** from Kaggle.

```python
# Load dataset
df = pd.read_csv("customer_churn.csv")

# Display first 5 rows
print(df.head())
```

### **Dataset Overview**
The dataset contains customer details like:
- `gender`, `SeniorCitizen`, `Partner`, `Dependents` → Demographics  
- `tenure`, `MonthlyCharges`, `TotalCharges` → Account details  
- `Contract`, `PaymentMethod` → Subscription details  
- `Churn` → **Target variable** (Yes = Customer left, No = Customer stayed)  

---

## **📌 Step 3: Check for Missing Values**
```python
# Check missing values
print(df.isnull().sum())
```

💡 **Handling Missing Data**:
If any missing values exist, we fill them:
```python
df.fillna(df.mean(), inplace=True)
```

---

## **📌 Step 4: Data Visualization**
### **1️⃣ Churn Distribution**
```python
plt.figure(figsize=(6,4))
sns.countplot(x=df["Churn"], palette="coolwarm")
plt.title("Customer Churn Distribution")
plt.show()
```

💡 **Interpretation**:
- We check if churn is **balanced or imbalanced**.  
- If the dataset is highly imbalanced, we will use **SMOTE** later.  

---

### **2️⃣ Churn vs. Monthly Charges**
```python
plt.figure(figsize=(8,5))
sns.histplot(df[df["Churn"] == "Yes"]["MonthlyCharges"], kde=True, color="red", label="Churned")
sns.histplot(df[df["Churn"] == "No"]["MonthlyCharges"], kde=True, color="green", label="Stayed")
plt.legend()
plt.title("Monthly Charges Distribution by Churn")
plt.show()
```

💡 **Interpretation**:
- Customers with **higher monthly charges** tend to churn more.  

---

### **3️⃣ Tenure vs. Churn**
```python
plt.figure(figsize=(8,5))
sns.histplot(df[df["Churn"] == "Yes"]["tenure"], kde=True, color="red", label="Churned")
sns.histplot(df[df["Churn"] == "No"]["tenure"], kde=True, color="green", label="Stayed")
plt.legend()
plt.title("Tenure Distribution by Churn")
plt.show()
```

💡 **Interpretation**:
- **New customers (low tenure) churn more**.  
- This insight can help improve customer **onboarding strategies**.  

---

### **4️⃣ Churn by Contract Type**
```python
plt.figure(figsize=(8,5))
sns.countplot(x=df["Contract"], hue=df["Churn"], palette="coolwarm")
plt.title("Churn Rate by Contract Type")
plt.show()
```

💡 **Interpretation**:
- **Month-to-month contracts** have the highest churn.  
- Businesses can encourage customers to **switch to annual contracts**.  

---

## **📌 Step 5: Data Preprocessing**
### **Convert Categorical Features to Numeric**
We encode categorical columns using **Label Encoding**.

```python
le = LabelEncoder()

for column in df.select_dtypes(include="object").columns:
    df[column] = le.fit_transform(df[column])
```

💡 **Why?**
- ML models cannot handle text, so we convert it to numbers.

---

### **Feature Scaling**
We scale **continuous variables** like `MonthlyCharges` and `TotalCharges`.

```python
scaler = StandardScaler()
df[["MonthlyCharges", "TotalCharges"]] = scaler.fit_transform(df[["MonthlyCharges", "TotalCharges"]])
```

---

## **📌 Step 6: Train-Test Split**
We split data into **80% training** and **20% testing**.

```python
# Define features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

💡 **Stratified Sampling** ensures the same churn ratio in train & test sets.

---

## **📌 Step 7: Train Machine Learning Model**
We use a **Random Forest Classifier**.

```python
# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
```

---

## **📌 Step 8: Model Evaluation**
### **1️⃣ Confusion Matrix**
```python
# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

💡 **Interpretation**:
- **True Positives (TP)** → Correctly predicted churned customers.  
- **False Negatives (FN)** → Churned customers predicted as stayed (bad for business!).  

---

### **2️⃣ Precision, Recall, F1-score**
```python
print(classification_report(y_test, y_pred))
```

💡 **Key Metrics**:
- **Precision** → Correct churn predictions.  
- **Recall** → How many actual churns did we capture?  

---

### **3️⃣ Accuracy Score**
```python
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
```

---

## **📌 Step 9: Handling Imbalanced Data**
Since churn is rare, we apply **SMOTE (Synthetic Minority Over-sampling Technique).**

```python
from imblearn.over_sampling import SMOTE  

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {len(X_train)}")
print(f"After SMOTE: {len(X_resampled)}")
```

---

## **📌 Step 10: Re-train the Model on Balanced Data**
```python
# Train on SMOTE data
model_smote = RandomForestClassifier(n_estimators=100, random_state=42)
model_smote.fit(X_resampled, y_resampled)

# Predict and evaluate
y_pred_smote = model_smote.predict(X_test)
print(classification_report(y_test, y_pred_smote))
```

---

## **📌 Step 11: Feature Importance Analysis**
```python
importances = model.feature_importances_
feature_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x=feature_df["Importance"], y=feature_df["Feature"], palette="coolwarm")
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("Top Features Influencing Churn")
plt.show()
```

💡 **Key Insights**:
- **Tenure, MonthlyCharges, and Contract Type** play the biggest role in churn.  

---

## **🚀 Conclusion**
✅ **Key Takeaways**:
- Customers with **high monthly charges** churn more.  
- **New customers** are at higher risk of churn.  
- **Long-term contracts reduce churn**.  

🚀 **Next Steps**:
- Deploy the model for **real-time churn prediction**.  
- Use **Deep Learning (LSTMs) for better predictions**.  

Would you like me to **implement a deep learning approach**? 🚀