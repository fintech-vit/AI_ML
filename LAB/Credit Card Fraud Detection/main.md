
---

# **Fraud Detection in Credit Card Transactions**  
- https://colab.research.google.com/drive/1SMg7Itc6dLLAAg7vhkEBePhFcFe2aYWL?usp=sharing
---

# **Fraud Detection in Credit Card Transactions**  

## **1. Introduction**  

Credit card fraud is a major concern in financial transactions, costing billions of dollars worldwide. **Fraudulent activities include unauthorized transactions, identity theft, and card skimming.**  

To tackle this issue, **machine learning models** are widely used to detect fraudulent transactions by analyzing patterns in data. In this project, we will:  

- **Load and explore the dataset**
- **Perform Exploratory Data Analysis (EDA)**
- **Handle class imbalance**
- **Train a Logistic Regression model**
- **Evaluate model performance using precision, recall, and F1-score**
- **Discuss future improvements**  

This guide provides a **step-by-step** approach to fraud detection using Python, along with **visualizations and code explanations** to help students understand the process.  

---

## **2. Load and Explore the Dataset**  

### **2.1 Dataset Overview**  

The dataset used for this analysis is the **[Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)** available on Kaggle.  

**Key Features:**  
- `Time`: Time elapsed (in seconds) since the first transaction.  
- `V1, V2, ..., V28`: Anonymized numerical features extracted using PCA (Principal Component Analysis).  
- `Amount`: Transaction amount.  
- `Class`: Target variable indicating fraud (**1**) or non-fraud (**0**).  

---

### **2.2 Load the Data**  

We first import the dataset using **Pandas** and examine its structure.  

```python
import pandas as pd

# Load dataset
df = pd.read_csv("creditcard.csv")

# Display the first few rows
df.head()
```

#### **Output:**  
This will show the first five rows of the dataset, giving an idea of how the data looks.  

---

### **2.3 Check for Missing Values**  

Before analyzing the data, we need to check for missing values.  

```python
print(df.isnull().sum())  # Check for missing values
```

#### **Expected Output:**  
If there are **no missing values**, the output will be **zero for all columns**.  

---

### **2.4 Basic Statistics of the Dataset**  

We now analyze the **summary statistics** of the dataset.  

```python
print(df.describe())  # Summary statistics of numerical features
```

This helps in understanding:  
- **Mean, standard deviation, min, max** values for each feature.  
- **Transaction amount range** (helps in fraud detection).  

---

## **3. Exploratory Data Analysis (EDA)**  

### **3.1 Fraud vs. Genuine Transactions Distribution**  

Since fraudulent transactions are **rare**, we visualize the **class distribution**.  

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.countplot(x=df["Class"], palette=["blue", "red"])
plt.title("Transaction Class Distribution (0: Genuine, 1: Fraud)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()
```

#### **Expected Output:**  
A bar chart showing the imbalance in class distribution (**fraud cases are much fewer than genuine ones**).  

---

### **3.2 Distribution of Transaction Amounts**  

Fraudulent transactions may have **different spending patterns** compared to genuine ones.  

```python
plt.figure(figsize=(8, 5))
sns.boxplot(x="Class", y="Amount", data=df, palette=["blue", "red"])
plt.title("Transaction Amount vs. Fraud Status")
plt.show()
```

#### **Key Observations:**  
- Fraudulent transactions may have **smaller or higher** amounts than genuine ones.  
- Helps in **understanding spending patterns** of fraud cases.  

---

### **3.3 Correlation Heatmap**  

A correlation heatmap helps in understanding relationships between variables.  

```python
# Compute correlation
correlation_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
```

#### **Why is this useful?**  
- Helps identify **strongly related features** for fraud detection.  
- Features with high correlation can be used for **feature selection**.  

---

## **4. Data Preprocessing**  

### **4.1 Handling Class Imbalance**  

Fraud cases are much **fewer** than genuine transactions, making the dataset highly **imbalanced**.  

We use **SMOTE (Synthetic Minority Over-sampling Technique)** to create synthetic fraudulent cases to balance the dataset.  

```python
from imblearn.over_sampling import SMOTE

# Drop rows where 'Class' is NaN
df = df.dropna(subset=["Class"])

# Define features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# Fill missing values in features (if any)
X = X.fillna(X.mean())

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

```

---

## **5. Model Training**  

### **5.1 Splitting Data into Train and Test Sets**  

We split the dataset into **training and testing sets (80%-20%)**.  

```python
from sklearn.model_selection import train_test_split

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
```

---

### **5.2 Train a Logistic Regression Model**  

We use **Logistic Regression**, a simple yet effective classifier.  

```python
from sklearn.linear_model import LogisticRegression

# Initialize model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)
```

---

## **6. Model Evaluation**  

### **6.1 Predicting on Test Data**  

```python
y_pred = model.predict(X_test)
```

---

### **6.2 Performance Metrics: Accuracy, Precision, Recall, F1-Score**  

Since fraud detection is a **classification problem**, we evaluate:  
- **Accuracy**: How often the model is correct.  
- **Precision**: Out of all predicted frauds, how many were actually fraud.  
- **Recall**: Out of all actual fraud cases, how many were detected.  
- **F1-score**: Balance between precision and recall.  

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
```

---

## **Conclusion and Future Improvements**  

### **Summary of Findings**  
- Fraudulent transactions are **rare**, making class imbalance a problem.  
- **SMOTE helped balance** the dataset for better learning.  
- **Logistic Regression** provided a **baseline** model.  

### **Future Improvements**  
- Use **Random Forest, XGBoost, or Neural Networks** for better accuracy.  
- Implement **real-time fraud detection** for faster prevention.  
- Try **unsupervised learning** (autoencoders, anomaly detection) when fraud labels are missing.  

---

## **References**  
- Kaggle: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- Scikit-Learn Documentation: [Machine Learning for Fraud Detection](https://scikit-learn.org/stable/)  
