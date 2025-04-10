# Customer Churn Prediction
- https://colab.research.google.com/drive/1Qs1Q2N1KyvPnxOFTXqSVkKoWgygVEjPP#scrollTo=22ZePwEJFQ_M

## **1. Introduction**  
Customer churn refers to when a customer stops using a company's services. Predicting churn helps businesses retain customers by identifying the key factors that lead to customer attrition. In this project, we analyze a dataset containing customer demographics, service details, and payment history to recognize patterns in churn behavior using machine learning techniques.

---

## **2. Understanding the Dataset**  
### **2.1 Dataset Overview**  
The dataset used in this analysis is the [Customer Churn Dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset) from Kaggle. It contains information on customer demographics, account details, and service usage patterns, along with a target variable (`Churn`) that indicates whether a customer has left the company.

### **2.2 Key Features**  
- **Customer ID**: Unique identifier for each customer.  
- **Age**: Customer's age.  
- **Gender**: Male or Female.  
- **Tenure**: Number of months the customer has been with the company.  
- **Support Calls**: Number of support calls made by the customer.  
- **Payment Delay**: Number of days payment was delayed.  
- **Subscription Type**: The type of subscription the customer has.  
- **Contract Length**: Length of the contract in months.  
- **Total Spend**: The total amount spent by the customer.  
- **Last Interaction**: Number of days since last interaction.  
- **Churn**: The target variable (Yes/No) indicating whether the customer has churned.  

---

## **3. Loading and Exploring the Dataset**  
### **3.1 Load the Data**  
We start by importing the dataset using Pandas.

```python
import pandas as pd

# Load dataset
df_train = pd.read_csv("customer_churn_dataset-training-master.csv")
df_test = pd.read_csv("customer_churn_dataset-testing-master.csv")

# Display the first few rows
df_train.head()
```

### **3.2 Checking for Missing Values**  
Missing values can impact model performance, so we check for any missing data in the dataset.

```python
print(df_train.isnull().sum())
```

### **3.3 Summary Statistics**  
Understanding the numerical distribution of features helps identify trends and potential issues.

```python
print(df_train.describe())
```

---

## **4. Exploratory Data Analysis (EDA)**  
### **4.1 Churn Distribution**  
We visualize the distribution of churned vs. retained customers.

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.countplot(x=df_train["Churn"])
plt.title("Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()
```

### **4.2 Relationship Between Total Spend and Churn**  
To understand whether higher-spending customers are more likely to churn, we use a boxplot.

```python
plt.figure(figsize=(8, 5))
sns.boxplot(x="Churn", y="Total Spend", data=df_train)
plt.title("Total Spend vs. Churn")
plt.show()
```

### **4.3 Feature Correlation Heatmap**  
A correlation heatmap helps us identify relationships between features.

```python
# Clean column names
df_train.columns = df_train.columns.str.strip()

# Encode categorical variables
categorical_cols = ["Gender", "Subscription Type", "Contract Length"]
df_train_encoded = df_train.copy()
df_train_encoded[categorical_cols] = df_train_encoded[categorical_cols].apply(lambda col: col.astype("category").cat.codes)
df_train_encoded["Churn"] = df_train_encoded["Churn"].map({"Yes": 1, "No": 0})

# Compute correlation
correlation_matrix = df_train_encoded.corr()

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
```

---

## **5. Feature Engineering**  
### **5.1 Encoding Categorical Variables**  
To prepare the dataset for machine learning, we encode categorical variables.

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df_train['Churn'] = encoder.fit_transform(df_train['Churn'])  # Convert 'Yes'/'No' to 1/0
df_train = pd.get_dummies(df_train, drop_first=True)
```

---

## **6. Model Training**  
### **6.1 Splitting Data into Training and Validation Sets**  

```python
from sklearn.model_selection import train_test_split

X = df_train.drop(columns=["Churn", "CustomerID"])
y = df_train["Churn"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

### **6.2 Training a Random Forest Classifier**  
Random Forest is an ensemble learning method that improves prediction accuracy.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

---

## **7. Model Evaluation**  
### **7.1 Checking Model Accuracy**  

```python
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
```

### **7.2 Confusion Matrix**  
A confusion matrix helps us understand how well the model distinguishes between churned and retained customers.

```python
from sklearn.metrics import confusion_matrix, classification_report

conf_matrix = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

### **7.3 Classification Report**  
This provides precision, recall, and F1-score for churn prediction.

```python
print(classification_report(y_val, y_pred))
```

---

## **8. Key Insights and Recommendations**  
1. **Customers with higher total spending tend to churn more.**  
   - Businesses should offer loyalty rewards and personalized discounts to retain these customers.  

2. **New customers (low tenure) have a higher churn rate.**  
   - Improving the onboarding experience and customer engagement strategies can help retain new customers.  

3. **A Random Forest model provides high accuracy.**  
   - Further improvement can be achieved using hyperparameter tuning and advanced models like XGBoost.  

4. **Future Enhancements:**  
   - Implement deep learning models for churn prediction.  
   - Use SHAP values to explain model predictions.  

---

## **Conclusion**  
This project demonstrates how machine learning can help predict customer churn. By analyzing key factors such as spending, tenure, and customer interactions, businesses can take proactive measures to reduce churn and improve customer retention.

