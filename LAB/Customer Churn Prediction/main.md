Here's an improved Markdown file with more **graphs and explanations** to make it **self-explanatory** for students. It includes detailed **visualizations using Matplotlib and Seaborn** to help understand the dataset better.  

---

# **Customer Churn Prediction**  

## **Introduction**  
Customer churn occurs when a customer **stops using a company's service**. Predicting churn helps businesses **retain customers and reduce losses**.  

### **Why is Churn Prediction Important?**  
- **Reduces Customer Acquisition Costs** → Retaining a customer is cheaper than acquiring a new one.  
- **Improves Business Revenue** → Less churn = More stability.  
- **Enhances Customer Satisfaction** → Businesses can offer better services based on churn insights.  

---

## **Step 1: Understanding the Data**  
We use a dataset that contains customer details such as **age, tenure, monthly charges, and total charges**, along with whether they churned (`Yes` or `No`).  

### **Dataset Example**  

| Customer ID | Age | Tenure (months) | Monthly Charges | Total Charges | Churn (Yes/No) |
|------------|----|----------------|---------------|--------------|--------------|
| 1001       | 25 | 12             | 29.85         | 350.5        | No           |
| 1002       | 42 | 24             | 56.90         | 1200.0       | Yes          |
| 1003       | 30 | 8              | 42.30         | 450.2        | No           |

### **Loading the Dataset**  
```python
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt  

# Load dataset  
df = pd.read_csv("customer_churn.csv")  

# Display first few rows  
df.head()
```

---

## **Step 2: Data Exploration and Visualization**  

### **1️⃣ Churn Distribution**  
Let’s check how many customers **churned vs. stayed**.  

```python
sns.countplot(x=df['Churn'], palette=['#ff6f00', '#0066ff'])  
plt.title("Churn Distribution")  
plt.show()
```

**Expected Output:**  
- A **bar chart** showing the number of **customers who churned vs. those who stayed**.  

### **2️⃣ Tenure vs. Churn**  
Does **customer tenure** affect churn?  

```python
plt.figure(figsize=(10, 5))  
sns.histplot(df[df['Churn'] == 'Yes']['Tenure (months)'], bins=20, color='red', label="Churned")  
sns.histplot(df[df['Churn'] == 'No']['Tenure (months)'], bins=20, color='blue', label="Not Churned")  
plt.xlabel("Tenure (months)")  
plt.title("Distribution of Tenure by Churn Status")  
plt.legend()
plt.show()
```

**Insights:**  
- Customers with a **shorter tenure are more likely to churn**.  

### **3️⃣ Monthly Charges vs. Churn**  
Are customers with **higher bills** more likely to churn?  

```python
plt.figure(figsize=(10, 5))  
sns.boxplot(x=df['Churn'], y=df['Monthly Charges'], palette=['red', 'blue'])  
plt.title("Monthly Charges by Churn Status")  
plt.show()
```

**Insights:**  
- Customers with **higher monthly charges** are **more likely to churn**.  

---

## **Step 3: Data Preprocessing**  

### **Convert Categorical Variables to Numeric**  
Since `Churn` is **Yes/No**, we convert it to `1` (Yes) and `0` (No).  

```python
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})  
df = pd.get_dummies(df, drop_first=True)  # One-hot encoding for categorical variables
```

### **Splitting Data into Training & Test Sets**  
```python
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  

X = df.drop(columns=['Churn'])  
y = df['Churn']  

# Split data  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Scale numeric features  
scaler = StandardScaler()  
X_train = scaler.fit_transform(X_train)  
X_test = scaler.transform(X_test)
```

---

## **Step 4: Model Selection & Training**  

### **Logistic Regression Model**  
```python
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score, classification_report  

# Train model  
model = LogisticRegression()  
model.fit(X_train, y_train)  

# Predict churn  
y_pred = model.predict(X_test)  

# Evaluate model  
print("Accuracy:", accuracy_score(y_test, y_pred))  
print("Classification Report:\n", classification_report(y_test, y_pred))
```

---

## **Step 5: Model Evaluation**  

### **Confusion Matrix**  
```python
from sklearn.metrics import confusion_matrix  
import seaborn as sns  

cm = confusion_matrix(y_test, y_pred)  
sns.heatmap(cm, annot=True, cmap='coolwarm', fmt='d')  
plt.xlabel("Predicted")  
plt.ylabel("Actual")  
plt.title("Confusion Matrix")  
plt.show()
```

**Expected Output:**  
A **heatmap** showing the confusion matrix. It helps understand **false positives and false negatives**.  

### **ROC Curve**  
```python
from sklearn.metrics import roc_curve, auc  

y_prob = model.predict_proba(X_test)[:, 1]  
fpr, tpr, _ = roc_curve(y_test, y_prob)  
roc_auc = auc(fpr, tpr)  

plt.figure(figsize=(8, 6))  
plt.plot(fpr, tpr, color='blue', label="ROC curve (area = %0.2f)" % roc_auc)  
plt.plot([0, 1], [0, 1], 'r--')  
plt.xlabel("False Positive Rate")  
plt.ylabel("True Positive Rate")  
plt.title("Receiver Operating Characteristic (ROC) Curve")  
plt.legend()  
plt.show()
```

**Expected Output:**  
A **ROC curve** to measure how well the model distinguishes between churned and non-churned customers.  

---

## **Step 6: Improving the Model**  

### **Using a More Powerful Model (Random Forest)**  
```python
from sklearn.ensemble import RandomForestClassifier  

rf_model = RandomForestClassifier(n_estimators=200, max_depth=10)  
rf_model.fit(X_train, y_train)  

# Predictions  
y_pred_rf = rf_model.predict(X_test)  

# Accuracy  
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))  
```

---

## **Step 7: Deploying the Model**  
We can **deploy this model as an API** using **Flask**.  

```python
from flask import Flask, request, jsonify  
import pickle  

app = Flask(__name__)  

# Load trained model  
model = pickle.load(open('churn_model.pkl', 'rb'))  

@app.route('/predict', methods=['POST'])  
def predict():  
    data = request.json  
    prediction = model.predict([data['features']])  
    return jsonify({'churn_prediction': int(prediction[0])})  

if __name__ == '__main__':  
    app.run(debug=True)
```

---

## **Conclusion**  
We successfully built a **Customer Churn Prediction Model** using **machine learning**. Key takeaways:  
✅ Customers with **higher monthly charges are more likely to churn**.  
✅ **Shorter tenure increases churn risk**.  
✅ Machine learning helps businesses **reduce churn & improve customer retention**.  

---

Let me know if you need further improvements! 🚀