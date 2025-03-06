# 5. What is bias, what is variance, define the terms bias , variance and trade off 
---

## ✨ **Definitions**

### ✅ **Bias**  
Bias is the error caused by approximating a real-world problem, which might be complex, by a much simpler model.  
- **High bias** means your model is too simple, underfitting the data.  
- Example: Trying to fit a straight line to complex, curvy data.

---

### ✅ **Variance**  
Variance is the error caused by the model's sensitivity to small changes in the training dataset.  
- **High variance** means your model is too complex, overfitting the training data and failing on new data.  
- Example: A very flexible model that learns noise in the training data.

---

### ✅ **Bias-Variance Tradeoff**  
- If you make your model **simpler**, bias increases, variance decreases.  
- If you make your model **more complex**, bias decreases, variance increases.  
The goal is to find the **sweet spot** where **total error** (bias² + variance) is minimized.

---

## ✅ Real-time Example with Code

Let's use a real dataset like **"California Housing"** to show this with Linear Regression (high bias, low variance) and Decision Tree (low bias, high variance).

---

### 🔹 Install and Import Packages
```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
```

---

### 🔹 Load Dataset
```python
# Load real-world dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### 🔹 High Bias Model (Linear Regression)
```python
# Simple model (high bias, low variance)
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions
y_train_pred_lr = lr.predict(X_train)
y_test_pred_lr = lr.predict(X_test)

# Errors
print("Linear Regression (High Bias):")
print(f"Train MSE: {mean_squared_error(y_train, y_train_pred_lr):.4f}")
print(f"Test MSE: {mean_squared_error(y_test, y_test_pred_lr):.4f}")
```

---

### 🔹 High Variance Model (Decision Tree)
```python
# Complex model (low bias, high variance)
dt = DecisionTreeRegressor(random_state=42, max_depth=30)
dt.fit(X_train, y_train)

# Predictions
y_train_pred_dt = dt.predict(X_train)
y_test_pred_dt = dt.predict(X_test)

# Errors
print("\nDecision Tree (High Variance):")
print(f"Train MSE: {mean_squared_error(y_train, y_train_pred_dt):.4f}")
print(f"Test MSE: {mean_squared_error(y_test, y_test_pred_dt):.4f}")
```

---

### ✅ Expected Output Example:
```
Linear Regression (High Bias):
Train MSE: 0.5243
Test MSE: 0.5553

Decision Tree (High Variance):
Train MSE: 0.0000
Test MSE: 0.7321
```

---

### ✅ Explanation:
| Model               | Bias | Variance | Train Error | Test Error |
|---------------------|------|----------|-------------|------------|
| Linear Regression  | High | Low      | High        | High       |
| Decision Tree      | Low  | High     | Low (almost 0) | High (overfit) |

---

### ✅ Takeaway:
- **Linear Regression** underfits → high bias, similar errors on train and test.
- **Decision Tree** overfits → low bias on train, but high variance on test.

👉 The **trade-off** is to balance between these extremes to generalize well on unseen data.

---

# reference for 5 and 6
- https://colab.research.google.com/drive/1kiaYMAZuIHF7rrunfyVLo2vwqxKWky0o?usp=sharing


# 7 Regularization

- https://colab.research.google.com/drive/1Y2lxwOfSsGkakWWHi4Doh8upqnpCwgzT?usp=sharing

- https://colab.research.google.com/drive/1pPqJwlBPIYT1ep1SjiM8JD9MXuHOsE6G?usp=sharing


# 8
### Difference Between **Generative** and **Discriminative** Models:

| Feature                        | Generative Models                          | Discriminative Models                   |
|---------------------------------|--------------------------------------------|------------------------------------------|
| **Purpose**                    | Model the **joint probability** \( P(X, Y) \) and can generate data. | Model the **conditional probability** \( P(Y|X) \) to classify directly. |
| **What they learn**            | Learn how the data is **generated**.      | Learn the **decision boundary** between classes. |
| **Output**                     | Can generate new data similar to training data. | Predict class labels or probabilities. |
| **Complexity**                 | Generally more complex and computationally heavier. | Often simpler and focused on classification. |
| **Use Cases**                  | Data generation, semi-supervised learning. | Supervised learning (classification, regression). |

---

### ✅ **Examples of Generative Models:**
1. **Naive Bayes**
2. **Gaussian Mixture Models (GMM)**
3. **Hidden Markov Models (HMM)**
4. **Variational Autoencoders (VAE)**
5. **Generative Adversarial Networks (GANs)**

---

### ✅ **Examples of Discriminative Models:**
1. **Logistic Regression**
2. **Support Vector Machines (SVM)**
3. **Random Forest**
4. **K-Nearest Neighbors (KNN)**
5. **Neural Networks (for classification)**

---

### 🔑 **Quick memory tip:**
- **Generative = Generate data** (focus on data distribution).
- **Discriminative = Decide classes** (focus on boundaries).
