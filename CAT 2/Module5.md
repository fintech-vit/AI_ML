# CAT 2
Module 5
1. Explain how Support Vector Machine works 
2. What is naive Bayesian model .

# 1. **How Does Support Vector Machine (SVM) Work?**  

## **Introduction to SVM**
A **Support Vector Machine (SVM)** is a **supervised learning algorithm** used for **classification and regression** tasks. It is mainly used for **binary classification** but can be extended for multi-class problems.  

SVM works by **finding the best boundary (decision boundary) that separates different classes** in a dataset. It is especially effective in **high-dimensional spaces** and is useful when the number of features is greater than the number of samples.  

---

## **Why Do We Use SVM?**
SVM is preferred because:
1. **It works well for small datasets** where the number of features is high.
2. **It can classify complex patterns** using the **kernel trick**.
3. **It is effective in high-dimensional spaces** and avoids overfitting.
4. **It works well with structured and unstructured data** such as text and images.
5. **It finds the optimal decision boundary**, maximizing the margin between different classes.

---

## **Where Do We Use SVM?**
SVM is used in many applications, including:

1. **Text Classification**  
   - Spam detection (Gmail, Outlook)  
   - Sentiment analysis (detecting positive/negative reviews)  

2. **Image Recognition & Computer Vision**  
   - Handwritten digit classification (MNIST dataset)  
   - Facial expression recognition  

3. **Medical Diagnosis**  
   - Detecting cancerous tumors (breast cancer classification)  
   - Disease prediction from patient data  

4. **Finance & Business Analytics**  
   - Credit risk assessment  
   - Fraud detection in transactions  

5. **Stock Market Prediction**  
   - Predicting price movements based on financial data  

6. **Bioinformatics & Genetics**  
   - Protein classification  
   - Gene classification  

---

## **How Does SVM Work?**
SVM **finds the best hyperplane** that separates data into different classes. The goal is to find a hyperplane that has the **maximum margin** (i.e., the largest distance between data points of different classes).  

### **Step 1: Separating Data Using a Hyperplane**
- In a **2D space**, the hyperplane is a **straight line**.
- In a **3D space**, the hyperplane is a **plane**.
- In **higher dimensions**, it is a **hyperplane**.

### **Step 2: Finding Support Vectors**
- **Support Vectors** are the **data points closest to the hyperplane**.  
- These points **define the decision boundary**.  
- The **distance between these points and the hyperplane is called the margin**.  
- The goal of SVM is to **maximize this margin**.

### **Step 3: Handling Non-Linearly Separable Data (Kernel Trick)**
Sometimes, data is not **linearly separable**. In such cases, we use the **kernel trick** to transform the data into a higher-dimensional space where it becomes separable.

#### **Common Kernel Functions:**
1. **Linear Kernel** – Used when data is linearly separable.  
2. **Polynomial Kernel** – Useful when data has curved decision boundaries.  
3. **Radial Basis Function (RBF) Kernel** – Most commonly used for non-linear classification.  
4. **Sigmoid Kernel** – Similar to neural networks.

---

## **Example: Implementing SVM in Python**
Let's classify the **Iris dataset** using SVM.

### **Step 1: Install Required Libraries**
```python
pip install scikit-learn numpy matplotlib
```

### **Step 2: Import Libraries**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
```

### **Step 3: Load and Preprocess the Dataset**
```python
# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Take only the first two features for visualization
y = iris.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### **Step 4: Train an SVM Model**
```python
# Create and train an SVM model with an RBF kernel
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)
```

### **Step 5: Make Predictions and Evaluate**
```python
# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
```

### **Step 6: Visualizing the Decision Boundary**
```python
# Function to plot decision boundary
def plot_decision_boundary(model, X, y):
    h = 0.02  # Step size for meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.show()

# Plot decision boundary
plot_decision_boundary(svm_model, X, y)
```

---

## **How This SVM Model Works**
1. **Loads the Iris dataset** (a well-known dataset for classification).
2. **Splits data into training and testing sets**.
3. **Creates an SVM classifier** using an **RBF kernel**.
4. **Trains the model** on the training data.
5. **Makes predictions** and evaluates accuracy.
6. **Visualizes the decision boundary**.

---

## **Advantages of SVM**
**Works well for both linear and non-linear data**.  
**Effective for small datasets** with high-dimensional features.  
**Resistant to overfitting** due to its margin-based optimization.  
**Can handle complex classification problems** using kernel tricks.  

## **Disadvantages of SVM**
**Computationally expensive for large datasets**.  
**Choosing the right kernel and parameters requires tuning**.  
**Not ideal for datasets with high noise**.  

---

## **Conclusion**
**Support Vector Machines (SVMs) are powerful algorithms** for classification and regression. They work by **finding the optimal decision boundary** and **maximizing the margin** between different classes. With the **kernel trick**, SVM can handle **non-linearly separable data** efficiently.  

# 2. **Naïve Bayes Model: A Simple Yet Powerful Classifier**  

## **Introduction to Naïve Bayes**  
**Naïve Bayes** is a **probabilistic machine learning algorithm** based on **Bayes' Theorem**. It is mainly used for **classification tasks** such as **spam detection, sentiment analysis, and document classification**.  

The **"naïve"** part comes from the assumption that **all features are independent**, which is rarely true in real-world data, but the model still works surprisingly well in many cases.  

---

## **Why Do We Use Naïve Bayes?**
Naïve Bayes is popular because:
**It is fast and efficient** for large datasets.  
**It requires little training data** to make predictions.  
**It performs well for text classification tasks** (e.g., spam filtering).  
**It is easy to interpret and implement**.  

---

## **Where Do We Use Naïve Bayes?**
Naïve Bayes is widely used in:
1. **Spam Detection** – Identifying spam emails based on words used in emails.  
2. **Sentiment Analysis** – Classifying reviews as positive or negative.  
3. **Document Classification** – Categorizing news articles, blogs, etc.  
4. **Medical Diagnosis** – Predicting diseases from symptoms.  
5. **Fraud Detection** – Detecting fraudulent transactions.  

---

## **How Does Naïve Bayes Work?**
### **Step 1: Understanding Bayes' Theorem**
Naïve Bayes is based on **Bayes' Theorem**, which states:

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

Where:  
- \( P(A|B) \) is the **posterior probability** (probability of A given B).  
- \( P(B|A) \) is the **likelihood** (probability of B given A).  
- \( P(A) \) is the **prior probability** (initial probability of A).  
- \( P(B) \) is the **evidence** (overall probability of B).  

### **Step 2: Applying Naïve Bayes for Classification**
For classification, we calculate:

$$
P(Class | Features) = \frac{P(Features | Class) \cdot P(Class)}{P(Features)}
$$

Since $$ P(Features) $$ is the same for all classes, we only need:

$$
P(Class | Features) \propto P(Features | Class) \cdot P(Class)
$$

We classify a new data point into the class that gives the **highest probability**.

---

## **Types of Naïve Bayes Classifiers**
1. **Gaussian Naïve Bayes** – Assumes features follow a **normal distribution**.  
2. **Multinomial Naïve Bayes** – Used for **text classification**, where features are word counts.  
3. **Bernoulli Naïve Bayes** – Used for binary features (e.g., words in spam detection).  

---

## **Example: Implementing Naïve Bayes in Python**
Let's classify spam and non-spam emails using Naïve Bayes.

### **Step 1: Install Required Libraries**
```python
pip install scikit-learn numpy pandas
```

### **Step 2: Import Libraries**
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
```

### **Step 3: Load the Dataset**
```python
# Sample dataset (spam detection)
data = {'Message': ['Win a lottery now', 'Meet me at 5 PM', 'Congratulations! You won', 'Hello, how are you?', 
                    'Get a free iPhone', 'Let’s catch up tomorrow'],
        'Spam': [1, 0, 1, 0, 1, 0]}  # 1 = Spam, 0 = Not Spam

df = pd.DataFrame(data)
```

### **Step 4: Preprocess and Split the Data**
```python
X = df['Message']
y = df['Spam']

# Convert text data to numerical data using CountVectorizer
vectorizer = CountVectorizer()
X_transformed = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
```

### **Step 5: Train Naïve Bayes Model**
```python
# Train Naïve Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
```

### **Step 6: Make Predictions and Evaluate**
```python
# Make predictions
y_pred = nb_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
```

---

## **How This Naïve Bayes Model Works**
1. **Loads a small dataset of spam and non-spam messages**.  
2. **Converts text data into numerical data** using `CountVectorizer`.  
3. **Splits the dataset into training and testing sets**.  
4. **Trains a Multinomial Naïve Bayes model**.  
5. **Predicts spam or not spam and evaluates accuracy**.  

---

## **Advantages of Naïve Bayes**
**Fast and efficient** even with large datasets.  
**Works well for text classification** tasks like spam filtering.  
**Requires less training data** compared to other algorithms.  
**Simple and interpretable**.  

## **Disadvantages of Naïve Bayes**
**Assumes all features are independent**, which is unrealistic in real-world data.  
**Not suitable for complex relationships** like deep learning models.  
**Sensitive to irrelevant features** (feature selection is important).  

---

## **Conclusion**
Naïve Bayes is a **powerful yet simple classification algorithm** based on **Bayes' Theorem**. Despite its **"naïve" assumption** that features are independent, it works remarkably well in practice, especially for **text classification** and **spam detection**.  
