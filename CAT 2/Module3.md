# CAT 2
Module 3
1. What is convergence in terms of machine learning 
2. What is model selection, when to use which kinds of models 
3. List any ten machine learning models and their use cases 
4. How to improve linear regression score in kaggle . 
5. What is bias, what is variance, define the terms bias , variance and trade off 
6. What is cross validation 
7. What is regularisation and its types.
8. Diffrence between generative and discirminative models . Give five examples each 


# 1. What is convergence in terms of machine learning 
    In machine learning, convergence is when a model's predictions stop improving and its error rate becomes constant. It's a state where the model has reached a point of stability and additional training won't improve it. 

# 2. & 3. What is model selection, when to use which kinds of models 
Great question! Here's **why these algorithms perform well** in their typical applications, along with a recap in a clean table:

---

| **Algorithm**               | **Why It Performs Well**                                                                                  | **Example Applications**                        |
|------------------------------|-----------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| **Linear Regression**        | Works well when there is a **linear relationship** between variables. Easy to interpret and fast.        | House price prediction, sales forecasting      |
| **Logistic Regression**      | Great for **probability-based binary outcomes** and is highly efficient on simple, linearly separable data. | Spam detection, medical diagnosis              |
| **Decision Trees**           | Can handle **non-linear data** and capture complex decision boundaries with easy interpretability.       | Loan approval, credit scoring                  |
| **Random Forest**            | Combines many trees to **reduce overfitting** and improve accuracy through **ensemble learning**.        | Fraud detection, stock market analysis         |
| **SVM (Support Vector Machine)** | Works well with **high-dimensional spaces** and finds optimal boundaries between classes.               | Face detection, bioinformatics classification  |
| **K-Nearest Neighbors (KNN)**| Excellent for datasets with **local patterns**, as it uses proximity to make predictions.                | Handwriting recognition, recommendation systems|
| **Naive Bayes**              | Works well when features are **independent** (or nearly so) and scales well to large text datasets.      | Sentiment analysis, spam filtering             |
| **K-Means Clustering**       | Effective when data naturally forms **distinct clusters** and you need to group similar data points.     | Customer segmentation, image compression       |
| **Principal Component Analysis (PCA)**| Reduces dimensions while preserving variance, helping in **noise reduction** and better visualization. | Image compression, feature reduction           |
| **Neural Networks (ANN, CNN, RNN)** | Highly flexible to learn **complex, non-linear patterns** in huge datasets through layers of computation. | Image recognition, speech recognition          |

---

### 🔍 Quick summary of **why they excel**:
- **Simple algorithms (like Linear/Logistic Regression)** do well when data relationships are straightforward.
- **Tree-based models** shine when capturing **interactions** and **non-linear relationships**.
- **Distance-based models (KNN, SVM)** work well where **proximity** or **margins** define decisions.
- **Probabilistic models (Naive Bayes)** are strong for **text and categorical data**.
- **Unsupervised models (K-Means, PCA)** discover **hidden patterns** without needing labeled data.
- **Deep learning (Neural Networks)** handles **complex, large-scale, unstructured data** like images, text, and audio.

---

- https://colab.research.google.com/drive/1DSXmOMv5G1SuYMfR8i3v-CoucXomNcCU?usp=sharing
- https://www.kaggle.com/code/janiezj/breast-cancer-analysis-using-machine-learning

# 3.1 How does Recursive Feature Elimination (RFE) work?
- Recursive Feature Elimination (RFE) is a feature selection technique that works by recursively removing the least important features one by one until only the most important remaining. It is used to reduce the complexity of a model by removing features that do not contribute to the model's output. RFE is effective at reducing overfitting and can help improve model performance by removing irrelevant features.

- https://colab.research.google.com/drive/1Y_HxCqpJ3OzJzeCzY-4I4oHJizkV5cU8?usp=sharing

# 3.2 What is cross-validation?
- Cross-validation is a technique used to evaluate a model's performance by splitting the data into training and testing sets. It helps in evaluating the model's performance by splitting the data into k equal parts, and using one part as the testing set and the remaining parts as the training set. This process is repeated k times, with each testing set used to evaluate the model's performance. The average performance across all k tests is then used as a measure of model's accuracy.
- https://colab.research.google.com/drive/1x6GeA_PCrR-_ogCWWUmoPurQcSviG5KL?usp=sharing


# 4. How to improve linear regression score in kaggle . 

Improving a **Linear Regression** score in a **Kaggle** competition (or any predictive task) requires a systematic approach. The goal is to reduce the error (like **RMSE**, **MAE**, **R²**, etc.) and make your model generalize better. Here's a detailed, step-by-step guide on how you can improve your linear regression model's performance:

---

## 🚀 Step-by-Step Approach to Improve Linear Regression Score

---

### **1. Data Cleaning**
Bad or noisy data kills model performance. Ensure your data is clean:
- **Handle missing values**: Fill them with mean, median, mode, or use domain knowledge.
  - Example: `df.fillna(df.mean(), inplace=True)`
- **Remove outliers**: Outliers skew linear regression since it's sensitive to extreme values. Use:
  - Z-score, IQR methods to identify and remove them.
- **Fix inconsistent data**: Check for typos, duplicates, or wrong units.

---

### **2. Feature Engineering**
This is often where Kaggle competitions are won.

#### ✅ Transform features:
- **Log transformation** for skewed distributions (e.g., income, prices).
  ```python
  df['price'] = np.log1p(df['price'])
  ```
- **Polynomial features**: Linear regression is only linear **in parameters**, so adding polynomial features helps model non-linear relations.
  ```python
  from sklearn.preprocessing import PolynomialFeatures
  poly = PolynomialFeatures(degree=2)
  X_poly = poly.fit_transform(X)
  ```

#### ✅ Combine or create new features:
- Ratios (e.g., `rooms_per_household`)
- Differences (e.g., `year_built - year_renovated`)

---

### **3. Feature Selection**
Too many irrelevant features can hurt performance.

#### 🔹 Techniques:
- **Correlation matrix**: Remove highly correlated variables (multicollinearity).
- **Regularization (Ridge/Lasso)**: Lasso can zero out irrelevant features.
- **Recursive Feature Elimination (RFE)**.

---

### **4. Scaling Features**
Linear regression assumes all features contribute equally. Standardization helps.
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

### **5. Model Regularization**
In Kaggle, pure LinearRegression (`sklearn.linear_model.LinearRegression`) may not work well with complex data.

#### Use these:
| Method | When to use |
|--------|-------------|
| **Ridge** (L2) | When multicollinearity is present |
| **Lasso** (L1) | For feature selection (sparse solutions) |
| **ElasticNet** | Combines Ridge + Lasso advantages |

Example:
```python
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
```

Tune the regularization strength (`alpha`) via cross-validation.

---

### **6. Cross-Validation**
Avoid overfitting to training data. Use:
```python
from sklearn.model_selection import cross_val_score
cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
```

Or better, use `GridSearchCV` or `RandomizedSearchCV` to optimize hyperparameters.

---

### **7. Remove Leakage**
Check if your data has any leakage. For example, using a "future" feature (like sales from the next month) as a predictor. Leakage falsely boosts your score during training but fails in reality.

---

### **8. Advanced Techniques**
While these go beyond "pure" linear regression, they help:
- **Stacking**: Combine several linear models.
- **Ensembling**: Average predictions from multiple models.
- **Target Encoding**: For categorical variables (be careful to avoid leakage).

---

### **9. Submission Strategy**
- Make sure your submission file format is perfect (Kaggle penalizes invalid submissions).
- Try different feature sets and compare results.
- Keep a validation strategy aligned with the competition (time-based split for time series, stratified for imbalanced data).

---

### **10. Bonus Tips**
✅ Check Kaggle forums for public kernels (not to copy but to get ideas).  
✅ Analyze feature importance (even in linear models via coefficients).  
✅ Visualize residuals to see patterns your model misses.  

---

## 💡 Summary:
Improving a linear regression score is **80% data work**, **10% model tuning**, and **10% luck**. Focus heavily on cleaning, engineering, and selecting features.

 **template notebook** 
    - https://colab.research.google.com/drive/1t7BM9Znb38_Evjoyd9w9ZohS2QTdkPhu?usp=sharing