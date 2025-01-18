# Artificial Intelligence and Machine Learning - Answer Key

## Question Paper 1

#### Q1. (a) Define Artificial Intelligence (AI).
AI is the field of computer science that focuses on creating machines capable of performing tasks that would normally require human intelligence. These tasks include reasoning, problem-solving, learning, perception, and language understanding.

#### Q1. (b) Explain the key terminologies used in AI, such as problem-solving, inference, and reasoning.
- **Problem-solving:** Refers to the AI's ability to process inputs and find a solution for a given problem, such as navigating a maze.
- **Inference:** The process of drawing logical conclusions from available data or knowledge.
- **Reasoning:** The ability to process information logically to make decisions based on given facts or rules.

#### Q1. (c) Briefly describe the evolution of AI, highlighting at least three major milestones.
1. **1950s – Turing Test:** Alan Turing proposed the Turing Test, which is used to determine if a machine can exhibit human-like intelligence.
2. **1960s – Expert Systems:** The development of systems that simulate the decision-making ability of human experts.
3. **2010s – Deep Learning:** The rise of deep neural networks, significantly improving tasks such as image recognition and speech processing.

#### Q1. (d) What are the major classifications/types of AI? Provide examples for each type.
1. **Reactive Machines:** AI systems that respond to stimuli but do not store past experiences (e.g., IBM's Deep Blue).
2. **Limited Memory:** AI that can learn from historical data to make decisions (e.g., self-driving cars).
3. **Theory of Mind:** AI that can understand and simulate human emotions and thought processes (e.g., future advancements in robotics).
4. **Self-aware AI:** AI with self-consciousness and the ability to understand its own state (still hypothetical).

#### Q1. (e) Differentiate between Artificial Intelligence, Machine Learning, and Deep Learning with examples.
- **Artificial Intelligence (AI):** Encompasses all efforts to make machines act intelligently, e.g., chatbots.
- **Machine Learning (ML):** A subset of AI where machines learn from data, e.g., recommendation systems.
- **Deep Learning (DL):** A further subset of ML that uses neural networks with many layers to analyze complex data, e.g., image recognition.

---

#### Q2. (a) List and explain any four applications of AI in financial services.
1. **Algorithmic Trading:** AI analyzes large datasets in real-time to make trading decisions faster than humans.
2. **Fraud Detection:** AI models detect abnormal patterns in transactions to prevent fraud.
3. **Risk Management:** AI evaluates risks by analyzing financial data and forecasting market trends.
4. **Customer Service:** AI chatbots handle customer queries and provide personalized services in banks and financial institutions.

#### Q2. (b) Discuss the role of AI in algorithmic trading and fraud detection.
- **Algorithmic Trading:** AI automates trading by analyzing vast amounts of market data and identifying patterns to execute trades.
- **Fraud Detection:** AI continuously monitors transactions and applies machine learning models to detect suspicious activity or anomalies.

#### Q2. (c) What is a chatbot? How does AI enhance its functionality?
A **chatbot** is a software application used to simulate human-like conversation through text or voice interactions. AI enhances its functionality by enabling the bot to understand natural language, provide personalized responses, and continuously improve from interactions (through machine learning).

#### Q2. (d) Define robotic advisory services and give an example of its use.
**Robotic advisory services** refer to automated financial advising using algorithms that analyze a customer's financial situation and suggest investment strategies. An example is **Betterment**, which offers automated portfolio management and investment advice.

---


#### Q3. (a) Define Machine Learning (ML).
Machine Learning is a branch of artificial intelligence that focuses on the development of algorithms that allow computers to learn from and make predictions or decisions based on data, without being explicitly programmed.

#### Q3. (b) Trace the historical development of ML, mentioning at least two key advancements.
1. **1950s – Perceptron:** Early neural networks were developed as the first step in machine learning.
2. **1990s – Support Vector Machines (SVM):** A significant algorithmic breakthrough in classification tasks.

#### Q3. (c) What are the three types of learning in ML? Briefly describe each with examples.
1. **Supervised Learning:** The model is trained on labeled data. Example: Predicting house prices based on historical data.
2. **Unsupervised Learning:** The model works with unlabeled data, finding patterns or groupings. Example: Customer segmentation in marketing.
3. **Reinforcement Learning:** The agent learns by interacting with an environment and receiving rewards or penalties. Example: Training a robot to navigate a maze.

#### Q3. (d) Explain the role of datasets in Machine Learning.
Datasets are essential in ML as they are the raw material for learning. They provide the examples and features used by algorithms to learn patterns, make predictions, and assess model performance.

#### Q3. (e) Differentiate between supervised and unsupervised learning in terms of objectives and techniques.
- **Supervised Learning:** The goal is to learn a mapping from inputs to outputs using labeled data. Techniques: Regression, Classification.
- **Unsupervised Learning:** The goal is to find hidden patterns in data without labels. Techniques: Clustering, Dimensionality reduction.

---

#### Q4. (a) What is data pre-processing, and why is it crucial in Machine Learning?
Data pre-processing is the process of cleaning, transforming, and organizing raw data into a usable format for machine learning models. It's crucial because data often comes with missing values, inconsistencies, or irrelevant features that can skew model performance.

#### Q4. (b) List and briefly describe any three common data pre-processing techniques.
1. **Normalization:** Scaling data to a standard range (e.g., [0, 1]) to improve model convergence.
2. **Missing Value Imputation:** Replacing missing values in datasets using strategies like mean or median imputation.
3. **Encoding Categorical Data:** Converting categorical data (like "red" and "blue") into numeric formats (like 0 and 1).

#### Q4. (c) Define training and testing datasets. How are they used in the ML lifecycle?
- **Training Dataset:** A subset of data used to train the machine learning model.
- **Testing Dataset:** A separate subset used to evaluate the performance of the model after training. This helps prevent overfitting.

#### Q4. (d) What is cross-validation? Explain the purpose of k-fold cross-validation.
**Cross-validation** is a technique used to assess the generalization of a model. In k-fold cross-validation, the dataset is split into k subsets. The model is trained on k-1 subsets and tested on the remaining one, iterating until each subset has been used for testing.

---

#### Q5. (a) Write a Python program to implement a simple linear regression model using the `sklearn` library.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Example dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 2, 3, 4, 5])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
```

#### Q5. (b) Explain the role of the following steps in your program:
- **Splitting the dataset into training and testing sets**: This ensures that we can train the model on one portion of the data and test it on unseen data to evaluate its performance.
- **Fitting the model to the training data**: This involves using the training data to train the linear regression model so it can make predictions.
- **Evaluating the model's performance on the test data**: By using the test set, we can assess how well the model generalizes to new, unseen data.
---

## Question Paper 2: Answer Key


**Q1.**  
(a) **Definition of Intelligence**  
In AI, intelligence is the ability of a machine to mimic human-like cognitive functions such as learning, reasoning, and problem-solving.  

(b) **Turing Test**  
The Turing Test assesses a machine's ability to exhibit intelligent behavior equivalent to or indistinguishable from that of a human.

(c) **Challenges in Developing AI**  
- **Lack of Data**: Many AI models require large datasets to perform well.  
- **Bias in Data**: AI can learn biased patterns if data is not carefully curated.  
- **Interpretability**: Many AI models (e.g., deep learning) are "black boxes," making it hard to understand their decisions.

---

**Q2.**  
(a) **AI in Healthcare**  
- AI is used for diagnosing diseases from images (e.g., detecting tumors in X-rays).  
- AI-powered systems can suggest personalized treatment plans.  

(b) **AI in Retail**  
- AI powers recommendation engines to suggest products based on previous purchases.  
- Chatbots are used for customer support, handling inquiries 24/7.  

(c) **Limitations of AI**  
- AI lacks common sense reasoning.  
- It is highly dependent on the quality of data.

---

### **Module 2: Overview of Machine Learning**

**Q3.**  
(a) **Generalization**  
Generalization refers to the model’s ability to perform well on unseen data. Overfitting leads to poor generalization, while underfitting leads to an underperforming model.

(b) **Positive and Negative Classes**  
- **Positive Class**: The class of interest (e.g., "fraudulent" in fraud detection).  
- **Negative Class**: The opposite class (e.g., "non-fraudulent").

(c) **Probabilistic Modeling**  
Probabilistic models estimate the probability of different outcomes. For example, Naive Bayes classifiers are based on probabilistic modeling.

---

**Q4.**  
(a) **Classification vs. Regression**  
- **Classification**: Predicts categorical outcomes (e.g., spam or not spam).  
- **Regression**: Predicts continuous outcomes (e.g., house prices).

(b) **Evaluation Metrics**  
- **Accuracy**: The proportion of correctly classified instances.  
- **Precision**: The proportion of positive predictions that are actually positive.  
- **Recall**: The proportion of actual positives that are correctly identified.  
- **F1-Score**: The harmonic mean of precision and recall.

(c) **Confusion Matrix**  
A confusion matrix is a table used to evaluate the performance of classification algorithms. It shows true positives, false positives, true negatives, and false negatives.

---

**Q5.**  
(a) **KNN Classifier Program**  
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('dataset.csv')

# Split data into features and target
X = data[['feature1', 'feature2']]
y = data['target']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

(b) **Normalization in KNN**  
Normalization scales the features to ensure that the distance between points is meaningful. KNN relies on distance metrics, so it’s crucial to standardize the data.

(c) **KNN Model Evaluation**  
- **Accuracy**: Proportion of correct predictions.  
- **Confusion Matrix**: Helps visualize the performance of the model.

---
