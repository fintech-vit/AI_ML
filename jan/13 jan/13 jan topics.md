# Jan 13, 2025

## Data Preprocessing
### What Does It Mean?
- **Data preprocessing** is the first and most important step in any machine learning project. It involves preparing raw data to make it suitable for training a model.
- Raw data often contains inconsistencies, missing values, and irrelevant information. Preprocessing ensures the data is clean, consistent, and usable.

### Key Steps:
1. **Data Cleaning**:
   - Removing duplicates.
   - Handling missing values (e.g., filling them with mean/median or dropping rows/columns).
   - Fixing inconsistencies (e.g., standardizing units like "km" vs. "kilometers").
2. **Data Transformation**:
   - **Scaling**: Standardizing numerical data so all features have the same scale.
   - **Encoding**: Converting categorical data (e.g., "Male" and "Female") into numerical values.
   - **Feature Extraction**: Selecting or creating meaningful features from raw data.
3. **Data Splitting**:
   - Dividing data into:
     - **Training Set**: Used to train the model.
     - **Validation Set**: Used to tune the model (optional).
     - **Testing Set**: Used to evaluate the model.

---

## ML Life Cycle
### What Does It Mean?
The **ML life cycle** is the step-by-step process used to develop, deploy, and maintain a machine learning model.

### Steps:
1. **Data Preprocessing**: Cleaning and transforming the data.
2. **Model Training**: Teaching the model to learn from the training data.
3. **Model Selection**: Comparing different models and selecting the best one.
4. **Model Evaluation**: Testing the model on unseen data to measure performance.
5. **Model Deployment**: Integrating the model into a real-world application.
6. **Model Maintenance**: Updating the model with new data as needed.
7. **Model Monitoring**: Continuously tracking model performance and identifying issues.

---

## Positive and Negative Classes
### What Does It Mean?
In **classification problems**, data points are divided into two or more categories. These categories are often referred to as **positive** and **negative classes**:
- **Positive Class**: The category of primary interest. For example:
  - In a disease prediction model, "has disease" is the positive class.
  - In email spam detection, "spam" is the positive class.
- **Negative Class**: The other category. For example:
  - In a disease prediction model, "no disease" is the negative class.
  - In email spam detection, "not spam" is the negative class.

### Why Is This Important?
- **Positive and negative classes** help us evaluate the model's performance using metrics like:
  - **Precision**: How many of the predicted positive cases are actually positive.
  - **Recall**: How many of the actual positive cases were correctly identified.
  - **F1 Score**: Balances precision and recall.

---

## Dataset for ML
### What Does It Mean?
A **dataset** is a collection of data used to train and test machine learning models. It consists of:
- **Features (X)**: Input variables (e.g., age, income, or pixel values in images).
- **Labels (Y)**: Output variables or targets (e.g., "yes/no," "cat/dog").

### Types of Datasets:
1. **Structured Data**: Organized into rows and columns (e.g., spreadsheets).
2. **Unstructured Data**: Data without a clear structure, like images, text, and audio.

---

## Training vs. Testing
### What Does It Mean?
- **Training**: The phase where the ML model learns patterns and relationships in the data.
- **Testing**: Evaluating how well the trained model performs on unseen data.

### Why Separate Training and Testing?
- To avoid **overfitting**, where the model memorizes the training data instead of generalizing.

---

## Cross-Validation
### What Does It Mean?
Cross-validation is a technique used to evaluate the performance of a model by dividing the data into multiple subsets or **folds**.

### How It Works:
1. Split the data into `k` folds.
2. Train the model on `k-1` folds and test it on the remaining fold.
3. Repeat the process for all folds and compute the average performance.

### Why Use It?
- Ensures the model is tested on different parts of the dataset, reducing bias and overfitting.

---

## Generalization
### What Does It Mean?
**Generalization** is the ability of a model to perform well on new, unseen data.

### Factors Affecting Generalization:
1. **Data Quality**: Clean and diverse data improves generalization.
2. **Model Complexity**: A model that's too simple may underfit, while a model that's too complex may overfit.
3. **Sufficient Training**: A well-trained model generalizes better.

---

## Parameter Estimation
### What Does It Mean?
In machine learning, **parameters** are the values that the model learns during training. **Parameter estimation** involves finding the optimal values for these parameters.

### Example:
In linear regression, the parameters are the **weights** and **bias** values that define the line of best fit.

---

## Probabilistic Modelling & Inference
### What Does It Mean?
- **Probabilistic Modelling**: Represents uncertainty in data using probability distributions.
- **Inference**: Making predictions or decisions based on the probabilistic model.

### Example:
In spam detection, a probabilistic model might calculate:
- Probability(email = "spam") = 0.85.
- Probability(email = "not spam") = 0.15.
The model then predicts "spam."

---

# Natural Language Processing (NLP)

## Problems & Perspectives
### What Does It Mean?
**NLP** is a field of AI that focuses on teaching machines to understand and process human language.

### Problems:
1. Text classification (e.g., spam detection).
2. Sentiment analysis (e.g., identifying positive or negative reviews).
3. Machine translation (e.g., translating English to French).

### Perspectives:
- NLP bridges the gap between human communication and machine understanding.

---

## Evaluation of NLP Applications
### What Does It Mean?
Evaluating NLP models involves checking their accuracy and effectiveness in solving language-based tasks.

### Key Metrics:
1. **Accuracy**: Correct predictions / Total predictions.
2. **Precision**: True Positives / (True Positives + False Positives).
3. **Recall**: True Positives / (True Positives + False Negatives).
4. **F1 Score**: A balance between precision and recall.

---

