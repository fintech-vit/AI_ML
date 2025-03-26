# **Understanding the Relationship Between Credit Limit and Average Purchase**
- https://colab.research.google.com/drive/1PZQytSNf2r3DZiz5LCDg3_G1HODT9PS2?usp=sharing

## **1. Introduction**
In financial analysis, it is essential to understand how different factors influence customer spending behavior. One such relationship is between **credit limit** and **average purchase amount**. This study aims to determine whether customers with higher credit limits tend to spend more on average.

## **2. Objective**
The main objective of this analysis is to explore the relationship between a customer's **credit limit** and their **average purchase amount**. Specifically, we will:
- Generate a dataset of **150 customers** with relevant financial attributes.
- Create a **bivariate plot** (scatter plot) to visualize the relationship.
- Compute the **correlation coefficient** to measure the strength of the relationship.
- Classify customers based on their spending behavior.
- Identify potential **outliers** that deviate significantly from expected spending patterns.

---

## **3. Understanding Key Terms**
### **3.1. Bivariate Analysis and Bivariate Plot**
- **Bivariate analysis** refers to the statistical analysis of two variables to determine the relationship between them.
- A **bivariate plot** is a graphical representation of the relationship between two numerical variables. In this case, we use a **scatter plot** to show how credit limit and average purchase amount are related.

### **3.2. Correlation**
- **Correlation** measures the degree to which two variables move in relation to each other.
- The **Pearson correlation coefficient (r)** is a commonly used measure:
  - `r > 0`: Positive correlation (when one variable increases, the other also increases).
  - `r < 0`: Negative correlation (when one variable increases, the other decreases).
  - `r = 0`: No correlation (the variables do not affect each other).
- A strong correlation (closer to **1 or -1**) suggests a strong relationship, whereas a weak correlation (closer to **0**) suggests little or no relationship.

### **3.3. Outliers**
- **Outliers** are extreme values that deviate significantly from the overall pattern of the data. Identifying outliers helps in understanding unusual spending behavior.

---

## **4. Approach**
We follow a structured approach to analyze the relationship:

### **Step 1: Generate Synthetic Data**
Since we do not have real customer data, we generate a dataset containing 150 customers with randomly assigned financial attributes:
- **Customer_ID**: A unique identifier for each customer.
- **Credit_Limit**: The maximum amount a customer can spend using their credit card.
- **Average_Purchase**: The average amount the customer spends per transaction.
- **Income**: A feature to analyze spending behavior based on earnings.

### **Step 2: Save Data to CSV**
The dataset is saved as `customer_credit_data.csv` for further analysis.

### **Step 3: Create a Bivariate Scatter Plot**
A **scatter plot** is created to visualize the relationship between **credit limit** and **average purchase amount**. A **trendline** is added to observe the general pattern.

### **Step 4: Compute Correlation**
We calculate the **correlation coefficient** between credit limit and average purchase to measure the strength of their relationship.

### **Step 5: Categorize Customers Based on Spending Behavior**
Customers are divided into three categories:
- **Low Spenders** (lower one-third of average purchases).
- **Medium Spenders** (middle one-third of average purchases).
- **High Spenders** (upper one-third of average purchases).

### **Step 6: Outlier Detection**
A **boxplot** is used to detect any customers whose spending deviates significantly from the average.

### **Step 7: Multi-Feature Analysis**
A **pairplot** is created to visualize interactions among multiple financial attributes.

---

## **5. Step-by-Step Implementation**
Below is the complete Python code implementing the approach:

### **Step 1: Generate the Dataset**
```python
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of customers
num_customers = 150

# Generate synthetic data
data = {
    "Customer_ID": range(1, num_customers + 1),
    "Credit_Limit": np.random.randint(5000, 50000, num_customers),  # Random limits between $5,000 and $50,000
    "Average_Purchase": np.random.randint(500, 5000, num_customers),  # Random spending between $500 and $5,000
    "Income": np.random.randint(20000, 150000, num_customers)  # Random income levels between $20,000 and $150,000
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("customer_credit_data.csv", index=False)

# Display first few rows
df.head()
```
```
Output:
   	Customer_ID	Credit_Limit	Average_Purchase	Income
0	1	            20795	            4127	     26102
1	2	            5860	            1863	     70336
2	3	            43158	            2481	     138015
3	4	            49732	            2163	     105314
4	5	            16284	            2029	     143007
```
---

### **Step 2: Create a Bivariate Scatter Plot**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

# Create scatter plot
plt.figure(figsize=(8, 6))
sns.regplot(x="Credit_Limit", y="Average_Purchase", data=df, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})

# Labels and title
plt.xlabel("Credit Limit ($)")
plt.ylabel("Average Purchase ($)")
plt.title("Bivariate Analysis: Credit Limit vs. Average Purchase")

# Show plot
plt.show()
```
![alt text](image.png)
---

### **Step 3: Compute Correlation**
```python
# Compute Pearson correlation coefficient
correlation = df["Credit_Limit"].corr(df["Average_Purchase"])

# Print correlation result
print(f"Correlation between Credit Limit and Average Purchase: {correlation:.2f}")
```
```
Output:
Correlation between Credit Limit and Average Purchase: -0.10
```

---

### **Step 4: Categorize Customers Based on Spending Behavior**
```python
# Divide customers into Low, Medium, and High spenders based on average purchase
df["Spending_Category"] = pd.qcut(df["Average_Purchase"], q=3, labels=["Low", "Medium", "High"])

# Display first few rows
df.head()
```
```
Output:

    Customer_ID	Credit_Limit	Average_Purchase	Income	Spending_Category
0	    1	        20795	            4127	    26102	    High
1	    2	        5860	            1863	    70336	    Low
2	    3	        43158	            2481	    138015	    Medium
3	    4	        49732	            2163	    105314	    Low
4	    5	        16284	            2029	    143007	    Low


---

### **Step 5: Outlier Detection Using Boxplot**
```python
# Create boxplot for average purchases
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Average_Purchase"])

# Labels and title
plt.xlabel("Average Purchase ($)")
plt.title("Distribution of Average Purchases")

# Show plot
plt.show()
```
![alt text](image-1.png)

---

### **Step 6: Multi-Feature Analysis Using Pairplot**
```python
# Create pairplot to visualize relationships across multiple features
sns.pairplot(df, hue="Spending_Category", palette="coolwarm")

# Show plot
plt.show()
```
![alt text](image-2.png)
---

## **6. Analysis of Results**
1. **Scatter Plot Interpretation**
   - If the scatter points form an upward trend, it indicates a **positive correlation** (higher credit limits lead to higher spending).
   - If the points are scattered randomly, it means there is **no strong correlation**.
   - If the points form a downward trend, it suggests a **negative correlation** (higher credit limits lead to lower spending).

2. **Correlation Interpretation**
   - A **high positive value (close to 1)** suggests that customers with higher credit limits tend to spend more.
   - A **low or zero correlation** suggests that credit limit does not significantly impact spending behavior.

3. **Boxplot Insights**
   - If there are **outliers**, it means some customers are spending much more or much less than the general trend.

4. **Spending Categories**
   - **Low spenders** may be those who are more financially cautious or have lower financial needs.
   - **High spenders** might be customers who frequently use their credit cards for large purchases.

---

## **7. Conclusion**
- This analysis helps in understanding whether increasing a customer’s credit limit leads to higher spending.
- Financial institutions can use these insights to optimize **credit policies** and identify **potentially risky customers** who may default on payments.
- Further analysis can be performed by considering **credit utilization ratios, payment history, and debt-to-income ratios**.

