**Predictive Analytics for Stock Prices** using **Linear Regression**.  
- https://colab.research.google.com/drive/1pzW6SoOo32dMImY7_i6yeWjJV1XLQgd1?usp=sharing
---

## **Step 1: Import Necessary Libraries**
We first import essential Python libraries:  

- `yfinance` – To fetch stock market data.  
- `pandas` – For data manipulation and analysis.  
- `numpy` – For numerical operations.  
- `matplotlib` – For visualizing the stock price trends.  
- `scikit-learn` – To split the dataset, train a **Linear Regression** model, and make predictions.

```python
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
```

---

## **Step 2: Download Stock Data**  
We use `yfinance` to fetch historical stock price data.  
For this example, we fetch **Apple Inc. (AAPL)** stock prices from **January 1, 2020, to January 1, 2024**.

```python
# Define the stock ticker symbol
ticker = "AAPL"

# Download historical stock data
stock_data = yf.download(ticker, start="2020-01-01", end="2024-01-01")

# Display first few rows
print(stock_data.head())
```

💡 **Output Explanation:**  
This dataset includes columns like **Open, High, Low, Close, Adj Close, and Volume**.  
We focus on the **"Close"** price, which represents the stock's final price at the end of each trading day.

---

## **Step 3: Prepare the Dataset**  
Since stock prices are time-series data, we convert dates into numerical values to use them as features.

```python
# Reset index to extract the date column
stock_data["Date"] = stock_data.index

# Convert the Date column into number of days since the start
stock_data["Days"] = (stock_data["Date"] - stock_data["Date"].min()).dt.days

# Selecting features (X) and target variable (y)
X = stock_data["Days"].values.reshape(-1, 1)  # Independent variable (Days)
y = stock_data["Close"].values  # Dependent variable (Stock Price)

# Display first few values
print(X[:5], y[:5])
```

💡 **Explanation:**  
- We **convert the dates into numeric values** (days since the first date) because regression models require numerical input.
- `X` contains the transformed date values (independent variable).  
- `y` contains the **closing stock prices** (dependent variable).

---

## **Step 4: Split the Data into Training & Testing Sets**  
To evaluate our model, we split the data into **80% training** and **20% testing**.

```python
# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display dataset sizes
print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")
```

💡 **Explanation:**  
- **Training Data (80%)** – Used to train the model.  
- **Testing Data (20%)** – Used to check how well the model predicts stock prices.  

---

## **Step 5: Train the Linear Regression Model**
We use **Linear Regression**, which models the relationship between **days** and **stock prices**.

```python
# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Display model coefficients
print(f"Model Coefficient (Slope): {model.coef_[0]}")
print(f"Model Intercept: {model.intercept_}")
```

💡 **Explanation:**  
- The **coefficient (slope)** represents how much the stock price changes per day.  
- The **intercept** is the predicted price at day 0.  

---

## **Step 6: Make Predictions**
Now, we use our trained model to predict stock prices.

```python
# Make predictions on the test set
y_pred = model.predict(X_test)

# Display some predictions
comparison = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(comparison.head(10))
```

💡 **Explanation:**  
- We compare the **actual stock prices** vs. **predicted stock prices**.

---

## **Step 7: Evaluate the Model**  
We use **Mean Absolute Error (MAE)** and **Mean Squared Error (MSE)** to measure how well the model performs.

```python
# Calculate MAE and MSE
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
```

💡 **Explanation:**  
- **MAE**: The average absolute difference between actual and predicted prices.  
- **MSE**: The average squared difference, penalizing larger errors.

---

## **Step 8: Visualizing the Results**  
We **plot actual vs. predicted prices** to see how well the model fits.

```python
plt.figure(figsize=(10, 5))

# Scatter plot of actual prices
plt.scatter(X_test, y_test, color="blue", label="Actual Prices")

# Line plot of predicted prices
plt.plot(X_test, y_pred, color="red", label="Predicted Prices")

# Labels and title
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.legend()
plt.title(f"{ticker} Stock Price Prediction")

# Show the plot
plt.show()
```

💡 **Explanation:**  
- **Blue dots** represent actual prices.  
- **Red line** represents predicted prices.  

---

---

## **Next Steps & Improvements**
✅ This model is a **basic predictive analysis** using **Linear Regression**.  
🚀 **To improve accuracy**, you can:  
- Use **LSTMs (Long Short-Term Memory networks)** for better time-series forecasting.  
- Include **technical indicators** like Moving Averages, RSI, and MACD.  
- Experiment with **Random Forests, ARIMA, or XGBoost** for better predictions.

