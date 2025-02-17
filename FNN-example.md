Here’s an example of training a **Feed-Forward Neural Network (FNN)** using **PyTorch** to predict house prices based on features like square footage, number of bedrooms, and location.  

---

## **Step 1: Import Libraries**
We will use **PyTorch** for defining the neural network, **NumPy & Pandas** for data handling, and **Matplotlib** for visualization.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

---

## **Step 2: Create a Sample Housing Dataset**
For simplicity, let's generate synthetic data.

```python
# Create a dataset with 3 features: Square Footage, Number of Bedrooms, and Location Score
np.random.seed(42)
num_samples = 1000

square_footage = np.random.randint(800, 5000, num_samples)  # House size in square feet
bedrooms = np.random.randint(1, 6, num_samples)  # Number of bedrooms
location_score = np.random.randint(1, 10, num_samples)  # A score representing neighborhood quality

# Generate house prices with some noise
house_prices = 50000 + (square_footage * 150) + (bedrooms * 10000) + (location_score * 5000) + np.random.randn(num_samples) * 5000

# Convert to DataFrame
df = pd.DataFrame({'SquareFootage': square_footage, 'Bedrooms': bedrooms, 'LocationScore': location_score, 'Price': house_prices})

# Display first 5 rows
print(df.head())
```

---

## **Step 3: Prepare the Data**
We need to:
1. Normalize the features.
2. Split the data into training and test sets.
3. Convert the NumPy arrays into PyTorch tensors.

```python
# Split into features (X) and target (y)
X = df[['SquareFootage', 'Bedrooms', 'LocationScore']].values
y = df['Price'].values

# Standardize features (important for neural networks)
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Normalize feature values

# Convert data to PyTorch tensors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Reshape for PyTorch
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
```

---

## **Step 4: Define the Feed-Forward Neural Network**
We'll create a simple FNN with:
- **3 input neurons** (corresponding to the three features)
- **2 hidden layers** with ReLU activation
- **1 output neuron** for predicting price  

```python
class HousePriceNN(nn.Module):
    def __init__(self):
        super(HousePriceNN, self).__init__()
        self.hidden1 = nn.Linear(3, 10)  # 3 input features → 10 hidden neurons
        self.hidden2 = nn.Linear(10, 5)  # 10 hidden neurons → 5 neurons
        self.output = nn.Linear(5, 1)  # 5 neurons → 1 output (house price)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)  # No activation in output for regression
        return x

# Instantiate the model
model = HousePriceNN()
```

---

## **Step 5: Define Loss Function and Optimizer**
- Since this is a regression task, we use **Mean Squared Error (MSE) Loss**.
- The **Adam optimizer** is used to update weights.

```python
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

---

## **Step 6: Train the Model**
The training loop consists of:
1. Forward pass: Predict house prices.
2. Compute loss: Compare predictions with actual values.
3. Backward pass: Update model weights.

```python
num_epochs = 500
loss_history = []

for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    
    # Backward pass
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()
    optimizer.step()  # Update weights
    
    # Store loss for visualization
    loss_history.append(loss.item())

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.2f}')
```

---

## **Step 7: Visualizing Training Progress**
Let's plot the loss over epochs to check if the model is learning properly.

```python
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()
```

---

## **Step 8: Evaluate the Model**
We now test our model on unseen data.

```python
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    y_pred = model(X_test_tensor)

# Convert tensors to NumPy arrays
y_pred_np = y_pred.numpy().flatten()
y_test_np = y_test_tensor.numpy().flatten()

# Compare actual vs. predicted prices
plt.scatter(y_test_np, y_pred_np, alpha=0.5)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs. Predicted Prices")
plt.show()
```

---

## **Step 9: Making New Predictions**
Now, let's predict the price of a new house with given features.

```python
# Example: Predict the price of a house with 2500 sqft, 3 bedrooms, and location score 7
new_house = np.array([[2500, 3, 7]])
new_house_scaled = scaler.transform(new_house)  # Standardize using the same scaler

# Convert to PyTorch tensor
new_house_tensor = torch.tensor(new_house_scaled, dtype=torch.float32)

# Make prediction
model.eval()
predicted_price = model(new_house_tensor).item()
print(f'Predicted Price: ${predicted_price:.2f}')
```

---

## **Conclusion**
This simple **Feed-Forward Neural Network** learns how to predict house prices based on input features like square footage, number of bedrooms, and location score. We:
1. Created a **synthetic dataset** for house prices.
2. **Preprocessed** the data (normalization and splitting).
3. Defined a **neural network** with two hidden layers.
4. **Trained** the model using **backpropagation and gradient descent**.
5. **Evaluated** the model using a test dataset and visualized predictions.

This model can be further improved with **more data, feature engineering, and hyperparameter tuning**.

Let me know if you have any questions or need modifications!