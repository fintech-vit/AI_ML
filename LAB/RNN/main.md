# **Recurrent Neural Networks (RNNs) – A Comprehensive Guide**  
- https://colab.research.google.com/drive/1GN_pkxXkrMbZwqZ_zs_PC7fYIk5hfai6?usp=sharing

Recurrent Neural Networks (RNNs) are a class of neural networks designed to handle **sequential data**. Unlike traditional feedforward networks, RNNs retain memory of past inputs, making them suitable for tasks such as **time series forecasting, natural language processing (NLP), and speech recognition**.

---

## **1. Why Use RNNs?**  

RNNs are useful for processing data where the order of inputs matters. Traditional neural networks process inputs independently, while RNNs remember past information and use it to influence current predictions.

### **Common Applications of RNNs:**  
- **Text Processing** – Sentiment analysis, text generation, machine translation  
- **Speech Recognition** – Voice assistants, speech-to-text systems  
- **Time Series Prediction** – Stock price forecasting, weather prediction  

---

## **2. Structure of an RNN**  

An RNN processes input data sequentially by passing the output of one time step as input to the next.  

### **Key Components of an RNN:**  
- **Input (`x_t`)** – The input at time step `t`.  
- **Hidden State (`h_t`)** – Stores memory from previous time steps.  
- **Weights (`W_h`, `W_x`, `W_y`)** – Parameters that determine how inputs influence the hidden state and output.  
- **Output (`y_t`)** – The prediction at time step `t`.  

### **Mathematical Formulation**  

At each time step `t`, the RNN updates its hidden state using the formula:  

\[
h_t = \tanh(W_x x_t + W_h h_{t-1} + b)
\]

where:  
- `h_t` = current hidden state  
- `h_{t-1}` = previous hidden state  
- `x_t` = current input  
- `W_x, W_h` = weight matrices  
- `b` = bias  

The output is computed as:  

\[
y_t = W_y h_t
\]

---

## **3. Implementing a Simple RNN in PyTorch**  

This section demonstrates how to build a basic RNN using PyTorch.

### **3.1 Importing Required Libraries**  
```python
import torch
import torch.nn as nn
import torch.optim as optim
```

---

### **3.2 Defining the RNN Model**  
```python
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])  # Take the last time step output
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
```

### **Explanation:**  
- The `RNN` layer processes sequential data.  
- The `Linear` layer maps the hidden state to the output.  
- The hidden state is initialized to zeros at the start.

---

### **3.3 Creating Sample Data and Training**  
```python
# Hyperparameters
input_size = 1
hidden_size = 10
output_size = 1
batch_size = 5

# Initialize model
model = SimpleRNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Sample input (batch_size=5, sequence_length=10, input_size=1)
sample_input = torch.randn(batch_size, 10, input_size)
hidden = model.init_hidden(batch_size)

# Forward pass
output, hidden = model(sample_input, hidden)
print("Output Shape:", output.shape)
```

---

## **4. Challenges with RNNs**  

### **Vanishing Gradient Problem**  
When training deep RNNs, gradients become smaller during backpropagation, making it difficult to update earlier layers. This leads to poor learning for long sequences.

### **Long-Term Dependencies**  
RNNs struggle to remember information from distant past time steps. When a sequence is long, information from the earlier part may be lost by the time the later steps are reached.

### **Solutions to These Problems**  
- **Use LSTM (Long Short-Term Memory) networks** – LSTMs have mechanisms like forget gates that help retain long-term dependencies.  
- **Use GRU (Gated Recurrent Unit) networks** – GRUs are a simpler alternative to LSTMs and help mitigate vanishing gradients.  

---

## **5. Visualizing RNN Training Performance**  

To monitor the training progress, plotting the loss curve helps understand how well the model is learning.

```python
import matplotlib.pyplot as plt

# Simulated loss values
loss_values = [0.9, 0.75, 0.65, 0.50, 0.40, 0.35, 0.28, 0.22, 0.18, 0.12]

# Plot loss curve
plt.plot(loss_values, marker='o', linestyle='-')
plt.title("RNN Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
```

---

## **Conclusion**  

- RNNs are effective for sequential tasks but struggle with long-term dependencies.  
- LSTMs and GRUs are improved architectures that address these issues.  
- PyTorch provides an efficient way to build and train RNN models.  