# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .
### Neural Network Model
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/7dd91b9e-6aad-4d4d-be74-811281d768b6" />


## PROGRAM

### Name: Cholimgapuram Sai Likitha

### Register Number:212224230046

```
**
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

torch.manual_seed(71)
X = torch.linspace(1, 50, 50).reshape(-1, 1)
e = torch.randint(-8, 9, (50, 1), dtype=torch.float)
y = 2 * X + 1 + e

plt.scatter(X, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Dataset')
plt.show()

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

torch.manual_seed(59)
model = Model(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

epochs = 50
losses = []

for epoch in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, "
          f"Weight={model.linear.weight.item():.4f}, "
          f"Bias={model.linear.bias.item():.4f}")

plt.plot(range(1, epochs+1), losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss vs Epoch')
plt.show()

x_line = np.linspace(0, 50, 50)
current_weight = model.linear.weight.item()
current_bias = model.linear.bias.item()
y_line = current_weight * x_line + current_bias

plt.scatter(X, y, label='Original Data')
plt.plot(x_line, y_line, 'r', label='Best Fit Line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

new_input = torch.tensor([[60.0]])
prediction = model(new_input)

print("\nSample Input:", new_input.item())
print("Predicted Output:", prediction.item())



# Initialize the Model, Loss Function, and Optimizer

torch.manual_seed(59)
model = Model(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

```

### Dataset Information
Include screenshot of the generated data
<img width="831" height="260" alt="image" src="https://github.com/user-attachments/assets/a4bd7848-0904-412c-802f-0f66c405a3ad" />

### OUTPUT
#### Training Loss Vs Iteration Plot
<img width="772" height="558" alt="image" src="https://github.com/user-attachments/assets/44e0e9d1-ee4a-4e4d-a441-333b8a04f5d2" />

#### Best Fit line plot
<img width="753" height="572" alt="image" src="https://github.com/user-attachments/assets/02fa47f2-6f5b-40a0-9c76-8271d1a1d823" />

#### Include your plot here
<img width="772" height="582" alt="image" src="https://github.com/user-attachments/assets/42336e1d-ced0-4f8e-a923-4a672820201f" />

#### New Sample Data Prediction
<img width="786" height="276" alt="image" src="https://github.com/user-attachments/assets/c367a092-70bd-49e8-a768-f680e23857ee" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
