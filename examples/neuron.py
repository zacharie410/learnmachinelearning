import numpy as np

# Define the neuron
def simple_neuron(x, weight, bias):
    return weight * x + bias

# Initial parameters (weight, bias)
weight = 0.0
bias = 0.0

# Learning rate
learning_rate = 0.01

# Data: x (input) and y (output)
data = np.array([
    [1, 3],   # y = 2(1) + 1 = 3
    [2, 5],   # y = 2(2) + 1 = 5
    [3, 7],   # y = 2(3) + 1 = 7
    [4, 9]    # y = 2(4) + 1 = 9
])

# Training the neuron
for epoch in range(1000):
    for x, y in data:
        y_pred = simple_neuron(x, weight, bias)
        error = y - y_pred

        # Update weight and bias
        weight += learning_rate * error * x
        bias += learning_rate * error

# After training, check the final weight and bias
print(f"Trained weight: {weight}, Trained bias: {bias}")

# Predicting with the trained neuron
x_test = 5
y_test = simple_neuron(x_test, weight, bias)
print(f"For x = {x_test}, predicted y = {y_test}")