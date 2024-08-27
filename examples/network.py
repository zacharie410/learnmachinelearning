import numpy as np

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Neural network with one hidden layer
class BasicNeuralNetwork:
    def __init__(self):
        # Initialize weights and biases using Xavier initialization
        self.input_to_hidden_weights = np.random.randn(2, 4) * np.sqrt(2 / 2)
        self.hidden_to_output_weights = np.random.randn(4, 1) * np.sqrt(2 / 4)
        self.hidden_bias = np.zeros(4)
        self.output_bias = np.zeros(1)

    def forward(self, x):
        # Forward pass
        self.hidden_layer_input = np.dot(x, self.input_to_hidden_weights) + self.hidden_bias
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.hidden_to_output_weights) + self.output_bias
        self.output = sigmoid(self.output_layer_input)
        return self.output

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute the error
            error = y - output
            
            # Backward pass
            # Output layer gradients
            d_output = error * sigmoid_derivative(output)
            d_hidden_to_output_weights = np.dot(self.hidden_layer_output.T, d_output)
            d_output_bias = np.sum(d_output, axis=0, keepdims=True)

            # Hidden layer gradients
            d_hidden_layer = np.dot(d_output, self.hidden_to_output_weights.T) * sigmoid_derivative(self.hidden_layer_output)
            d_input_to_hidden_weights = np.dot(X.T, d_hidden_layer)
            d_hidden_bias = np.sum(d_hidden_layer, axis=0)

            # Update weights and biases
            self.hidden_to_output_weights += learning_rate * d_hidden_to_output_weights
            self.output_bias += learning_rate * d_output_bias.flatten()
            self.input_to_hidden_weights += learning_rate * d_input_to_hidden_weights
            self.hidden_bias += learning_rate * d_hidden_bias

            # Optional: Print loss every 1000 epochs to track progress
            if epoch % 1000 == 0:
                loss = np.mean(np.square(error))
                print(f'Epoch {epoch}, Loss: {loss}')

# Data: X (input) and y (output)
X = np.array([
    [0, 0],   # Expect 0
    [0, 1],   # Expect 1
    [1, 0],   # Expect 1
    [1, 1]    # Expect 0
])
y = np.array([
    [0],   # Expect 0
    [1],   # Expect 1
    [1],   # Expect 1
    [0]    # Expect 0
])

# Create and train the network
nn = BasicNeuralNetwork()
nn.train(X, y, learning_rate=0.5, epochs=10000)
 # What happens when you increase or decrease the amount of epochs?

# Test the network
for x in X:
    print(f"For input {x}, predicted output: {nn.forward(x)}")
