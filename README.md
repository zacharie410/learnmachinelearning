### Python Machine Learning Tutorial: Introductory Lesson

**Objective**: Introduce machine learning using Python, starting with a single neuron, progressing to a full neural network to solve complex problems.

---

**1. Setup**

- **Install Python & Libraries**: Ensure Python is installed. Use pip to install required libraries: `numpy`, `tensorflow`.

---

### Step 2: Neuron Creation

#### Objective:
- Create a simple neuron using Python.
- Use the neuron to perform a basic task: predicting the outcome of a linear equation.

---

### 1. Neuron Creation: Predicting a Simple Linear Relationship

**Task**: Given `y = 2x + 1`, predict `y` for any `x`.

**Code**:

```python
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
```

---

### 2. Understanding Sets and Matrices

**Concept**:
- **Set**: Collection of distinct values. In machine learning, a dataset is a set of input-output pairs.
- **Matrix**: A rectangular array of numbers. Used to represent weights in neural networks.

---

### 3. Expanding to Neural Networks

**Building Blocks**:
- **Input Layer**: Takes input data.
- **Hidden Layer(s)**: Layers between input and output, where computations happen.
- **Output Layer**: Provides the final prediction.

**Math Behind**:
- **Matrix Multiplication**: In a network, input is multiplied by a weight matrix, then added to a bias vector.
- **Activation Functions**: Applied to introduce non-linearity.

**Next Steps**: Build a network using these concepts to solve more complex problems. 

---

### Step 3: Neural Network Creation

#### Objective:
- Create a simple neural network using Python.
- Use the network to perform a basic task: predicting the outcome based on multiple inputs.

---

### 1. Neural Network Creation: Solving a Simple Classification Task

**Task**: Given two inputs, classify whether the output is 0 or 1.

**Code**:

```python
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

```

---

### 2. Key Definitions

**Building Blocks**:
- **Neural Network**: A collection of neurons organized into layers. Each layer processes inputs and passes the output to the next layer.
- **Input Layer**: The first layer that receives input data.
- **Hidden Layer(s)**: Intermediate layers where computation happens. Networks can have multiple hidden layers.
- **Output Layer**: The final layer that provides the network's prediction.

---

### 3. Math Behind Neural Networks

**Matrix Multiplication**:
- **Weights as Matrices**: In neural networks, weights are represented as matrices. Input data is multiplied by these matrices to generate outputs for each layer.
- **Bias Vector**: A vector added to the weighted sum to shift the output, ensuring the model can fit the data better.

**Activation Functions**:
- **Sigmoid Function**: Maps input values to a range between 0 and 1, introducing non-linearity to the model, which allows it to learn complex patterns.
- **ReLU (Rectified Linear Unit)**: Often used in hidden layers, this function returns the input if positive, otherwise zero, helping to solve the vanishing gradient problem.

---

### 4. Applying Neural Networks

**Classification Example**:
- **Binary Classification**: The network learns to distinguish between two classes (e.g., 0 or 1) by adjusting weights and biases during training.

**Next Steps**:
- Expand the network to solve more complex problems by increasing the number of layers and neurons.
- Introduce additional concepts like loss functions, backpropagation, and optimization algorithms.

---

### Step 4: Building Blocks and Core Concepts of Neural Networks

#### Objective:
- Understand key components like gradients, backpropagation, and loss functions.
- Simplify these concepts into approachable building blocks and mathematical principles.

---

### 1. Key Building Blocks

**1.1 Neuron**:
- **Definition**: A basic unit of computation that receives input, applies weights and bias, then passes the result through an activation function to produce output.
- **Role**: Forms the foundation of a neural network. Each neuron processes a small part of the data.

**1.2 Layers**:
- **Input Layer**: First layer, receives raw data.
- **Hidden Layers**: Intermediate layers where complex processing happens.
- **Output Layer**: Final layer, provides predictions or classifications.

**1.3 Weights and Biases**:
- **Weights**: Parameters multiplied with input data, determining the importance of each input.
- **Bias**: A parameter added to the weighted input, allowing the model to fit data better.

**1.4 Activation Functions**:
- **Purpose**: Introduce non-linearity into the model, enabling it to learn and represent complex patterns.
- **Common Functions**:
    - **Sigmoid**: Maps input to a range between 0 and 1.
    - **ReLU**: Returns the input if positive, otherwise zero. Commonly used in hidden layers.

**1.5 Loss Function**:
- **Definition**: Measures how well the network’s predictions match the actual values.
- **Example**: Mean Squared Error (MSE) for regression, Cross-Entropy Loss for classification.

**1.6 Gradients**:
- **Definition**: Partial derivatives of the loss function with respect to each weight. They show the direction and rate of change of the loss.
- **Role**: Used in gradient descent to update weights and minimize the loss.

**1.7 Backpropagation**:
- **Definition**: An algorithm used to calculate gradients by propagating the error backward through the network.
- **Role**: Essential for training, allowing the network to learn by adjusting weights based on error.

**1.8 Optimization Algorithms**:
- **Gradient Descent**: A method to minimize the loss by iteratively adjusting weights in the opposite direction of the gradient.
- **Variants**: 
    - **Stochastic Gradient Descent (SGD)**: Updates weights for each data point, leading to faster convergence.
    - **Adam**: An advanced optimizer combining momentum and adaptive learning rates for efficient training.

---

### 2. The Math Behind Neural Networks

**2.1 Gradient Calculation**:
- **Gradient of the Loss**: 
    - Compute the derivative of the loss function with respect to each weight.
    - Example: For a simple neuron, if `loss = (y - y_pred)^2`, the gradient with respect to weight `w` is `-2(x)(y - y_pred)`.

**2.2 Backpropagation Algorithm**:
- **Forward Pass**: Calculate the output of the network by passing input through all layers.
- **Backward Pass**:
    - Calculate the gradient of the loss with respect to the output.
    - Propagate this gradient back through each layer, adjusting the weights and biases.

**2.3 Updating Weights**:
- **Gradient Descent**:
    - Adjust weights using the formula: `weight = weight - learning_rate * gradient`.
    - This reduces the loss iteratively, improving the model’s accuracy.

**2.4 Optimization**:
- **Minimizing Loss**:
    - The goal is to find the set of weights that minimizes the loss function.
    - Optimizers like Adam use adaptive learning rates and momentum to speed up this process.

---

### 3. Applying These Concepts

**3.1 Training Process**:
- **Data**: Provide input and correct output.
- **Forward Pass**: Calculate predictions.
- **Loss Calculation**: Compare predictions with actual output.
- **Backpropagation**: Calculate gradients and update weights.
- **Iteration**: Repeat the process until the loss is minimized.

**3.2 Example**: If training a network to recognize handwritten digits, these building blocks and mathematical principles work together to adjust the model’s parameters until it accurately identifies the digits.

---

## Visualization of Gradient Descent to find Local Minimum
![](/img/gradientdescentlocalminimum.jpg)

The idea of gradient descent is to make the neural network's predictions more accurate by adjusting the weights until it finds the fastest and most favorable result. Some algorithms use a big picture view of the entire loss gradient, while others are more optimized and only analyze small areas to make incremental gains.