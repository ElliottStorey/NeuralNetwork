import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    pass

def mean_squared_error(output: list[np.ndarray], predicted: list[np.ndarray]) -> float:
    return (output - predicted)**2

def mean_squared_error_derivative(output: np.ndarray, predicted: np.ndarray) -> float:
    pass

class NeuralNetwork:
    def activate(self, x):
        return sigmoid(x)
    
    def activate_derivative(self, x):
        return sigmoid_derivative(x)
    
    def cost(self, x):
        return mean_squared_error(x)
    
    def cost_derivative(self, x):
        return mean_squared_error_derivative(x)
        
    def __init__(self, dimensions: tuple[int]) -> None:
        """
        Initialize the neural network with the given dimensions of input, hidden, and output layers.
        Initialize weights and biases.

        Args:
        - dimensions (tuple[int]): Tuple of integers representing the number of neurons in each layer.
                                   The first element is the input layer size, the last element is the output layer size,
                                   and the elements in between are the sizes of the hidden layers.
        """

        self.dimensions = dimensions
        self.weights = []
        self.biases = []
        for i in range(1, len(dimensions)):
            # Initialize weights using He initialization
            weight = np.random.standard_normal((dimensions[i-1], dimensions[i])) * np.sqrt(2 / dimensions[i-1])
            self.weights.append(weight)

            # Initialize biases as zeroes
            bias = np.zeros((1, dimensions[i]))
            self.biases.append(bias)

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass through the neural network.
        Calculate the outputs based on the input and current weights and biases.

        Args:
        - input (np.ndarray): Matrix of input values to the network.

        Returns:
        - output (np.ndarray): Matrix of output values from the network.
        """

        for i in range(len(self.dimensions) - 1):
            output = self.activate(input @ self.weights[i] + self.biases[i])
            input = output
        return output

    def backward(self, output: list[np.ndarray], predicted: list[np.ndarray], learning_rate: float) -> None:
        """
        Perform a backward pass through the neural network.
        Update the weights and biases based on the calculated gradients.

        Args:
        - predicted (list[np.ndarray]): List of predicted values from the network.
        - output (list[np.ndarray]): List of output values for the corresponding input.
        """

        #W -= rate * (dcost/dW)
        for i in reversed(range(len(self.dimensions) - 1)):
            weight_gradient = input.T @ (self.cost_derivative(output, predicted) * self.activate_derivative(predicted))
            bias_gradient = self.cost_derivative(output, predicted) * self.activate_derivative(predicted)

            # Update the weights and biases
            self.weights[i] -= learning_rate * weight_gradient
            self.biases[i] -= learning_rate * bias_gradient

    def train(self, inputs: list[np.ndarray], outputs: list[np.ndarray], epochs: int, learning_rate: float = 1.0) -> None:
        """
        Train the neural network using the given inputs and expected outputs.
        Perform forward and backward passes for the specified number of epochs,
        adjusting the weights and biases using the specified learning rate.

        Args:
        - inputs (list[list[float]]): List of input value lists for training.
        - expected (list[list[float]]): List of output value lists for the corresponding inputs.
        - epochs (int): Number of training epochs.
        - learning_rate (float): Learning rate for weight updates.
        """

        for epoch in range(epochs):

            for x, y in zip(inputs, outputs):
                predicted = self.forward(x)
                self.backward(x, predicted, y, learning_rate)