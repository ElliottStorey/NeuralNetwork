import numpy as np

class NeuralNetwork:
    def activate(self, x):
        return sigmoid(x)
    
    def activate_derivative(self, x):
        return sigmoid_derivative(x)
    
    def cost(self, actual_outputs: list[np.ndarray], predicted_outputs: list[np.ndarray]):
        return mean_squared_error(actual_outputs, predicted_outputs)
    
    def cost_derivative(self, actual_outputs: list[np.ndarray], predicted_outputs: list[np.ndarray]):
        return mean_squared_error_derivative(actual_outputs, predicted_outputs)
        
    def __init__(self, dimensions: tuple[int]) -> None:
        self.dimensions = dimensions
        self.weights = []
        self.biases = []
        for i in range(1, len(dimensions)):
            # He initialization
            weight = np.random.standard_normal((dimensions[i-1], dimensions[i])) * np.sqrt(2 / dimensions[i-1])
            self.weights.append(weight)

            # Zero initialization
            bias = np.zeros((1, dimensions[i]))
            self.biases.append(bias)

    def forward(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        predicted_outputs = []
        for input in inputs:
            for i in range(len(self.dimensions) - 1):
                predicted_output = self.activate(input @ self.weights[i] + self.biases[i])
                input = predicted_output
            predicted_outputs.append(predicted_output)
        return predicted_outputs

    def backward(self, actual_outputs: list[np.ndarray], predicted_outputs: list[np.ndarray], learning_rate: float) -> None:
        # Convert lists to numpy arrays
        actual_outputs = np.array(actual_outputs)
        predicted_outputs = np.array(predicted_outputs)
        
        # Calculate the error and delta for the output layer
        output_delta = (actual_outputs - predicted_outputs) * sigmoid_derivative(predicted_outputs)
        
        # Calculate the error and delta for the hidden layers
        deltas = [output_delta]
        for layer in range(len(self.dimensions) - 2, 0, -1):
            delta = np.dot(self.weights[layer].T, deltas[-1]) * sigmoid_derivative(self.hidden_layer_activations[layer])
            deltas.append(delta)
        
        # Reverse the deltas list to align with the layer indices
        deltas.reverse()
        
        # Update the weights and biases for each layer
        for layer in range(self.num_layers - 1, 0, -1):
            self.weights[layer-1] += np.dot(deltas[layer-1], self.hidden_layer_activations[layer-1].T) * learning_rate
            self.biases[layer-1] += np.sum(deltas[layer-1], axis=1, keepdims=True) * learning_rate

    def train(self, inputs: list[np.ndarray], actual_outputs: list[np.ndarray], epochs: int, learning_rate: float = 1.0) -> None:
        for epoch in range(epochs):
            predicted_outputs = self.forward(inputs)
            self.backward(actual_outputs, predicted_outputs, learning_rate)

# math functions

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mean_squared_error(actual_output: np.ndarray, predicted_output: np.ndarray) -> float:
    return (actual_output - predicted_output)**2

def mean_squared_error_derivative(actual_output: np.ndarray, predicted_output: np.ndarray) -> float:
    return 2 * (predicted_output - actual_output)