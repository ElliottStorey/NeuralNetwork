import numpy as np

activation_functions = {
    'sigmoid': lambda x : 1 / (1 + np.exp(-x)),
    'sigmoid_derivative': lambda x : 1 / (1 + np.exp(-x)) * (1 - (1 / (1 + np.exp(-x))))
}

cost_functions = {
    'mean_squared_error': lambda actual_output, predicted_output : (actual_output - predicted_output)**2,
    'mean_squared_error_derivative': lambda actual_output, predicted_output : 2 * (predicted_output - actual_output)
}

class NeuralNetwork:
    def __init__(self, dimensions: tuple[int], activation_function = 'sigmoid', cost_function = 'mean_squared_error') -> None:
        self.dimensions = dimensions
        self.weights = []
        self.biases = []
        self.activation_function = activation_functions[activation_function]
        self.activation_derivative = activation_functions[activation_function + '_derivative']
        self.cost_function = cost_functions[cost_function]
        self.cost_derivative = cost_functions[cost_function + '_derivative']
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
                predicted_output = self.activation_function(input @ self.weights[i] + self.biases[i])
                input = predicted_output
            predicted_outputs.append(predicted_output)
        return predicted_outputs

    def backward(self, actual_outputs: list[np.ndarray], predicted_outputs: list[np.ndarray], learning_rate: float) -> None:
        deltas = []
        delta = self.cost_derivative(actual_output, predicted_output) * self.activation_derivative(predicted_output)
        deltas.append(delta)

        for i in reversed(range(len(self.dimensions) - 1)):
            current_deltas = deltas[-1]
            current_activations = all_activations[i]

            delta = np.dot(current_deltas, self.weights[i].T) * self.activation_derivative(current_activations)
            deltas.append(delta)

        # Weight and bias updates
        for i in reversed(range(len(self.dimensions) - 1)):
            current_activations = all_activations[i]
            previous_activations = all_activations[i - 1] if i > 0 else input

            current_deltas = deltas.pop()

            weight_gradients = np.dot(previous_activations.T, current_deltas)
            bias_gradients = np.mean(current_deltas, axis=0)

            self.weights[i] -= learning_rate * weight_gradients
            self.biases[i] -= learning_rate * bias_gradients

    def train(self, inputs: list[np.ndarray], actual_outputs: list[np.ndarray], epochs: int, learning_rate: float = 1.0) -> None:
        for epoch in range(epochs):
            predicted_outputs = self.forward(inputs)
            self.backward(actual_outputs, predicted_outputs, learning_rate)