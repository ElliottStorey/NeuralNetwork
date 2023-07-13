import numpy as np

class NeuralNetwork:
    def __init__(self, dimensions: tuple[int]) -> None:
        self.dimensions = dimensions
        self.weights = []
        self.biases = []

        for i in range(1, len(dimensions)):
            # Zero initialization
            weight = np.random.random((dimensions[i-1], dimensions[i]))
            self.weights.append(weight)

            # Zero initialization
            bias = np.zeros((1, dimensions[i]))
            self.biases.append(bias)

    def forward(self, input: np.ndarray) -> np.ndarray:
        for i in range(len(self.dimensions) - 1):
            output = input @ self.weights[i] + self.biases[i]
            input = output
        return output
    
    def backward(self, input: np.ndarray, predicted_output: np.ndarray, actual_output: np.ndarray, learning_rate: float) -> None:
        b_delta = 2 * (predicted_output - actual_output)

        for i in reversed(range(len(self.dimensions) - 1)):
            delta = self.weights[i] @ delta
            if i == 0:
                delta = input @ delta

            self.weights[i] -= learning_rate * delta
            self.biases[i] -= learning_rate * b_delta

    def train(self, inputs: np.ndarray, actual_outputs: np.ndarray, epochs: int, learning_rate: float = 1.0) -> None:
        for i in range(epochs):
            error = []

            for input, actual_output in zip(inputs, actual_outputs):
                predicted_output = self.forward(input)
                error.append((actual_output - predicted_output) ** 2)
                self.backward(input, predicted_output, actual_output, learning_rate)

            error = np.mean(error)
            print(f'Epoch {i+1}/{epochs}\nError {error}\n')                