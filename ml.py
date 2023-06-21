import numpy as np

class NeuralNetwork:
    def activate(self, x):
        return 1 / (1 + np.exp(-x))
        
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
        self.num_layers = len(dimensions)
        self.weights = [np.random.randn(dimensions[i], dimensions[i-1]) for i in range(1, self.num_layers)]
        self.biases = [np.random.randn(dimensions[i], 1) for i in range(1, self.num_layers)]

    def forward(self, input: list[float]) -> list[float]:
        """
        Perform a forward pass through the neural network.
        Calculate the outputs based on the input and current weights and biases.

        Args:
        - input (list[float]): List of input values to the network.

        Returns:
        - output (list[float]): List of output values from the network.
        """

        a = np.array(input).reshape((self.dimensions[0], 1))
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.activate(z)
        output = a.flatten().tolist()
        return output

    def backward(self, input: list[float], predicted: list[float], output: list[float], learning_rate: float) -> None:
        """
        Perform a backward pass through the neural network.
        Update the weights and biases based on the calculated gradients.

        Args:
        - input (list[float]): List of input values to the network.
        - predicted (list[float]): List of predicted values from the network.
        - output (list[float]): List of output values for the corresponding input.
        """

        error = np.array(predicted).reshape((self.dimensions[-1], 1)) - np.array(output).reshape((self.dimensions[-1], 1))
        delta = error * np.array(predicted).reshape((self.dimensions[-1], 1)) * (1 - np.array(predicted).reshape((self.dimensions[-1], 1)))

        deltas = [delta]
        for i in range(self.num_layers - 2, 0, -1):
            delta = np.dot(self.weights[i].T, delta) * predicted[i-1] * (1 - predicted[i-1])
            deltas.insert(0, delta)

        a = np.array(input).reshape((self.dimensions[0], 1))
        for i in range(self.num_layers - 1):
            dw = np.dot(deltas[i], a.T)
            db = deltas[i]

            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

            a = self.activate(np.dot(self.weights[i], a) + self.biases[i])

    def train(self, inputs: list[list[float]], outputs: list[list[float]], epochs: int, learning_rate: float = 1.0) -> None:
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

    def predict(self, inputs: list[list[float]]) -> list[list[float]]:
        """
        Make predictions using the trained neural network.
        Perform a forward pass to calculate the outputs for the given inputs.

        Args:
        - inputs (list[list[float]]): List of input value lists for prediction.

        Returns:
        - predictions (list[list[float]]): List of predicted value lists for the corresponding inputs.
        """

        predictions = []
        for x in inputs:
            output = self.forward(x)
            predictions.append(output)
        return predictions