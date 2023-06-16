import random
import math
import numpy as np
import matplotlib.pyplot as plt

def mean_squared_error(predicted: list[int], expected: list[int]):
    summation = 0
    for index in range(len(expected)):
        summation += (expected[index] - predicted[index])**2
    return summation/len(expected)
    
def mean_squared_error_gradients(predicted: list[int], expected: list[int]):
    gradients = [0] * len(expected)
    for index in range(len(expected)):
        gradients[index] = 2 * (predicted[index] - expected[index]) / len(expected)
    return gradients

def sigmoid(x):
    return 1 / (1 + math.pow(math.e, -x))
    
def sigmoid_derivative(self, x):
    return self.sigmoid(x) * (1 - self.sigmoid(x))

class NeuralNetwork:
    def __init__(self, dimensions: list[int]):
        self.dimensions = dimensions
        # populate random weights and biases
        self.weights = [[random.random() for _ in range(dimensions[index] * dimensions[index - 1])] for index in range(1, len(dimensions))]
        self.biases = [[random.random() for _ in range(dimensions[index])] for index in range(1, len(dimensions))]

    def train(self, inputs: list[list[int]], outputs: list[list[int]], epochs: int, learning_rate: float = 1.0):
        for epoch in range(epochs):
            output = self.forward(input)
        # todo
        return None

    def forward(self, input: list[int]) -> list[int]:
        for index in range(len(self.dimensions) - 1):
            weights = self.weights[index]
            weights_index = 0
            dimension = self.dimensions[index]
            output = []
            for _ in range(self.dimensions[index + 1]):
                neuron = 0
                for index in range(dimension):
                    neuron += input[index] * weights[weights_index]
                    weights_index += 1
                output.append(neuron)
            input = output
        return output

    def backward(self):
        # todo
        return None
    
    def visualize(self) -> None:
        for index, dimension in enumerate(self.dimensions):
            for neuron in range(dimension):
                plt.scatter(index, neuron - dimension/2)
        plt.show()