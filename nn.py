import random
import math
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    # [3, 3, 2, 2]
    def __init__(self, dimensions: list[int]):
        self.dimensions = dimensions
        self.weights = []
        self.biases = []
        # generate random weights/biases
        for index in range(1, len(dimensions)):
            dimension = dimensions[index]
            self.weights.append([1 for _ in range(dimension * dimensions[index - 1])])
            self.biases.append([1 for _ in range(dimension)])
        print(self.weights)
        print()
        print(self.biases)
        print()
        print(self.dimensions)
        print()
        print([1, 0, 1])
        print()

    # [1, 0, 1]
    def forward_propagation(self, input: list[int]) -> list[int]:
        # 1*1 + 0*1 + 1*1 = 2
        # 1*1 + 0*1 + 1*1 = 2
        # 1*1 + 0*1 + 1*1 = 2
        # 
        # 2*1 + 2*1 + 2*1 = 6
        # 2*1 + 2*1 + 2*1 = 6
        # 
        # 6*1 + 6*1 = 12
        # 6*1 + 6*1 = 12
        weights = self.weights
        for index in range(len(weights)):
            for index, node in input:
                print(node)
            print(index)
                
        return None

    def back_propagation(self):
        return None
    
    def activate(self, x):
        return 1 / (1 + math.pow(math.e, -x))
    
    def visualize(self) -> None:
        for index, dimension in enumerate(self.dimensions):
            for neuron in range(dimension):
                plt.scatter(index, neuron - dimension/2)
        plt.show()