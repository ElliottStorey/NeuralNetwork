#import tensorflow.keras as keras
import nn

'''
# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# Print the shape of the training and test data
print("Training images shape:", train_images.shape)  # (60000, 28, 28)
print("Training labels shape:", train_labels.shape)  # (60000,)
print("Test images shape:", test_images.shape)  # (10000, 28, 28)
print("Test labels shape:", test_labels.shape)  # (10000,)
'''

mynet = nn.NeuralNetwork([3, 3, 2, 2])
#mynet.visualize()
mynet.forward_propagation([1, 0, 1])