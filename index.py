import ml
import data
import numpy as np

training_data = data.generate(100)
testing_data = data.generate(3)

net = ml.NeuralNetwork((4, 4, 1))
print(net.forward(training_data['inputs'][0]))
#net.train(training_data['inputs'], training_data['outputs'], 1000)
#print(net.predict(testing_data['inputs']))

#print(testing_data['outputs'])