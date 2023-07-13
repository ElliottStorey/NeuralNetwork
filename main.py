from ml import NeuralNetwork
import data

training_data = data.generate(500)
testing_data = data.generate(3)

nn = NeuralNetwork((4, 4, 1))

forward = nn.forward(training_data['inputs'][0])
print(forward, training_data['outputs'][0])

nn.train(training_data['inputs'], training_data['outputs'], 1000, 0.01)

forward = nn.forward(testing_data['inputs'][0])
print(forward, testing_data['outputs'][0])