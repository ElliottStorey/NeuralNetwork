import ml
import data

training_data = data.generate(3)
testing_data = data.generate(3)

nn = ml.NeuralNetwork((4, 4, 1))
forward = nn.forward(training_data['inputs'])
backward = nn.backward(training_data['outputs'], forward, 1.0)
#net.train(training_data['inputs'], training_data['outputs'], 10)
#print(net.predict(testing_data['inputs']))

#print(testing_data['outputs'])