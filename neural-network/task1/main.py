import os
import mnist_loader
import network

# Change the current working directory
os.chdir('/home/evgen/Documents/University/master-1/neural-network/task1')

# Load the MNIST data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Create a neural network with the specified architecture
net = network.Network([784, 30, 10])

# Train the network using Stochastic Gradient Descent
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

