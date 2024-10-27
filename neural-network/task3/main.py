import os
import mnist_loader
import network2

# Change the current working directory
os.chdir('/home/evgen/Documents/University/master-1/neural-network/task3')

# Load the MNIST data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Create a neural network with the specified architecture
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)

# Train the network using Stochastic Gradient Descent
net.SGD(training_data, 30, 10, 0.5, lmbda = 5.0,evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)
