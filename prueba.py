import mnist_loader
import network

training_data , test_data, _ = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])
net.adam(training_data, 90, 10, 0.01, test_data=test_data)