import mnist_loader
import tarea3

training_data , test_data, _ = mnist_loader.load_data_wrapper()

net = tarea3.Network([784, 30, 10], loss_function= "cross_entropy")
net.adam(training_data, 10, 10, 0.001, test_data=test_data)