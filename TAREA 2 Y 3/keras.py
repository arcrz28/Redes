#Primero vamos a importar las librerías.
import tensorflow as tf
from tensorflow import keras as keras 
from keras.datasets import mnist
from keras.models import Sequentials
from keras.layers import Dense, Dropout
from keras. optimizers import Adam, RMSprop, SGD
from keras.regularizers import L1, L2, L1L2
import numpy as np

dataset= mnist.load_data() 
(x_train, y_train), (x_test,y_test) = dataset #Son las variables separadas de los datos de entrenamiento y pruebas

#Ahora aplanamos las imágenes y convertimos en punto flotante
x_train = x_train.reshape(60000,784)
x_testv = x_test.reshape(6000,784)
x_train = x_train.astype('float32')
x_testv = x_test.astype('float32')

num_clases = 10
y_trainc = keras.utils.tocategorical(y_train, num_clases)
y_testc = keras.utils.tocategorical(y_test, num_clases)
