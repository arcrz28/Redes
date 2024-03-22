#Se entrenará una red neuronal que de la solución de dos ecuaciones diferenciales en [-5, 5]
#Se graficará la solución numérica junto con la analítica

#Inciso a) xy' + y = x^2cos(x). con y(0)=0
#Inciso b) d^2y/dx^2 = -y con y(0)=1, y'(0)= -0.5


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, Reshape, Layer
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

