#Se trata de una red neuronal que reproduzca dos funciones en el intervalo [-1, 1]
#también se grafica la red junto la gráfica de la función

#Inciso a) 3sin(xpi)
#Inciso b) 1+2x+4x^3

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, Reshape, Layer
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import random