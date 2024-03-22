#Se hará una capa entrenable que representará un polinomio de grado 3: a_0 + a_1x + a_2x^2 + a_3x^3
#Los parámetros son los coeficientes a_0, a_1, a_2, a_3.
#Posteriormente , se entrenará para ajustarla a la funciˀón f(x)= cos(2)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from matplotlib import pyplot as plt
import numpy as np
import math