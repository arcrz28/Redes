#Aquí va el inciso 1
#Se diseñará una capa en keras que transforme imágenes de color a escala de grises

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout,Activation
from tensorflow.keras.optimizers import RMSprop, Adam
from matplotlib import pyplot as plt
import numpy as np