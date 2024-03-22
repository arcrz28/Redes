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

loss_tracker = keras.metrics.Mean(name="loss")
class Function(Sequential):
    @property
    def metrics(self):
        return [keras.metrics.Mean(name="loss")] #igual cambia el loss_tracker

    def train_step(self, data):
        def funcA(x):
            return tf.math.sin(np.pi*x)*3 ##cambiooo

        def funcB(x):
            return 1 + 2*x + 4*x**3 ##cambioooo
        batch_size =100 #Calibra la resolucion
        x = tf.random.uniform((batch_size,1), minval=-1, maxval=1)
        f = funcB(x)


        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = tf.math.reduce_mean(tf.math.square(y_pred-f))

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        #actualiza metricas
        loss_tracker.update_state(loss)

        return {"loss": loss_tracker.result()}
    #Modelo 
model = Function([
    Dense(10, activation="tanh", input_shape=(1,)),
    Dense(5, activation="tanh"),
    Dense(1)
])

model.summary()