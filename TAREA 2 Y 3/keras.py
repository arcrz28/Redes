#El código está completo, el de los experimentos con y sin regularizadores, porque tuve problemas con Jupyter
#Además de que acá están los commits.
#Se subieron dos archivos a GitHub: Experimentos y Regularizadores, con cada código por separado

#Primero vamos a importar las librerías.
import tensorflow as tf
from tensorflow import keras as keras 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras. optimizers import Adam, RMSprop, SGD
from keras.regularizers import L1, L2, L1L2
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import History

dataset= mnist.load_data() 
(x_train, y_train), (x_test,y_test) = dataset #Son las variables separadas de los datos de entrenamiento y pruebas

#Ahora aplanamos las imágenes y convertimos en punto flotante
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)
x_trainv = x_train.astype('float32')
x_testv = x_test.astype('float32')

lr = 0.001  #learning rate
num_clases = 10
y_trainc = keras.utils.to_categorical(y_train, num_clases)
y_testc = keras.utils.to_categorical(y_test, num_clases)

#Ahora vayamos con la parte a) en ta tarea pasada traté de usar Adam y categorical cross entropy
#Pero no me salió :(. 
#Llamemoslo experimento 1 (de 4)
exp1 = Sequential([
    Dense(30, activation='sigmoid', input_shape=(784,)),
    Dense(10,activation='softmax') 
      ])
exp1.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
history = exp1.fit(x_trainv, y_trainc, batch_size = 10, epochs = 20, validation_data=(x_testv, y_testc))

# Gráfica de pérdida-epoca del experimento 1
plt.xlabel("# Epoca")
plt.ylabel("Perdida")
plt.plot(history.history["loss"])

import matplotlib.pyplot as plt
image = x_train[5].reshape((28, 28))
plt.figure()
plt.imshow(image, cmap="gray")#número de imagen en el mnist
plt.colorbar()
plt.grid(False)
plt.show()

score = exp1.evaluate(x_testv, y_testc, verbose=1) #evaluar la eficiencia del modelo
print(score)
a=exp1.predict(x_testv) #predicción de la red entrenada
print(a.shape)
print(a[1])
print("resultado correcto:")
print(y_testc[1])     #si sale :)



"Pasemos al inciso b.1) que es hacer el segundo experinento cambiando neuronas" 

exp2 = Sequential([
    Dense(512, activation='sigmoid', input_shape=(784,)),
    Dense(10,activation='softmax') 
      ])
exp2.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
history = exp2.fit(x_trainv, y_trainc, batch_size = 10, epochs = 10, verbose=1, validation_data=(x_testv, y_testc))

# Gráfica de pérdida-epoca del experimento 2
plt.xlabel("# Epoca")
plt.ylabel("Perdida")
plt.plot(history.history["loss"])

import matplotlib.pyplot as plt
image = x_train[5].reshape((28, 28))
plt.figure()
plt.imshow(image, cmap="gray")#número de imagen en el mnist
plt.colorbar()
plt.grid(False)
plt.show()

score = exp2.evaluate(x_testv, y_testc, verbose=1) #evaluar la eficiencia del modelo
print(score)
a=exp2.predict(x_testv) #predicción de la red entrenada
print(a.shape)
print(a[1])
print("resultado correcto:")
print(y_testc[1])     #si sale :)


"experimento 3"
#En el inciso b.2) que es el tercer experimento, cambiemos el número de neuronas y épocas

exp3 = Sequential([
    Dense(256, activation='sigmoid', input_shape=(784,)),
    Dense(50, activation='sigmoid'),
    Dense(10, activation='softmax'),
    Dense(10,activation='relu') 
      ])
exp3.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
history = exp3.fit(x_trainv, y_trainc, batch_size = 10, epochs = 10, verbose=1, validation_data=(x_testv, y_testc))





#Inciso b.3) cuarto experimento: Cambiaremos una función de activación, número de 
#epocas y neuronas.

exp4 = Sequential([
    Dense(30, activation='relu', input_shape=(784,)),
    Dense(30, activation='relu'),
    Dense(10, activation='relu'),
    Dense(10,activation='softmax') 
      ])
exp4.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
history = exp4.fit(x_trainv, y_trainc, batch_size = 10, epochs = 30, verbose=1, validation_data=(x_testv, y_testc))






#Ahora comencemos con el inciso c) de la tarea agregando regularizaciones
#inciso c.1) regularización L1
expL1 = Sequential([
    Dense(512, activation='sigmoid', input_shape=(784,), activity_regularizer=L1(l1=0.01)),
    Dense(10,activation='softmax') 
      ])
expL1.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
history = expL1.fit(x_trainv, y_trainc, batch_size = 10, epochs = 20, verbose=1, validation_data=(x_testv, y_testc))

#inciso c.2) regularización L2
expL2 = Sequential([
    Dense(512, activation='sigmoid', input_shape=(784,), activity_regularizer=L2(l2=0.01)),
    Dense(10,activation='softmax') 
      ])
expL2.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
history = expL2.fit(x_trainv, y_trainc, batch_size = 10, epochs = 20, verbose=1, validation_data=(x_testv, y_testc))

#inciso c.3) regularización L1L2
expL1L2 = Sequential([
    Dense(512, activation='sigmoid', input_shape=(784,), activity_regularizer=L1L2(l1=0.01, l2=0.01)),
    Dense(10,activation='softmax') 
      ])
expL1L2.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
history = expL1L2.fit(x_trainv, y_trainc, batch_size = 10, epochs = 20, verbose=1, validation_data=(x_testv, y_testc))

#inciso c.4) regularización Dropout
exp_drop = Sequential([
    Dense(512, activation='sigmoid', input_shape=(784,), 
    Dropout(0.2)),
    Dense(10,activation='softmax') 
      ])
exp_drop.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
history = exp_drop.fit(x_trainv, y_trainc, batch_size = 10, epochs = 20, verbose=1, validation_data=(x_testv, y_testc))

#Inciso c.5) regularización L1L2 y Dropout
exp_L1L2_drop = Sequential([
    Dense(512, activation='sigmoid', input_shape=(784,), activity_regularizer=L1L2(l1=0.01, l2=0.01)),
    Dropout(0.2),
    Dense(10,activation='softmax') 
      ])
exp_L1L2_drop.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
history = exp_L1L2_drop.fit(x_trainv, y_trainc, batch_size = 10, epochs = 20, verbose=1, validation_data=(x_testv, y_testc))