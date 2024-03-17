import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

train_dir = 'Dataset/minitrain' #directorio de entrenamiento
test_dir = 'Dataset/minitest' #directorio de prueba

### Para saber cuantas imágenes hay
cat_files_path = os.path.join(train_dir, 'cat/*')
dog_files_path = os.path.join(train_dir, 'dog/*')

cat_files = sorted(glob(cat_files_path))
dog_files = sorted(glob(dog_files_path))

n_files = len(cat_files) + len(dog_files)
print(n_files)

cat_files_path_test = os.path.join(test_dir, 'cat/*')
dog_files_path_test = os.path.join(test_dir, 'dog/*')

cat_files_test = sorted(glob(cat_files_path_test))
dog_files_test = sorted(glob(dog_files_path_test))

n_files_test = len(cat_files_test) + len(dog_files_test)
print(n_files_test)

learning_rate = 0.00025 
#learning_rate = 0.00001
#learning_rate = 0.01
epochs = 20
batch_size = 32
loss = "binary_crossentropy"
#todos mis learning rates estaban siendo elegidos aleatoriamente

import wandb
from wandb.keras import WandbCallback
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
wandb.login()


wandb.init(project="perros y gatos2")
wandb.config.learning_rate = learning_rate
wandb.config.epochs = epochs
wandb.config.batch_size = batch_size
wandb.config.loss = loss
#wandb.config.optimizer = optimizer


ih, iw = 150, 150 #tamano de la imagen
input_shape = (ih, iw, 3) #forma de la imagen: alto ancho y numero de canales


num_class = 2 #cuantas clases --> perro o gato
epochs = 20
 #cuantas veces entrenar. En cada epoch hace una mejora en los parametros
#intentar con 50

batch_size = 32 #batch para hacer cada entrenamiento. Lee 50 'batch_size' imagenes antes de actualizar los parametros. Las carga a memoria
#y si pongo 32
num_train = n_files #numero de imagenes en train
num_test =  n_files_test #numero de imagenes en test


epoch_steps = num_train // batch_size
test_steps = num_test // batch_size

print(train_dir)
gentrain = ImageDataGenerator(rescale=1. / 255.) #indica que reescale cada canal con valor entre 0 y 1.


train = gentrain.flow_from_directory(train_dir,
                batch_size=batch_size,
                target_size=(iw, ih),
                class_mode='binary')


print(test_dir)
gentest = ImageDataGenerator(rescale=1. / 255)

test = gentest.flow_from_directory(test_dir,
                batch_size=batch_size,
                target_size=(iw, ih),
                class_mode='binary')


#########  Modelo
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(ih, iw,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
#model.add(Activation('softmax')) no me funcionó
model.add(Activation('sigmoid')) primero se intentó con este

model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=learning_rate),
              metrics=['accuracy'])


model.fit(train,
                steps_per_epoch=epoch_steps,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=test,
                validation_steps=test_steps,
                callbacks=[WandbMetricsLogger(log_freq=5),
                      WandbModelCheckpoint("models")]
                )

##esto solo es un experimento de un código para que aparezca en las gráficas pero creo que le falta editarlo más
random.seed(1)
wandb.init()
# definimos la metrica del loss, del cual buscamos el mínimo
wandb.define_metric("loss", summary="min")
# definimos la metrica del accuracy, del cual buscamos el máximo
wandb.define_metric("acc", summary="max")
for i in range(10):
    log_dict = {
        "loss": random.uniform(0, 1 / (i + 1)),
        "acc": random.uniform(1 / (i + 1), 1),
    }
    wandb.log(log_dict)

model.save('prueba.h5')

#Nota: Esto solo lo subo por si no se ve el que hice en  Google Colab
#Realmente es el mismo código.
#solo que tengo problemas en poner los permisos públicos en wandb