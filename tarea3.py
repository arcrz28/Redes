"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#En este archivo se encuentra la tarea 3 (optimixación y cross-entropy) 
#al igual que la tarea 2 sobre comentar el código network.py
#Se agregaron comentarios nuevos para comentar el cross-entropy y el optimizador.
#
#### Libraries
# Standard library  
import random  #importa datos aleatorios

# Third-party libraries
import numpy as np #Es la librería que importa ayuda para trabajar con
#largos arreglos multi-dimensionales y matrices      

class Network(object): #Define de que tipo de clase será la neurona

    def __init__(self, sizes, loss_function="mean_square_avg"): #Define los parámetros del número de neuronas por capas y pesos
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes) #Este es el tamaño de las capas
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #Esta función crea los biases de una manera aleatoria
        self.weights = [np.random.randn(y, x) #Esta función crea los pesos de una manera aleatoria
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.loss_function = loss_function
        #numpy.random.randn(y,x), numpy,random.randn(y,1) crea arreglos de un tamaño específico
        # y lo "llena" con valores aleatorios

    def feedforward(self, a): #Inicia el ciclo de la red neuronal, es un valor de inicialización por la función sigmoide
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b) #Hace un producto punto y le sumamos los bias
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        #El SGD es el statistic Gradient Descent y hace una aproximación al descenso de cada iteración
        #En este caso, obtiene  los w para después hacer el backpropagation, además de que nos evita que se "atore"

        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data:         #son los datos que utiliza cada red para hacer pruebas o entrenar
            test_data = list(test_data) # crea una lista con los valores de prueba
            n_test = len(test_data) #Este da el tamaño de los datos a entrenar

        training_data = list(training_data)  #se crea una lista con valores que serán entrenados
        n = len(training_data)  #Calcula el número total de muestras de entrenamiento
        for j in range(epochs):
            random.shuffle(training_data) # Barajea o reorganiza los datos que serán entrenados aleatoriamente 
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:  #Agrupa los datos en subconjuntos o "conjuntos más pequeños"
                self.update_mini_batch(mini_batch, eta) #Actualiza el mini batc
            if test_data:
                print("Epoch {0}: {1} / {2}".format( #Este es un ciclo que nos permitirá ver las epocas
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j)) #Devuelve el valor de las épocas.
                #.format(j) sustituye a {0} con el valor de la variable j 
                #Es una secuencia que contiene a {0} que será sustituido

#"Aquí vamos a implementar el optimizador Adam "
    def adam(self, training_data, epochs, mini_batch_size, eta, test_data=None, beta_1=0.9, beta_2=0.999, epsilon=1e-07):            
        self.mini_batch_size = mini_batch_size

        def update_mini_batch(mini_batch, eta, beta_1, beta_2, epsilon, t, mini_batch_size):
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            # Llamo a las variables m y v que están fuera de esta función y las actualizo
            nonlocal m_b
            nonlocal m_w
            nonlocal v_b
            nonlocal v_w
        

    def update_mini_batch(self, mini_batch, eta, mini_batch_size):  ##Agregamos primero lmbda, n
        #Actualiza los valores de los pesos y los biases calculando
        #el gradiente para el mini batch 
        #lmbda es un parámetro que regulariza
        #n es el número total de datos para entrenar
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] #lo que hace es llenar de ceros
        #Se llena de la suma de nuestro gradiente por el mini batch
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)  #Función de costo
            #Utiliza el método de backpropagation para 
            #calcular las derivadas de Cx con respecto de w y de b
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            #El nabla_b y nabla_w invocan el algoritmo de backpropagation 
            #el cual es una forma de calcular el gradiente de la función de costo
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)] #A los pesos se les resta
        #el valor de aprendizaje
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        #crea listas para guardar los gradientes de los biases y pesos respectivamente
        #Luego los calcula de la forma de backpropagation y los actualiza.
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] #Es la lista de las activaciones capa por capa
        zs = [] # Es la lista que almancenará los vectores de z capa a capa
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z) #Se aplica la función sigmoide al producto punto que se 
                                    #definió en el renglón anterior
            activations.append(activation)
        # backward pass
            #El delta (línea 127) calcula el error de atrás para adelante
            # El símbolo \ hace que cambie de renglón
        delta = (self.cost).delta(zs[-1], activations[-1], y) #quitamos la derivada
        #self.cost_derivative(activations[-1], y) * \
        #  sigmoid_prime(zs[-1]) #esta y el renglón de arriva es el código anterior la cual
        #calcula la derivada de la función sigmoide.
        nabla_b[-1] = delta    #Da como resultado delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        #Calcula los gradientes de la función de costo con respecto a 
        #los pesos (w) de la capa de salida
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            #Busca los pesos (w) de las entradas en z para calcular la capa
            #actual y zs "mantiene" los pesos en cada una de las capas
            sp = sigmoid_prime(z)
            #Calcula la derivada de la función sigmoide para 
            #los pesos (w) en z
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            #calcula un "nuevi" delta, es decir, calcula un delta en 
            #la capa actual para la siguiente capa.
            nabla_b[-l] = delta   #Gradiente con respecto de b
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)  #Regresa los valores de los parámentros

    def evaluate(self, test_data): #El self es el la instancia de clase y el test_data
        #es un parámetro que tendrá los datos que serán evaluados.
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
        #Clasifica la red.
        #Nos dice cuál es el número que la red predice
                        for (x, y) in test_data]
        #Se utiliza para tomar en cuenta los máximos de los datos
        return sum(int(x == y) for (x, y) in test_results)
        #En esta función se predice cuántos datos coincidieron, luego se suman y los regresa 
    def cost_derivative(self, output_activations, y):
        #El comando cost_derivative toma los parámetros  donde self es la instancia de la clase
        #Y output_activations son arreglos de las activaciones de salida; y la salida
        #esperada 
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
    #Decuelve la derivada elemento a elemento de Cx para la salida
    #de las activaciones

#### Miscellaneous functions
def sigmoid(z): #Función sigmoide
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):   #Derivada de la función sigmoide
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
#Devuelve el valor de la derivada
#
#Ahora utilicemos un optimizador, en nuestro caso utilizaremos SGD momentum