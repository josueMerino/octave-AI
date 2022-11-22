# Importar librerías a utilizar
import numpy as np
import matplotlib.pyplot as plt

# Cargar la base de datos MNIST y asignar las imágenes y etiquetas de los conjuntos para
# entrenamiento y prueba
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Las imágenes están codificadas en arrays (0 o 1), y las etiquetas son un array números (0 a 9)
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

# Consultar algunas imágenes y etiquetas para entrenamiento
# Visualizar algunos ejemplos
fig = plt.figure()
for i in range(9):
 plt.subplot(3,3,i+1)
 plt.tight_layout()
 plt.imshow(X_train[i], cmap='gray', interpolation='none')
 plt.title("Dígito: {}".format(y_train[i]))
 plt.xticks([])
 plt.yticks([])
fig
plt.show() 


# Visualizar un ejemplo de las 60.000 imágenes
digit = X_train[4345]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# Antes de realizar el entrenamiento, preparar los datos transformando las imágenes
# iniciales con valores entre 0 y 255 (negro a blanco), a valores binarizados (0 a 1)
X_train = X_train.reshape((60000, 28 * 28))
X_train = X_train.astype('float32') / 255
X_test = X_test.reshape((10000, 28 * 28))
X_test = X_test.astype('float32') / 255