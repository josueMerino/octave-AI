# Librerías
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Leer los datos
data = pd.read_csv('data12.csv', names=['x', 'y', 'r', 'g', 'b', 'valid'])

# obtener los 5 primero campos y el ultimo como x
x = data.iloc[:, 0:5].values

# obtener el ultimo campo como y
y = data.iloc[:, 5].values

# dividir los datos en entrenamiento y test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

# Crear el clasificador basado en Redes Neuronales
# definir numero de capas y de hidden nodes, función de activación
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10),
                    activation='relu', max_iter=500)
# Entrenamiento y ajuste de la red neuronal
mlp.fit(x_train, y_train)

# Gráfica para ver rendimiento del entranamiento
plt.plot(mlp.loss_curve_)
# mlp.loss_curve nos da una idea de como converge con los datos de
# entrenamiento
plt.xlabel('Iteracción')
plt.ylabel('Función de coste')
plt.show()

# Validar (test) la red neuronal
y_pred = mlp.predict(x_test)

# Evaluar la red neuronal
print("Matriz de confusión")
print(confusion_matrix(y_test, y_pred))
# C_ii Positivos verdaderos, C_ij!=0 Falsos Positivos
print()
print("Falsos positivos (0): ", confusion_matrix(y_test, y_pred)[0][1])
print("Falsos positivos (1): ", confusion_matrix(y_test, y_pred)[1][0])
print()
# Aplicar métrica para clasificar los datos usados para validación (test)
print("Clasificación de los resultados de la Validación(test)")
print("precision = num detecciones correctas / numero detecciones")
print("recall = num detecciones correctas / numero total de objetos en esa clase")
print()
print(classification_report(y_test, y_pred))
# obtener la precision del modelo
print("Presición del modelo: ")
print(accuracy_score(y_test, y_pred))

# Gráfica para comprobar datos de validación que son falsos positivos
plt.scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], c='r')
plt.scatter(x_test[y_test == 0, 0], x_test[y_test == 0, 1], c='b')
plt.scatter(x_test[(y_test == 1) & (y_pred == 0), 0], x_test[(y_test == 1) & (y_pred == 0), 1
                                                             ], c='orange')
plt.scatter(x_test[(y_test == 0) & (y_pred == 1), 0], x_test[(y_test == 0) & (y_pred == 1), 1
                                                             ], c='orange')
plt.show()
