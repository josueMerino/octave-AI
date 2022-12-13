# importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

csvData = pd.read_csv("data12.csv")

print(csvData['col1'])

openingData = pd.DataFrame(csvData, columns=['col1'])

x_plot = openingData.index[20:]
y_plot = openingData['col1'][20:]

x = openingData.index
x_train = np.sort(x[20:])
y_train = openingData['col1'][20:]
x_test = np.sort(x[:20])
y_test = openingData['col1'][:20]

colors = ['red','orange', 'green', 'black']


# Definir la anchura de la linea a dibujar
lw = 2 

# # Dibujar la gráfica teórica f(x)
plt.plot(x_plot, y_plot, color='blue', linewidth=lw,label="Función teórica")
# # y los puntos utilizados para entrenamiento
plt.scatter(x_train, y_train, color='navy', s=30, marker='o', label="Puntos de entrenamiento")
plt.title("Función teórica y puntos de entrenamiento")
plt.show()

# REGRESION POLINOMICA (utilizando la función polyfit de la librería numpy)
print("Ajuste de Regresión polinómica")
# Ajuste para ecuaciones de grado 3, 4 y 5
# Polinomio de grado 3: y = t0+t1*X+t2*X^2+t3*X^3
# Polinomio de grado 4: y = t0+t1*X+t2*X^2+t3*X^3+t4*X^4
# Polinomio de grado 5: y = t0+t1*X+t2*X^2+t3*X^3+t4*X^4+t5*X^5
# Dibujar la gráfica teórica f(x)
plt.title("Regresión polinómica")
plt.plot(x_plot, y_plot, color='blue', linewidth=lw,label="Función teórica")
# y los puntos utilizados para entrenamiento
plt.scatter(x_train, y_train, color='navy', s=30, marker='o', label="Puntos de entrenamiento")

for count, degree in enumerate([2, 8, 17]):
 # Ajuste del polinomio de grado 'degree' a los datos de entrenamiento x,y
 coeffs = np.polyfit(x_train,y_train,deg=degree)
 # Determinar y escribir la forma del polinomio
 p = np.poly1d(np.polyfit(x_train, y_train, deg=degree), variable='X')
 print("Polinomio de grado ",degree," : ")
 print(p)
 print("")

 y_pred = np.polyval(np.poly1d(coeffs), x_train)
 print("Error cuadrático medio (ECM): ",1/20*(sum((y_train-y_pred)**2)))
 print("")
 print(y_pred)
 # Dibujar la gráfica del polinomio
 # Calcular la y de la gráfica 'y_plot'
 y_plot = np.polyval(np.poly1d(coeffs), x_plot)
 # Dibujar la gráfica
 plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw,label="grado %d" % degree) 
 
# Leyenda del gráfico
plt.legend(loc='lower left')
# Dibujar el gráfico
plt.show()

# Predecir para un valor de X=6 con el modelo de regresión polinómica de grado 5
coeffs = np.polyfit(x_test,y_test,deg=17)
y_pred = np.polyval(np.poly1d(coeffs), 6)
print("Predicción para X=6: y=",y_pred) 

# Mientras más aumenta el grado del polinomio el error disminuye pero esto puede dar lugar al overfitting