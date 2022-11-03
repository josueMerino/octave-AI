# importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

excelData = pd.read_excel("regresion/regresion5/IBEX35_Sept2018.xls")

openingData = pd.DataFrame(excelData, columns=['Apertura'])


# # Generar los puntos X usados para representar la función
x_plot = openingData.index
y_plot = openingData['Apertura']
# # Generar de forma aleatoria la muestra de puntos x y las etiquetas y
x = openingData.index
# rng = np.random.RandomState(0)
# rng.shuffle(x)
x = np.sort(x[:20])
y = openingData['Apertura']

# # crear la lista de colores
colors = ['red','orange', 'green', 'black']
# # Definir la anchura de la linea a dibujar
lw = 2 

# # Dibujar la gráfica teórica f(x)
plt.plot(x_plot, y_plot, color='blue', linewidth=lw,label="Función teórica")
# # y los puntos utilizados para entrenamiento
plt.scatter(x, y, color='navy', s=30, marker='o', label="Puntos de entrenamiento")
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
plt.scatter(x, y, color='navy', s=30, marker='o', label="Puntos de entrenamiento")

# Realizar el ajuste de los polinomios de grados 3,4 y 5
for count, degree in enumerate([7, 23, 25]):
 # Ajuste del polinomio de grado 'degree' a los datos de entrenamiento x,y
 coeffs = np.polyfit(x,y,deg=degree)
 # Determinar y escribir la forma del polinomio
 p = np.poly1d(np.polyfit(x, y, deg=degree), variable='X')
 print("Polinomio de grado ",degree," : ")
 print(p)
 print("")

 y_pred = np.polyval(np.poly1d(coeffs), x)
 print("Error cuadrático medio (ECM): ",1/20*(sum((y-y_pred)**2)))
 print("")
 # Dibujar la gráfica del polinomio
 # Calcular la y de la gráfica 'y_plot'
 y_plot = np.polyval(np.poly1d(coeffs), x_plot)
 # Dibujar la gráfica
 plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw,label="grado %d" % degree) 
 
# Leyenda del gráfico
plt.legend(loc='lower left')
# Dibujar el gráfico
plt.show()