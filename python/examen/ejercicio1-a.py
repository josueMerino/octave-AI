# importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error


def regresion(x, y, x_test, y_test, color='red', grado=1):
    # obtener los coeficientes del polinomio
    coeffs = np.polyfit(x, y, deg=grado)

    # obtener el polinomio
    p = np.poly1d(coeffs, variable='X')
    print("Polinomio de grado ", grado, " : ")
    print(p)
    print("")

    # obtener los valores predichos para los valores de entrenamiento
    y_pred = np.polyval(p, x)

    # obtener el error cuadrático medio para los valores de entrenamiento
    print('Error cuadrático medio para los valores de entrenamiento: %.2f\n' %
          mean_squared_error(y, y_pred))

    # obtener el coeficiente de correlación de pearson para los valores de entrenamiento
    print('Correlación de Pearson para los valores de entrenamiento: %.2f\n' %
          np.corrcoef(y, y_pred)[0, 1])

    # obtener los valores predichos para los valores de test
    y_pred_test = np.polyval(p, x_test)

    # obtener el error cuadrático medio para los valores de test
    print('Error cuadrático medio para los valores de test: %.2f\n' %
          mean_squared_error(y_test, y_pred_test))

    # obtener el coeficiente de correlación de pearson para los valores de test
    print('Correlación de Pearson para los valores de entrenamiento: %.2f\n' %
          np.corrcoef(y_test, y_pred_test)[0, 1])

    # graficar el polinomio
    plt.plot(x, y_pred, color=color, linewidth=2, label="grado% d" % grado)


csvData = pd.read_csv("data12.csv")

x = csvData.iloc[:, 0].values
y = csvData.iloc[:, 1].values

n_data = len(x)

n_train = int(n_data * 0.8)

x_train = np.array(x[:n_train])
y_train = np.array(y[:n_train])

x_test = np.array(x[n_train:])
y_test = np.array(y[n_train:])

colors = ['red', 'orange', 'green', 'purple']

# REGRESION POLINOMICA (utilizando la función polyfit de la librería numpy)
print("Ajuste de Regresión polinómica")
# Ajuste para ecuaciones de grado 3, 4 y 5
# Polinomio de grado 3: y = t0+t1*X+t2*X^2+t3*X^3
# Polinomio de grado 4: y = t0+t1*X+t2*X^2+t3*X^3+t4*X^4
# Polinomio de grado 5: y = t0+t1*X+t2*X^2+t3*X^3+t4*X^4+t5*X^5
# Dibujar la gráfica teórica f(x)
plt.title("Regresión polinómica")

for count, degree in enumerate([2, 8, 17]):
    regresion(x_train, y_train, x_test, y_test,
              color=colors[count], grado=degree)

plt.scatter(x_train, y_train, color='black')

plt.scatter(x_test, y_test, color='red')

# Leyenda del gráfico
plt.legend(loc='lower left')
# Dibujar el gráfico
plt.show()

# Predecir para un valor de X=6 con el modelo de regresión polinómica de grado 17
coeffs = np.polyfit(x_test, y_test, deg=17)
y_pred = np.polyval(np.poly1d(coeffs), 6)
print("Predicción para X=6: y=", y_pred)

# Mientras más aumenta el grado del polinomio el error disminuye pero esto puede dar lugar al overfitting
