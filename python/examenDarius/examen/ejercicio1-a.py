import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd

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
    print('Error cuadrático medio para los valores de entrenamiento: %.2f\n' % mean_squared_error(y, y_pred))
    
    # obtener el coeficiente de correlación de pearson para los valores de entrenamiento
    print('Correlación de Pearson para los valores de entrenamiento: %.2f\n' % np.corrcoef(y, y_pred)[0, 1])

    # obtener los valores predichos para los valores de test
    y_pred_test = np.polyval(p, x_test)

    # obtener el error cuadrático medio para los valores de test
    print('Error cuadrático medio para los valores de test: %.2f\n' % mean_squared_error(y_test, y_pred_test))

    # obtener el coeficiente de correlación de pearson para los valores de test
    print('Correlación de Pearson para los valores de entrenamiento: %.2f\n' % np.corrcoef(y_test, y_pred_test)[0, 1])

    # graficar el polinomio
    plt.plot(x, y_pred, color=color, linewidth=2, label="grado% d" % grado)

# Leer los datos
data = pd.read_csv('data12.csv')

# Obtener los valores de x e y
x = data.iloc[:, 0].values
y = data.iloc[:, 1].values

# Obtener los valores de x e y para test y training
n_data = len(x)

n_train = int(n_data * 0.8)

x_train = np.array(x[:n_train])
y_train = np.array(y[:n_train])

x_test = np.array(x[n_train:])
y_test = np.array(y[n_train:])

# aplicar una regresion de grado 5 con los datos
regresion(x_train, y_train, x_test, y_test, color='blue', grado=5)

plt.scatter(x_train, y_train, color='black')

plt.scatter(x_test, y_test, color='red')

plt.show()

"""
    Conclusiones:
        - El error cuadrático medio para los valores de entrenamiento es menor que el de test, lo que indica que el modelo se ajusta mejor a los datos de entrenamiento que a los de test.
        - El coeficiente de correlación de pearson para los valores de entrenamiento es mayor que el de test, lo que indica que el modelo se ajusta mejor a los datos de entrenamiento que a los de test.
        - El polinomio de grado 5 se ajusta mejor a los datos de entrenamiento que a los de test.
        - Por tanto el modelo no es representativo para todos los datos.
"""