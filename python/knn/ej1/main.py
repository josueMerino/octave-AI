# Importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Lectura de la base de datos "Base_diabetes.csv"
df = pd.read_csv("ej1/Base_diabetes.csv")
# Visualizar las primera 5 filas de la tabla de datos (DataFrame)
print(df.head())

# Ver el tamaño de la tabla de datos
print(df.shape)

# La base de datos tiene 768 registros, con 8 atributos y 1 etiqueta
# Atributos: Embarazos, Glucosa, Presión_sangre, Espesor_piel, Insulina, BMI, Histórico,Edad
# Etiqueta: Categoría (1-> Tiene Diabetes 0-> No tiene diabetes)

# Crear los arrays para X e y
X = df.drop('Categoria',axis=1).values
y = df['Categoria'].values

# Dividir el conjunto de datos en entrenamiento y test
# Utilizar el método 'train_test_split' de la librería sklearn
from sklearn.model_selection import train_test_split
# Considerar como test un 40 % (test_size=0.4) del total del conjunto de datos
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)

# Crear el clasificador basado en KNN (k-Nearest Neighbors)
from sklearn.neighbors import KNeighborsClassifier
# Crear los arrays donde registramos las precisiones para entrenamiento y test
neighbors = np.arange(1,9)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors)) 

# Realizar el cálculo con diferentes valores de 'k'
# para identificar el valor de k que da mejores resultados
for i,k in enumerate(neighbors):
 # Configurar el clasificador KNN con 'k' vecinos (neighbors)
 knn = KNeighborsClassifier(n_neighbors=k)

 # Ajustar el modelo a los datos de entrenamiento
 knn.fit(X_train, y_train)

 # Registrar las precisiones para los datos de entrenamiento
 train_accuracy[i] = knn.score(X_train, y_train)

 # Registrar las precisiones para los datos de test
 test_accuracy[i] = knn.score(X_test, y_test) 
 
# Visualizar el gráfico de la precisión de entrenamiento y test
plt.title('k-NN para ''k'' vecinos')
plt.plot(neighbors, test_accuracy, label='Precisión de Test')
plt.plot(neighbors, train_accuracy, label='Precisión de Entrenamiento')
plt.legend()
plt.xlabel('Número de vecinos')
plt.ylabel('Precisión')
plt.grid()
plt.show() 