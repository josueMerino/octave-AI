# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

# Definir centroides iniciales
centroides = [[1, 1], [-1, -1], [1, -1]]

# Generar un conjunto de datos X utilizando 'make_blobs'
X, labels_true = make_blobs(n_samples=750, centers=centroides, cluster_std=0.4,
random_state=0)
# Visualizar los datos generados de forma aleatoria
plt.scatter(X[:,0],X[:,1],s=5)
plt.show()

# Ejecutar DBSCAN (eps => radio, min_samples => mínimo número de puntos para considerar un
# cluster). Ajustar a los datos generados X
radio=0.1
print("Algoritmo DBScan con radio (eps) = ",radio)
clustering = DBSCAN(eps=radio, min_samples=10).fit(X)