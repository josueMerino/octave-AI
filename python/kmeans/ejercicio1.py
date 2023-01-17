# Importar librerías a utilizar
import numpy as np
import matplotlib.pyplot as plt
# Para visualizar gráficos en 3D
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets

# Cargar la base de datos IRIS
iris = datasets.load_iris()

# Asignar los atributos (X) y las etiquetas (y)
X = iris.data
y = iris.target

# Definir el conjunto de estimadores para clustering
estimators = [('k_means_iris_8', KMeans(n_clusters=8, n_init=1)),
              ('k_means_iris_3', KMeans(n_clusters=3, n_init=1)),
              ('k_means_iris_bad_init', KMeans(n_clusters=3, n_init=1, init='random'))]

# Construir los gráficos para k=8, k=3 y k=3 (con mala inicialización)
fignum = 1
titles = ['8 clusters', '3 clusters', '3 clusters, con mala inicialización']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(8, 6))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    # Entrenar el modelo KMEANS
    est.fit(X)
    # estimar/predecir las etiquetas sobre el conjunto X
    labels = est.labels_
    # Visualizar los puntos (ancho_pétalo, largo_sépalo, largo_pétalo)
    ax.scatter(X[:, 3], X[:, 0], X[:, 2],
               c=labels.astype(float), edgecolor='k')
    # Definir parámetros del gráfico. Ejes, títulos,...
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.set_xlabel('Ancho del pétalo')
    ax.set_ylabel('Largo del sépalo')
    ax.set_zlabel('Largo del pétalo')
    ax.set_title(titles[fignum - 1])

    fignum = fignum + 1

# Visualizar el gráfico correcto
fig = plt.figure(fignum, figsize=(8, 6))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    # Visualizar el texto de la especie de la planta en el punto medio del cluster
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean(),
              X[y == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))

# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k')
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
ax.set_xlabel('Ancho del pétalo')
ax.set_ylabel('Largo del sépalo')
ax.set_zlabel('Largo del pétalo')
ax.set_title('Clasificación verdadera')

fig.show()
