# create a decision tree
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Leer los datos
data = pd.read_csv('data12.csv', names=['x', 'y', 'r', 'g', 'b', 'valid'])

# obtener los 5 primero campos y el ultimo como x
x = data.iloc[:, 0:5].values

# obtener el ultimo campo como y
y = data.iloc[:, 5].values

# dividir los datos en entrenamiento y test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# crear el arbol de decision
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)

# entrenar el arbol de decision
classifier.fit(x_train, y_train)

# predecir los valores de test
y_pred = classifier.predict(x_test)

# obtener la precision del modelo
print("Presición del modelo: ")
print(accuracy_score(y_test, y_pred))

# hacer una prediccion
pred = classifier.predict([[435, 675, 243, 181, 214]])
if pred[0] == 1:
    print("Valido")
else:
    print("No valido")

# mostrar el arbol de decision
dot_data = tree.export_graphviz(classifier, out_file=None, filled=True, rounded=True, special_characters=True, feature_names=['x', 'y', 'r', 'g', 'b'], class_names=['no valid', 'valid'])
graph = graphviz.Source(dot_data)
graph.render("examen-2")

graph.view()

"""
    Conclusiones:
    - El arbol de decision tiene una presición casi del 100%.
    - El modelo se ajusta a los datos
    - El modelo es representativo de los datos.
"""