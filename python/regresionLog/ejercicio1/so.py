# Librerías
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sb

# Cargar la información del fichero CSV
dataframe = pd.read_csv(r"usuarios_win_mac_lin.csv")

# Visualizar las 5 primeras filas del fichero
dataframe.head()
# Clases de usuarios: 0 -> Windows, 1-> Macintosh, 2-> Linux
# Consultar información de la base de datos
dataframe.describe()
# Analizar cuantos ejemplos existen de cada clase
print(dataframe.groupby('clase').size())