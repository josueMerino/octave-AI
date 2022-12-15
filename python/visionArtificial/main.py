# Caso Práctico (VC_Practica_02_Caras). Ejemplo de Detección de Caras por ordenador
# Práctica de visión por computador con OpenCV y Python
# [OpenCV](http://opencv.org/releases.html) debe estar instalado
# OpenCV contiene clasificadores entrenados para detectar caras
# Los ficheros XML con los clasificadores entrenados se encuentran en el directorio `opencv/data/`.
# Para detección de caras existen dos clasificadores entrenados:
# 1. Clasificador Haar Cascade
# 2. Clasificador LBP Cascade
# Importar librerías
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# Definir una función para convertir a color RGB
def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Cargar el clasificador entrenado "Haar Cascade" desde el fichero XML
haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
print("Cargado el clasificador entrenado Haar Cascade")

# Cargar una imagen de test
test1 = cv2.imread('data/test1.jpg')

# Convertir la imagen de test a escala de grises. El detector de caras espera una imagen de este tipo
gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)

# Visualizar la imagen con OpenCV
# cv2.imshow('Imagen de Test', gray_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# Visualizar la imagen con matplotlib
plt.imshow(gray_img, cmap='gray')
plt.show()

# Buscar las caras en la imagen y devolver las posiciones detectadas con un rectángulo (x,y,w,h)
faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);

# Devolver el número de caras detectadas en la imagen
print('Caras encontradas: ', len(faces))

# Registrar la información de los rectángulos de las caras
for (x, y, w, h) in faces:
    cv2.rectangle(test1, (x, y), (x+w, y+h), (0, 255, 0), 2)
# Convertir la imagen a RGB y visualizarla
plt.imshow(convertToRGB(test1))
plt.show()