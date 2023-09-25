import cv2
import numpy as np

# Función para encontrar el representante (padre) de un conjunto
def find(parents, x):
    if parents[x] == x:
        return x
    parents[x] = find(parents, parents[x])
    return parents[x]

# Función para unir dos conjuntos
def union(parents, x, y):
    root_x = find(parents, x)
    root_y = find(parents, y)
    if root_x != root_y:
        parents[root_x] = root_y

# Cargar la imagen y convertirla a escala de grises
image = cv2.imread('cropped_parking_lot_1.JPG', cv2.IMREAD_GRAYSCALE)

# Binarizar la imagen usando un umbral (ajusta el valor según sea necesario)
_, binary_image = cv2.threshold(image, 60, 255, cv2.THRESH_BINARY)

# Obtener dimensiones de la imagen
height, width = binary_image.shape

# Inicializar una matriz para el etiquetado de componentes conectados
labels = np.zeros((height, width), dtype=int)

# Inicializar un conjunto de padres para el etiquetado
parents = list(range(height * width))  # Suponemos un máximo de height * width componentes

