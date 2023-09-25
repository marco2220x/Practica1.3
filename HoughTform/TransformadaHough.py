import numpy as np
import cv2
import matplotlib.pyplot as plt
from Bordes import Canny
from Bordes import Canny2

image_file = 'um_000014.png'
original_image = cv2.imread(image_file, 1)
grayscale_image = cv2.imread(image_file, 0)

canny_edges = Canny.deteccion_bordes_canny(grayscale_image)

# Acumulador de hough para la imagen
def hough_lines_acc(img, rho_resolution=1, theta_resolution=1):
    height, width = img.shape #Altura y ancho para calcular la diagonal.
    img_diagonal = np.ceil(np.sqrt(height**2 + width**2)) # a**2 + b**2 = c**2
    rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution) #rhos
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution)) #thetas

    # Crear el acumulador Hough con dimensiones iguales de rhos y thetas
    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img) # find all edge (nonzero) pixel indexes

    for i in range(len(x_idxs)): #Recorremos los puntos del borde
        x = x_idxs[i]
        y = y_idxs[i]

        for j in range(len(thetas)): #Recorremos thetas y calcular rho
            rho = int((x * np.cos(thetas[j]) +
                       y * np.sin(thetas[j])) + img_diagonal)
            H[rho, j] += 1

    return H, rhos, thetas

H, rhos, thetas = hough_lines_acc(canny_edges) # Calcula la acumuladora de Hough