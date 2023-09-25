import numpy as np
import cv2
import matplotlib.pyplot as plt
from bordes import Canny
from bordes import Canny2

image_file = 'Images/um_000001.png'
original_image = cv2.imread(image_file, 1)
grayscale_image = cv2.imread(image_file, 0)

blurred = cv2.GaussianBlur(grayscale_image, (5, 5), 1.5)
canny_edges = cv2.Canny(blurred, 100, 200)
#canny_edges = Canny.deteccion_bordes_canny(grayscale_image)
#canny_edges = Canny.deteccion_bordes_canny(grayscale_image)

# Acumulador de hough para la imagen
def hough_lines_acc(img, rho_resolution=1, theta_resolution=1):
    height, width = img.shape #Altura y ancho para calcular la diagonal.
    img_diagonal = np.ceil(np.sqrt(height**2 + width**2)) # a**2 + b**2 = c**2
    rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution) #rhos
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution)) #thetas

    # Crear el acumulador Hough con dimensiones iguales de rhos y thetas
    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img) # encontrar todos los índices de píxeles de borde (distintos de cero)

    for i in range(len(x_idxs)): #Recorremos los puntos del borde
        x = x_idxs[i]
        y = y_idxs[i]

        for j in range(len(thetas)): #Recorremos thetas y calcular rho
            rho = int((x * np.cos(thetas[j]) +
                       y * np.sin(thetas[j])) + img_diagonal)
            H[rho, j] += 1

    return H, rhos, thetas

# Espacio de Hough
def hough_peaks(H, num_peaks, nhood_size=3):

    #Recorrer el numero de picos 
    indicies = []
    H1 = np.copy(H) #Crear una copia de H
    for i in range(num_peaks):
        idx = np.argmax(H1) # Encontrar lo maximos de de la matriz Hough
        H1_idx = np.unravel_index(idx, H1.shape) # Reasignar 
        indicies.append(H1_idx)

        idx_y, idx_x = H1_idx # Separamos los indices 
        # Elegir los valores apropiados si idx_x se encuentra cerca de los bordes
        if (idx_x - (nhood_size/2)) < 0: min_x = 0
        else: min_x = idx_x - (nhood_size/2)
        if ((idx_x + (nhood_size/2) + 1) > H.shape[1]): max_x = H.shape[1]
        else: max_x = idx_x + (nhood_size/2) + 1

        # Eligir los valores apropiados, si idx_y está demasiado cerca de los bordes
        if (idx_y - (nhood_size/2)) < 0: min_y = 0
        else: min_y = idx_y - (nhood_size/2)
        if ((idx_y + (nhood_size/2) + 1) > H.shape[0]): max_y = H.shape[0]
        else: max_y = idx_y + (nhood_size/2) + 1

        # Vincular cada índice (por el tamaño de la vecindad y establecer todos los valores en 0
        for x in range(int(min_x), int(max_x)):
            for y in range(int(min_y), int(max_y)):
                # eliminar vecindarios en H1
                H1[y, x] = 0

                # Resaltar picos en H
                if (x == min_x or x == (max_x - 1)):
                    H[y, x] = 255
                if (y == min_y or y == (max_y - 1)):
                    H[y, x] = 255

    # Devolver los índices y el espacio de Hough original con los puntos seleccionados
    return indicies, H


#Graficar el espacio de Hough
def plot_hough_acc(H):
    fig = plt.figure(figsize=(10, 10))
    
    plt.imshow(H, cmap='jet')
    plt.xlabel('Direccion Theta'), plt.ylabel('Direccion Rho')
    plt.tight_layout()
    plt.show()


#Graficar las lineas de Hough
def hough_lines_draw(img, indicies, rhos, thetas):
    ''' Toma indices de una tabla rhos y thetas y dibuja y dibuja las lineas 
    en la imagen que corresponde a estos valores'''
    for i in range(len(indicies)):
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


H, rhos, thetas = hough_lines_acc(canny_edges) # Calcula la acumuladora de Hough
indicies, H = hough_peaks(H, 5, nhood_size=11) # Encuentra los picos en la acumuladora de Hough
plot_hough_acc(H) # Mostrar el espacio de Hough
'''Toma la imagen original, los índices de los picos, las tablas de valores rho y theta, 
Dibuja las líneas correspondientes en la imagen original.'''
hough_lines_draw(original_image, indicies, rhos, thetas)
# Muestra la imagen con las lineas de transformacion de Hough
cv2.imshow('Deteccion de lineas con la transformada de Hough', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()