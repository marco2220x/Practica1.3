import cv2
import numpy as np


def aplicar_suavizado_gaussiano(imagen, tamano_kernel):
    # Aplicar suavizado Gaussiano a la imagen
    kernel_gaussiano = np.outer(cv2.getGaussianKernel(tamano_kernel, 0), cv2.getGaussianKernel(tamano_kernel, 0).T)
    imagen_suavizada = cv2.filter2D(imagen, -1, kernel_gaussiano)
    return imagen_suavizada


def calcular_gradiente(imagen):
    # Calcular gradientes en las direcciones X e Y
    gradiente_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
    gradiente_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)

    # Calcular la magnitud del gradiente y la dirección
    magnitud_gradiente = np.sqrt(gradiente_x ** 2 + gradiente_y ** 2)
    direccion_gradiente = np.arctan2(gradiente_y, gradiente_x)

    return magnitud_gradiente, direccion_gradiente


def suprimir_no_maximos(magnitud_gradiente, direccion_gradiente):
    # Supresión de no máxima
    alto, ancho = magnitud_gradiente.shape
    magnitud_suprimida = np.zeros_like(magnitud_gradiente, dtype=np.uint8)

    for i in range(1, alto - 1):
        for j in range(1, ancho - 1):
            angulo = direccion_gradiente[i, j] * 180 / np.pi
            q = magnitud_gradiente[i, j + 1]
            r = magnitud_gradiente[i, j - 1]

            if (0 <= angulo < 22.5) or (157.5 <= angulo <= 180):
                q = magnitud_gradiente[i, j + 1]
                r = magnitud_gradiente[i, j - 1]
            elif (22.5 <= angulo < 67.5):
                q = magnitud_gradiente[i + 1, j - 1]
                r = magnitud_gradiente[i - 1, j + 1]
            elif (67.5 <= angulo < 112.5):
                q = magnitud_gradiente[i + 1, j]
                r = magnitud_gradiente[i - 1, j]
            elif (112.5 <= angulo < 157.5):
                q = magnitud_gradiente[i - 1, j - 1]
                r = magnitud_gradiente[i + 1, j + 1]

            if (magnitud_gradiente[i, j] >= q) and (magnitud_gradiente[i, j] >= r):
                magnitud_suprimida[i, j] = magnitud_gradiente[i, j]

    return magnitud_suprimida


def umbral_histéresis(imagen, umbral_bajo, umbral_alto):
    # Histéresis de umbral
    umbral_debil = 25
    umbral_fuerte = 255

    fuertes_i, fuertes_j = np.where(imagen >= umbral_alto)
    debiles_i, debiles_j = np.where((imagen <= umbral_alto) & (imagen >= umbral_bajo))

    resultado = np.zeros_like(imagen, dtype=np.uint8)
    resultado[fuertes_i, fuertes_j] = umbral_fuerte
    resultado[debiles_i, debiles_j] = umbral_debil

    return resultado


def deteccion_bordes_canny(imagen, tamano_kernel=5, umbral_bajo=50, umbral_alto=150):
    # Paso 1: Aplicar suavizado Gaussiano
    imagen_suavizada = aplicar_suavizado_gaussiano(imagen, tamano_kernel)

    # Paso 2: Calcular el gradiente
    magnitud_gradiente, direccion_gradiente = calcular_gradiente(imagen_suavizada)

    # Paso 3: Supresión de no máxima
    magnitud_suprimida = suprimir_no_maximos(magnitud_gradiente, direccion_gradiente)

    # Paso 4: Umbralización histéresis
    bordes = umbral_histéresis(magnitud_suprimida, umbral_bajo, umbral_alto)

    return bordes


