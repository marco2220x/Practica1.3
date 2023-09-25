# imports
import numpy as np

import matplotlib.pyplot as plt

def noise_reduction(grayscale_image, sigma, tamano_kernel):
    # Crear un kernel Gaussiano
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) * np.exp(-((x - (tamano_kernel - 1) / 2) ** 2 + (y - (tamano_kernel - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (tamano_kernel, tamano_kernel)
    )
    kernel = kernel / np.sum(kernel)  # Normalizar el kernel para que sume 1

    # Aplicar la convolución manualmente
    smoothed_image = np.zeros_like(grayscale_image, dtype=np.float64)
    grayscale_image = grayscale_image.astype(np.float64)

    for i in range(grayscale_image.shape[0] - tamano_kernel + 1):
        for j in range(grayscale_image.shape[1] - tamano_kernel + 1):
            ventana = grayscale_image[i:i+tamano_kernel, j:j+tamano_kernel]
            smoothed_image[i, j] = np.sum(ventana * kernel)

    # Convertir la imagen resultante a tipo uint8
    smoothed_image = smoothed_image.astype(np.uint8)
    
    return smoothed_image


def intensity_gradient(smoothed_image):
    Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    [rows, columns] = np.shape(smoothed_image)
    sobel_gm_image = np.zeros(shape=(rows, columns))  # inicialización del arreglo del gradiente de magnitud
    sobel_gd_image = np.zeros(shape=(rows, columns))  # inicialización del arreglo del gradiente de direccion

    for i in range(rows - 2):
        for j in range(columns - 2):
            gx = np.sum(np.multiply(Gx, smoothed_image[i:i + 3, j:j + 3]))  # dirección x
            gy = np.sum(np.multiply(Gy, smoothed_image[i:i + 3, j:j + 3]))  # dirección y
            sobel_gm_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)# calculamos la magnitud (mezclamos ambas direcciones)
            sobel_gd_image[i + 1, j + 1] = np.arctan2(gy, gx)
            
    return (sobel_gm_image, sobel_gd_image)


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)  # Crear una matriz de ceros para el resultado
    angle = D * 180. / np.pi  # Convertir la dirección del gradiente a grados
    angle[angle < 0] += 180  # Asegurarse de que los ángulos estén en el rango [0, 180]

    for i in range(1, M - 1):  # Recorrer filas (excepto los bordes)
        for j in range(1, N - 1):  # Recorrer columnas (excepto los bordes)
            try:
                q = 255  # Valor predeterminado para q
                r = 255  # Valor predeterminado para r

                # Comprobar la dirección del gradiente en el píxel actual
                # y determinar los píxeles vecinos correspondientes.

                # Ángulo 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # Ángulo 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # Ángulo 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # Ángulo 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                # Si el valor del píxel actual es mayor o igual que los vecinos,
                # se conserva en la matriz de salida (Z); de lo contrario, se establece en 0.
                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass  # Manejar excepciones de índice fuera de los bordes

    return Z  # Devolver la matriz resultante después de la supresión de no máximos


def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    # Calcular los umbrales alto y bajo a partir de los ratios y el valor máximo en la imagen.
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    
    M, N = img.shape  # Obtener las dimensiones de la imagen de entrada.
    res = np.zeros((M, N), dtype=np.int32)  # Crear una matriz de ceros para el resultado.
    
    # Definir valores para píxeles "débiles" y "fuertes".
    weak = np.int32(25)
    strong = np.int32(255)
    
    # Encontrar las coordenadas de los píxeles que superan el umbral alto.
    strong_i, strong_j = np.where(img >= highThreshold)
    
    # Encontrar las coordenadas de los píxeles que están por debajo del umbral bajo.
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    # Encontrar las coordenadas de los píxeles que están entre los umbrales alto y bajo.
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    # Asignar los valores "fuertes" a los píxeles que superan el umbral alto.
    res[strong_i, strong_j] = strong
    
    # Asignar los valores "débiles" a los píxeles que están entre los umbrales alto y bajo.
    res[weak_i, weak_j] = weak
    
    # Devolver la imagen resultante y los valores "débiles" y "fuertes".
    return (res, weak, strong)


def hysteresis(img, weak, strong=255):
    M, N = img.shape  # Obtener las dimensiones de la imagen de entrada.

    # Recorrer la imagen píxel por píxel (excepto los bordes).
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i, j] == weak):  # Si el píxel actual es débil.
                try:
                    # Comprobar si alguno de los píxeles vecinos es fuerte.
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong  # Asignar el valor fuerte al píxel actual.
                    else:
                        img[i, j] = 0  # Si no hay vecinos fuertes, establecer el valor en 0.
                except IndexError as e:
                    pass  # Manejar excepciones de índice fuera de los bordes.

    return img  # Devolver la imagen después de aplicar el proceso de histeresis.


def canny_detection(grayscale_image):
    # Paso 1: Aplicar suavizado Gaussiano
    tamano_kernel = 5  # Tamaño de la máscara (debe ser un número impar)
    sigma = 1 #Valor de sigma

    smoothed_image = noise_reduction(grayscale_image, sigma, tamano_kernel)

    # Paso 2: Calcular el gradiente
    sobel_gm_image, sobel_gd_image = intensity_gradient(smoothed_image)

    # Paso 3: Supresión de no máxima
    img_nms = non_max_suppression(sobel_gm_image, sobel_gd_image)

    # Paso 4: Umbralización histéresis
    img_thresh, weak, strong = threshold(img_nms, lowThresholdRatio=0.07, highThresholdRatio=0.19)
    canny_edges = hysteresis(img_thresh, weak, strong=strong)

    return canny_edges
