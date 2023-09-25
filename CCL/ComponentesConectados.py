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

# Etiquetar componentes conectados
current_label = 1

for y in range(height):
    for x in range(width):
        if binary_image[y, x] == 255:
            neighbors = []
            if y > 0 and binary_image[y - 1, x] == 255:
                neighbors.append(labels[y - 1, x])
            if x > 0 and binary_image[y, x - 1] == 255:
                neighbors.append(labels[y, x - 1])

            if not neighbors:
                labels[y, x] = current_label
                current_label += 1
            else:
                min_neighbor = min(neighbors)
                labels[y, x] = min_neighbor
                for neighbor in neighbors:
                    if neighbor != min_neighbor:
                        union(parents, min_neighbor, neighbor)

# Recorrer la matriz de etiquetas para actualizar los componentes conectados
for y in range(height):
    for x in range(width):
        if binary_image[y, x] == 255:
            labels[y, x] = find(parents, labels[y, x])

# Crear un mapeo de colores para los componentes
color_map = {}
colored_image = np.zeros((height, width, 3), dtype=np.uint8)

for y in range(height):
    for x in range(width):
        if binary_image[y, x] == 255:
            label = labels[y, x]
            if label not in color_map:
                color_map[label] = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            colored_image[y, x] = color_map[label]

# Imprimir los componentes detectados (padres e hijos)
component_info = {}
for y in range(height):
    for x in range(width):
        if binary_image[y, x] == 255:
            label = labels[y, x]
            root = find(parents, label)
            if root not in component_info:
                component_info[root] = []
            component_info[root].append((x, y))

for root, components in component_info.items():
    print(f"Componente {root}:")
    for x, y in components:
        print(f"Hijo en ({x}, {y})")

# Guardar la imagen etiquetada
cv2.imwrite('componentes_conectados.jpg', colored_image)

# Mostrar la imagen etiquetada
cv2.imshow(colored_image)
cv2.waitKey(0)
cv2.destroyAllWindows()