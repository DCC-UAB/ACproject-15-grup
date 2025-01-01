import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt

def detect_and_crop_uniform_regions(image, threshold=1):

    if len(image.shape) != 2:  # Assegurar que és en escala de grisos
        raise ValueError("La imatge no és en escala de grisos.")

    h, w = image.shape

    # Detectar canvis en totes les direccions
    diff_x = np.abs(np.diff(image, axis=1)) > threshold
    diff_y = np.abs(np.diff(image, axis=0)) > threshold
    diff_diag1 = np.abs(image[1:, 1:] - image[:-1, :-1]) > threshold  # Diagonal principal
    diff_diag2 = np.abs(image[1:, :-1] - image[:-1, 1:]) > threshold  # Diagonal secundària

    # Crear màscara unificada per totes les direccions
    mask_x = np.pad(diff_x, ((0, 0), (1, 0)), mode='constant', constant_values=False)
    mask_y = np.pad(diff_y, ((1, 0), (0, 0)), mode='constant', constant_values=False)
    mask_diag1 = np.pad(diff_diag1, ((1, 0), (1, 0)), mode='constant', constant_values=False)
    mask_diag2 = np.pad(diff_diag2, ((1, 0), (0, 1)), mode='constant', constant_values=False)

    combined_mask = mask_x | mask_y | mask_diag1 | mask_diag2

    # Coordenades dels píxels no uniformes
    coords = np.argwhere(combined_mask)
    if coords.size == 0:  # Si tota la imatge és uniforme
        return image

    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0) + 1  # Incloure límits
    cropped_image = image[x_min:x_max, y_min:y_max]
    return cropped_image

def make_image_square(image, size=224):
    height, width = image.shape[:2]
    if height > width:
        diff = (height - width) // 2
        square_image = cv2.copyMakeBorder(image, 0, 0, diff, diff, cv2.BORDER_CONSTANT, value=0)
    elif width > height:
        diff = (width - height) // 2
        square_image = cv2.copyMakeBorder(image, diff, diff, 0, 0, cv2.BORDER_CONSTANT, value=0)
    else:
        square_image = image

    return cv2.resize(square_image, (size, size))

def clean_and_save_images(dataset, labels, output_path, size=224, threshold=1):
    cleaned_images = []
    valid_labels = []
    for i, image in enumerate(dataset):
        try:
            if image is None or not isinstance(image, np.ndarray) or len(image.shape) < 2:
                print(f"Imatge {i} no vàlida. Saltant...")
                continue

            cleaned_image = detect_and_crop_uniform_regions(image, threshold=threshold)
            square_image = make_image_square(cleaned_image, size=size)
            cleaned_images.append(square_image)
            valid_labels.append(labels[i])  # Només guardar l'etiqueta si la imatge és vàlida
        except Exception as e:
            print(f"Error processant la imatge {i}: {e}")
            continue

    # Guardar les imatges processades i les etiquetes en un pickle
    with open(output_path, 'wb') as f:
        pickle.dump((cleaned_images, valid_labels), f)

    return cleaned_images, valid_labels

def visualize_images_comparison(original_images, cleaned_images, labels, index=0):
    if index >= len(original_images) or index >= len(cleaned_images):
        print("Índex fora de rang!")
        return

    original = original_images[index]
    cleaned = cleaned_images[index]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Netejada - Etiqueta: {labels[index]}")
    plt.imshow(cleaned, cmap='gray')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    input_pickle_path = 'data/dataset.pkl'  # Ruta al pickle original
    output_pickle_path = 'data/cleaned_dataset.pkl'  # Ruta per guardar el nou pickle

    print("Carregant el dataset original...")
    with open(input_pickle_path, 'rb') as f:
        dataset, labels = pickle.load(f)

    print("Processant les imatges del dataset...")
    cleaned_images, labels = clean_and_save_images(dataset, labels, output_pickle_path, size=224, threshold=3)

    print("Mostrant una imatge de comparació...")
    visualize_images_comparison(dataset, cleaned_images, labels, index=12)



'''
NO EXECUTAR FRAGMENT QUE SUBSTITUEIX TOTES LES IMATGES ORIGINALS PER A LES NETEJADES
def clean_and_save_images_to_disk(input_dir, output_dir, size=224, threshold=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, relative_path)

        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                try:
                    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        print(f"Error carregant la imatge: {input_path}")
                        continue

                    # Netejar i ajustar dimensions
                    cleaned_image = detect_and_crop_uniform_regions(image, threshold=threshold)
                    square_image = make_image_square(cleaned_image, size=size)

                    # Desa la imatge processada
                    output_path = os.path.join(output_subdir, file)
                    cv2.imwrite(output_path, square_image)
                    print(f"Imatge processada desada a: {output_path}")
                except Exception as e:
                    print(f"Error processant la imatge {input_path}: {e}")

if __name__ == "__main__":
    input_dir = "data/Cervical_Cancer"  
    output_dir = "data/Cervical_Cancer" 

    print("Processant i desant les imatges al directori de sortida...")
    clean_and_save_images_to_disk(input_dir, output_dir, size=224, threshold=1)
'''


'''
COMPROVA REGIONS VERTICALS I HORITZONTALS
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle


def remove_uniform_borders(image, threshold):
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Detectar canvis horitzontalment i verticalment
    diff_x = np.abs(np.diff(gray, axis=1)) > threshold
    diff_y = np.abs(np.diff(gray, axis=0)) > threshold

    # Trobar límits horitzontals i verticals amb variació
    mask_x = diff_x.any(axis=0)
    mask_y = diff_y.any(axis=1)

    if not mask_x.any() or not mask_y.any():  # Si no hi ha cap variació significativa
        return image

    x_min, x_max = np.where(mask_x)[0][[0, -1]]
    y_min, y_max = np.where(mask_y)[0][[0, -1]]

    return image[y_min:y_max + 1, x_min:x_max + 1]


def make_image_square(image, size=224):
    height, width = image.shape[:2]
    if height > width:
        diff = (height - width) // 2
        square_image = cv2.copyMakeBorder(image, 0, 0, diff, diff, cv2.BORDER_CONSTANT, value=0)
    elif width > height:
        diff = (width - height) // 2
        square_image = cv2.copyMakeBorder(image, diff, diff, 0, 0, cv2.BORDER_CONSTANT, value=0)
    else:
        square_image = image

    return cv2.resize(square_image, (size, size))


def clean_and_save_images(dataset, labels, output_path, size=224, threshold=5):
    cleaned_images = []
    for idx, (image, label) in enumerate(zip(dataset, labels)):
        try:
            cleaned_image = remove_uniform_borders(image, threshold)
            square_image = make_image_square(cleaned_image, size)
            cleaned_images.append(square_image)
        except Exception as e:
            print(f"Error processant la imatge {idx}: {e}")
    
    with open(output_path, 'wb') as f:
        pickle.dump((cleaned_images, labels), f)

    return cleaned_images, labels


def visualize_images_comparison(original_images, cleaned_images, labels, index=0):
    original = original_images[index]
    cleaned = cleaned_images[index]
    label = labels[index]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cleaned, cmap="gray")
    plt.title(f"Netejada - Etiqueta: {label}")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    input_pickle_path = "data/dataset.pkl"
    output_pickle_path = "data/cleaned_dataset.pkl"

    with open(input_pickle_path, 'rb') as f:
        dataset, labels = pickle.load(f)

    print("Processant les imatges del dataset...")
    cleaned_images, labels = clean_and_save_images(dataset, labels, output_pickle_path, size=224, threshold=3)

    print("Comparant una imatge original amb la seva versió netejada...")
    visualize_images_comparison(dataset, cleaned_images, labels, index=12)
'''