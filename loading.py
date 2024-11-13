import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_images_from_directory(directory):
    images = []
    labels = []
    for class_name in os.listdir(directory):
        # print(f"class- name: {class_name}")
        class_folder = os.path.join(directory, class_name)
        if os.path.isdir(class_folder):
            for filename in os.listdir(class_folder):
                if filename.endswith(".png"):
                    image_path = os.path.join(class_folder, filename)
                    try:
                        image = Image.open(image_path)
                        image_array = np.array(image.resize((224, 224)))
                        image_array = image_array/255
                        print(f"type np: {type(image_array)}")
                        images.append(image_array)
                        if "covid_" in image_path:
                            labels.append(0)
                        else:
                            labels.append(1)
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")

    return images, labels


x, y = load_images_from_directory("Infection Segmentation Data/Test/COVID-19")
print(x)

plt.imshow(x[5])
# print(x[5].shape)
plt.show()

