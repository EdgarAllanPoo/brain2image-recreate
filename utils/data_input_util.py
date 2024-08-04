import os
from random import randint

import PIL
import PIL.Image
import numpy as np
from PIL import Image

CHARACTER_CLASSES = {'A': 0, 'C': 1, 'F': 2, 'H': 3, 'J': 4, 'M': 5, 'P': 6, 'S': 7, 'T': 8, 'Y': 9}

# 1-hot encodes a tensor
def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]

def randomize(samples, labels):
    if type(samples) is np.ndarray:
        permutation = np.random.permutation(samples.shape[0])
        shuffle_samples = samples[permutation]
        shuffle_labels = labels[permutation]
    else:
        permutation = np.random.permutation(len(samples))
        shuffle_samples = [samples[i] for i in permutation]
        shuffle_labels = [labels[i] for i in permutation]

    return (shuffle_samples, shuffle_labels)


def load_char_data(char_fonts_folders, resize_shape, num_classes):

    images = []
    labels = []
    for char_fonts_folder in char_fonts_folders:
        for char_folder in os.listdir(char_fonts_folder):
            if char_folder != ".DS_Store":
                char_class = CHARACTER_CLASSES[char_folder]
                for char_img in os.listdir(os.path.join(char_fonts_folder, char_folder)):
                    file_path = os.path.join(char_fonts_folder, char_folder, char_img)
                    img = Image.open(file_path).resize(resize_shape, PIL.Image.NEAREST).convert('L')
                    img_array = 255 - np.array(img)
                    images.append(img_array)
                    labels.append(char_class)
    
    images, labels = randomize(images, labels)
    train_size = int(3 * len(images)/4)

    x_train, y_train = np.array(images[0: train_size]), np.array(labels[0: train_size])
    x_test, y_test = np.array(images[train_size:]), np.array(labels[train_size:])

    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_test = (x_test.astype(np.float32) - 127.5) / 127.5



    x_train = x_train[:, None, :, :]
    x_test = x_test[:, None, :, :]
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test

# char_fonts_folders = ["./data/images/Char-Font"]
# x_train, y_train, x_test, y_test = load_char_data(char_fonts_folders, resize_shape=(28, 28), num_classes=10)
# print("Loaded Characters Dataset.")
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)