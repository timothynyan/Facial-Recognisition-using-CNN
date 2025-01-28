#%% Importing libraries
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split



def preprocess_image(image_path, target_size=(128, 128)):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize(target_size)  # Resize image
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=-1)  # Add a channel dimension
    return img_array

#%% Preprocessing images
def data_processing():
    data =pd.read_csv('../assets/labels.csv')
    image_paths = [f"../assets/{path}" for path in data['pth'].values]
    labels = data['label'].values

    images = np.array([preprocess_image(path) for path in image_paths])
    print(f"Shape of images array: {images.shape}")

    #%% Encoding labels
    label_classes = sorted(list(set(labels)))
    label_to_index = {label: idx for idx, label in enumerate(label_classes)}
    encoded_labels = np.array([label_to_index[label] for label in labels])
    one_hot_labels = to_categorical(encoded_labels, num_classes=len(label_classes))
    print(f"Shape of one-hot encoded labels: {one_hot_labels.shape}")

    #%% Train-test split
    X_train, X_test, y_train, y_test = train_test_split(images, one_hot_labels, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")

    #%% plot distribution of labels for train and test set
    fig, axes = plt.subplots(1, 2, figsize=(30, 14))

    # Convert label indices to label names
    train_labels = [label_classes[idx] for idx in np.argmax(y_train, axis=1)]
    test_labels = [label_classes[idx] for idx in np.argmax(y_test, axis=1)]

    # Plot for the train set
    train_label_counts = pd.Series(train_labels).value_counts().sort_index()
    axes[0].barh(train_label_counts.index, train_label_counts.values)
    axes[0].set_title("Train set")
    axes[0].set_xlabel("Frequency")
    axes[0].set_ylabel("Label")

    # Plot for the test set
    test_label_counts = pd.Series(test_labels).value_counts().sort_index()
    axes[1].barh(test_label_counts.index, test_label_counts.values)
    axes[1].set_title("Test set")
    axes[1].set_xlabel("Frequency")
    axes[1].set_ylabel("Label")

    plt.tight_layout()
    plt.show()

    #%%

    return X_test, X_train, y_test, y_train



