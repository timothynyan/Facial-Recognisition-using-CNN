#%%
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image

data = pd.read_csv('../assets/labels.csv')

image_paths = [f"../assets/{path}" for path in data['pth'].values]
labels = data['label'].values

train_data_file = "train_data.npy"
train_labels_file = "train_labels.npy"
test_data_file = "test_data.npy"
test_labels_file = "test_labels.npy"

def preprocess_image(image_path, target_size=(128, 128)):
    """Load and preprocess an image in grayscale."""
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize(target_size)  # Resize to target dimensions
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=-1)  # Add a channel dimension for TensorFlow
    return img_array

if all(os.path.exists(file) for file in [train_data_file, train_labels_file, test_data_file, test_labels_file]):
    # Load preprocessed data from disk
    X_train = np.load(train_data_file)
    y_train = np.load(train_labels_file)
    X_test = np.load(test_data_file)
    y_test = np.load(test_labels_file)
    print("Loaded preprocessed train and test data from disk.")
else:
    # Preprocess images
    images = np.array([preprocess_image(path) for path in image_paths])

    # Encode labels
    label_classes = sorted(list(set(labels)))
    label_to_index = {label: idx for idx, label in enumerate(label_classes)}
    encoded_labels = np.array([label_to_index[label] for label in labels])

    # Convert to one-hot encoding
    one_hot_labels = to_categorical(encoded_labels, num_classes=len(label_classes))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(images, one_hot_labels, test_size=0.2, random_state=42)

    # Save preprocessed train and test sets
    np.save(train_data_file, X_train)
    np.save(train_labels_file, y_train)
    np.save(test_data_file, X_test)
    np.save(test_labels_file, y_test)
    print("Preprocessed train and test data saved to disk.")

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=1000).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")
