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

preprocessed_data_file = "preprocessed_images.npy"
preprocessed_labels_file = "preprocessed_labels.npy"

def preprocess_image(image_path, target_size=(128, 128)):
    """Load and preprocess an image in grayscale."""
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize(target_size)  # Resize to target dimensions
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=-1)  # Add a channel dimension for TensorFlow
    return img_array

if os.path.exists(preprocessed_data_file) and os.path.exists(preprocessed_labels_file):
    images = np.load(preprocessed_data_file)
    encoded_labels = np.load(preprocessed_labels_file)
    print("Loaded preprocessed data from disk.")
else:
    # Preprocess images
    images = np.array([preprocess_image(path) for path in image_paths])

    # Encode labels
    label_classes = sorted(list(set(labels)))  
    label_to_index = {label: idx for idx, label in enumerate(label_classes)}
    encoded_labels = np.array([label_to_index[label] for label in labels])

    np.save(preprocessed_data_file, images)
    np.save(preprocessed_labels_file, encoded_labels)
    print("Preprocessed data and saved to disk.")

one_hot_labels = to_categorical(encoded_labels, num_classes=len(label_classes))

X_train, X_test, y_train, y_test = train_test_split(images, one_hot_labels, test_size=0.2, random_state=42)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=1000).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")
