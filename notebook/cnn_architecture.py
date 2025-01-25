#%% initializing
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from helper import load_dataset
# from data_processing import train_dataset, test_dataset

train_data_file = "../data_cache/train_data.npy"
train_labels_file = "../data_cache/train_labels.npy"
test_data_file = "../data_cache/test_data.npy"
test_labels_file = "../data_cache/test_labels.npy"

# Define the model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(8, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


model = create_model()
#%% load Dataset
train_dataset, test_dataset= load_dataset(train_data_file,train_labels_file,test_data_file,test_labels_file)

 #%% Model Loading
# Set up a directory to save weights
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "model.weights.h5")

# Add ModelCheckpoint callback to save weights during training
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,  # Save only the weights (not the entire model)
    save_best_only=True,    # Set to True if you want to save only the best weights
    verbose=1
)

# Load weights if available
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    print(f"Loading weights from {latest_checkpoint}")
    model.load_weights(latest_checkpoint)

model.summary()

# Train the model
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=1,
    callbacks=[checkpoint],
    verbose=1
)
#%% Evaluate the model

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# %%
