#%% Initializing
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from data_pipeline import data_processing
import pandas as pd




# Configuration class for hyperparameters
class Config:
    def __init__(self):
        self.input_shape = (128, 128, 1)
        self.num_classes = 8
        self.conv_layers = [
            {'filters': 16, 'kernel_size': (3, 3), 'activation': 'relu'},
            {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'},
            {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'}
        ]
        self.pool_size = (2, 2)
        self.dense_units = 128  
        self.dropout_rate = 0.4 
        self.optimizer = 'adam'
        self.loss = 'categorical_crossentropy'
        self.metrics = ['accuracy']
        self.epochs = 20
        self.batch_size = 32
        self.checkpoint_dir = "./checkpoints"
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "model.weights.h5")

# Define the model
def create_model(config):
    model = Sequential()
    for layer in config.conv_layers:
        model.add(Conv2D(layer['filters'], layer['kernel_size'], activation=layer['activation'], input_shape=config.input_shape, kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())  # Added Batch Normalization
        model.add(MaxPooling2D(config.pool_size))
    model.add(Flatten())
    model.add(Dense(config.dense_units, activation='relu', kernel_regularizer=l2(0.01)))  # Added L2 regularization
    model.add(Dropout(config.dropout_rate))
    model.add(Dense(config.num_classes, activation='softmax'))

    model.compile(
        optimizer=config.optimizer,
        loss=config.loss,
        metrics=config.metrics
    )
    return model

#%% Main execution block
if __name__ == "__main__":
    # Load preprocessed data
    X_test, X_train, y_test, y_train = data_processing()
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=1000).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

    # Initialize configuration
    config = Config()

    # Create model
    model = create_model(config)

    # Display model summary
    model.summary()

    # Define callbacks
    checkpoint = ModelCheckpoint(config.checkpoint_path, save_best_only=True)
    early_stopping = EarlyStopping(patience=3)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=2)

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=config.epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    #%% Evaluate the model
    # Evaluate the model
    train_loss, train_accuracy = model.evaluate(train_dataset)
    test_loss, test_accuracy = model.evaluate(test_dataset)

    # Display model performance
    model_performance = pd.DataFrame({
        "Dataset": ["Train", "Test"],
        "Loss": [train_loss, test_loss],
        "Accuracy": [train_accuracy, test_accuracy]
    })
    print(model_performance)

    # Plot training history
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    # Save model
    model.save("emotion_classifier.h5")