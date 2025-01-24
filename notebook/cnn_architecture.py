import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# Step 1: Define the CNN model
model = Sequential([
    # Convolutional layer 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),  # Grayscale input
    MaxPooling2D((2, 2)),  # Reduce spatial dimensions by half

    # Convolutional layer 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Convolutional layer 3
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Flatten layer
    Flatten(),

    # Fully connected layer 1
    Dense(128, activation='relu'),
    Dropout(0.5),  # Dropout for regularization

    # Output layer
    Dense(8, activation='softmax')  # 8 output classes with softmax for probabilities
])

# Step 2: Compile the model
model.compile(
    optimizer='adam',  # Adam optimizer for adaptive learning
    loss='categorical_crossentropy',  # Loss function for multi-class classification
    metrics=['accuracy']  # Evaluate with accuracy
)

# Step 3: Print model summary
model.summary()

# Step 4: Train the model
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=10,  # Start with 10 epochs (adjust based on performance)
    verbose=1
)

# Step 5: Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
