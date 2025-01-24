import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from data_processing import train_dataset,test_dataset


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

model.summary()

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=10,  
    verbose=1
)

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
