#%%
import os
from cnn_architecture import create_model,  Config
import tensorflow as tf
from helper import load_dataset
import pandas as pd
import matplotlib as plt



train_data_file = "../data_cache/train_data.npy"
train_labels_file = "../data_cache/train_labels.npy"
test_data_file = "../data_cache/test_data.npy"
test_labels_file = "../data_cache/test_labels.npy"
checkpoint_dir = "./checkpoints"

latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, "model.weights.h5")

train_dataset, test_dataset = load_dataset(
        train_data_file,
        train_labels_file,
        test_data_file,
        test_labels_file
    )

config = Config()

#%% Loading and checking model performance against testing data
model= create_model(config)
model.load_weights(checkpoint_path)
model.summary()
train_loss, train_accuracy = model.evaluate(train_dataset)
test_loss, test_accuracy = model.evaluate(test_dataset)
# show model performance in df
model_performance = pd.DataFrame({
    "Dataset": ["Train", "Test"],
    "Loss": [train_loss, test_loss],
    "Accuracy": [train_accuracy, test_accuracy]
})
print(model_performance)


# %%
images = []
labels = []
for batch in train_dataset:
    batch_images, batch_labels = batch
    images.extend(batch_images.numpy())
    labels.extend(batch_labels.numpy())


df = pd.DataFrame({
    'image': [img.squeeze() for img in images],  
    'label': [label.argmax() for label in labels]  
})

# Display images for each emotion
label_classes = sorted(list(set(df['label'])))
for label in label_classes:
    subset = df[df['label'] == label].head(4)
    fig, axes = plt.subplots(1, min(len(subset), 5), figsize=(20, 5))  
    fig.suptitle(f'Emotion: {label}')
    for img, ax in zip(subset['image'], axes):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.show()
# %%
