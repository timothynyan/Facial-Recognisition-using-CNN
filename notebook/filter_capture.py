import os
import matplotlib.pyplot as plt
import numpy as np

# Function to visualize and save feature maps for all test samples
def visualize_and_save_feature_maps(model, test_dataset, output_dir="./output_layer"):
    """
    Visualize and save feature maps for all test samples and all Conv2D layers.

    Args:
        model: Trained Keras model.
        test_dataset: TensorFlow dataset containing test data.
        output_dir: Base directory to save the feature maps.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all Conv2D layers
    for layer_idx, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            print(f"Processing layer {layer_idx}: {layer.name}")

            # Create an intermediate model for this layer
            intermediate_model = tf.keras.Model(inputs=model.layers[0].input, outputs=layer.output)

            # Create a directory for this layer
            layer_dir = os.path.join(output_dir, f"layer_{layer_idx}_{layer.name}")
            os.makedirs(layer_dir, exist_ok=True)

            # Process each test sample
            for sample_idx, (batch_images, batch_labels) in enumerate(test_dataset):
                for image_idx, sample_image in enumerate(batch_images):
                    # Add batch dimension to the image
                    sample_image = np.expand_dims(sample_image, axis=0)

                    # Get the feature maps for this sample
                    feature_maps = intermediate_model.predict(sample_image)

                    # Remove batch dimension
                    feature_maps = feature_maps[0]  # Shape: (height, width, num_filters)

                    # Plot feature maps for all filters
                    num_filters = feature_maps.shape[-1]
                    grid_size = int(np.ceil(np.sqrt(num_filters)))
                    plt.figure(figsize=(12, 12))
                    for filter_idx in range(num_filters):
                        plt.subplot(grid_size, grid_size, filter_idx + 1)
                        plt.imshow(feature_maps[..., filter_idx], cmap="viridis")
                        plt.axis("off")

                    # Save the plot
                    sample_dir = os.path.join(layer_dir, f"sample_{sample_idx}_{image_idx}")
                    os.makedirs(sample_dir, exist_ok=True)
                    save_path = os.path.join(sample_dir, f"filter_maps.png")
                    plt.suptitle(f"Feature Maps - {layer.name}, Sample {sample_idx}_{image_idx}", fontsize=16)
                    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
                    plt.close()

            print(f"Feature maps for layer {layer.name} saved in {layer_dir}")

# Call the function to visualize and save feature maps
visualize_and_save_feature_maps(model, test_dataset, output_dir="./output_layer")
