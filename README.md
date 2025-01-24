# Emotion Classification Using Convolutional Neural Networks (CNN)

This project implements a Convolutional Neural Network (CNN) to classify images into 8 distinct emotions using grayscale images. The dataset is preprocessed, and the model is trained to extract features and perform multi-class classification.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Workflow](#project-workflow)
4. [Model Architecture](#model-architecture)
5. [Results](#results)
6. [Feature Map Visualization](#feature-map-visualization)
7. [How to Run the Code](#how-to-run-the-code)
8. [Future Improvements](#future-improvements)
9. [Acknowledgments](#acknowledgments)

---

## Project Overview
The goal of this project is to classify images into 8 different emotion categories using deep learning. The CNN model is designed to handle grayscale images, reducing computational overhead while maintaining performance.

---

## Dataset
- **Structure**: The dataset consists of images grouped into 8 emotion folders.
- **Labels**: Emotion labels are provided in a `labels.csv` file containing:
  - `Index`: Image index.
  - `Path`: Path to the image file.
  - `Label`: Emotion label.
- **Preprocessing**:
  - Images are resized to `128x128` pixels.
  - Converted to grayscale and normalized to pixel values between 0 and 1.

---

## Project Workflow
1. **Data Loading**:
   - Load `labels.csv` and preprocess images into a standardized format.
2. **Preprocessing**:
   - Resize images to `128x128` and normalize pixel values.
   - Store processed images in `.npy` format for efficiency.
3. **Model Training**:
   - Train a CNN model with 3 convolutional layers and pooling layers.
4. **Evaluation**:
   - Evaluate the model's accuracy and loss on a test set.
5. **Feature Map Visualization**:
   - Visualize intermediate outputs of convolution and pooling layers.

---

## Model Architecture
The CNN model consists of:
- **Input Layer**: `(128, 128, 1)` for grayscale images.
- **Convolutional Layers**:
  - 3 layers with filters increasing from `32` to `128`.
- **Pooling Layers**:
  - MaxPooling layers to reduce spatial dimensions.
- **Fully Connected Layers**:
  - Flattened outputs are passed through dense layers for classification.
- **Output Layer**:
  - Softmax activation for predicting 8 emotion classes.

---

## Results
- **Training Accuracy**: Achieved an accuracy of ~XX% after 10 epochs.
- **Test Accuracy**: Evaluated the model with a test accuracy of ~YY%.
- **Loss**: The model converged with a final loss of ~ZZ%.

---

## Feature Map Visualization
Feature maps from convolutional and pooling layers have been saved to the `../assets/output` directory. These visualizations show how the model extracts spatial features at each layer.

### Example Visualization:
*Layer: `conv2d_1`, Channel: 0*

![Feature Map Example](../assets/output/conv2d_1_channel_0.png)

---

## How to Run the Code
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
