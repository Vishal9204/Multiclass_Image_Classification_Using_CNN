# Multiclass Image Classification using CNN

This project implements a **Multiclass Image Classifier** using a Convolutional Neural Network (CNN) architecture inspired by VGGNet, but with a more compact and lightweight design suitable for smaller datasets and faster training.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Overview

The goal of this project is to classify images into multiple categories using a CNN. The model is designed with simplicity and efficiency in mind, using fewer layers and parameters than VGGNet, while still achieving competitive performance on multiclass image classification tasks.

## Features

- Implements a compact CNN architecture inspired by VGGNet.
- Efficient for training on smaller datasets with limited computational resources.
- Supports data augmentation for improved generalization.
- Provides evaluation metrics such as accuracy and confusion matrix.

## Dataset

The dataset consists of labeled images divided into multiple classes. Example datasets:

- **CIFAR-10**: 10 classes of 32x32 images (e.g., airplane, car, bird).
- **Custom Datasets**: Can be used by organizing images into subfolders for each class.

The dataset is preprocessed by:
- Resizing images to a uniform size.
- Normalizing pixel values to the range [0, 1].
- Applying data augmentation (e.g., rotations, flips, and zoom).

## Model Architecture

The model uses a compact CNN architecture with the following structure:

1. **Convolutional Layers**:
   - 2-3 convolutional layers with ReLU activation and small filters (3x3).
   - MaxPooling layers to reduce spatial dimensions.

2. **Fully Connected Layers**:
   - Flattening layer followed by dense layers.
   - Dropout for regularization to prevent overfitting.

3. **Output Layer**:
   - Softmax activation for multiclass classification.

## Requirements

Install the required dependencies using the following:

```bash
pip install -r requirements.txt
```

### Key Libraries:
- Python 3.8+
- TensorFlow/Keras or PyTorch
- NumPy
- Pandas
- Matplotlib/Seaborn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multiclass-image-classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd multiclass-image-classification
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

1. Preprocess the dataset:
   ```bash
   python preprocess.py --dataset data/images
   ```
2. Train the model:
   ```bash
   python train.py --epochs 20 --batch_size 32
   ```

### Evaluating the Model

Evaluate the trained model on a test dataset:
```bash
python evaluate.py --test_set data/test_images
```

### Visualizing Results

Generate accuracy and loss plots:
```bash
python visualize.py --model model.h5 --data data/test_images
```

## Results

The model achieves an accuracy of approximately **85-90%** on benchmark datasets like CIFAR-10, demonstrating the effectiveness of the compact CNN architecture for multiclass image classification tasks.
