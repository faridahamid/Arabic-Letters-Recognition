# Arabic Letters Recognition

Arabic Letters Recognition is a machine learning project designed to classify handwritten Arabic characters using deep learning (CNN) and classical machine learning (SVM with and without feature extraction). The system recognizes 28 distinct Arabic letters from grayscale images represented by pixel values.

## Project Overview

This project explores multiple approaches to solve the classification task, including traditional SVM models, SVM with HOG features, and deep convolutional neural networks (CNNs). The dataset contains balanced and labeled examples for training and testing. Final evaluation was conducted using accuracy metrics and K-Fold cross-validation for reliability.

The goal is to build an effective classifier for Arabic handwritten letters and explore performance tradeoffs between simple models and deeper neural architectures.

## Dataset

- 13440 training images and 3360 testing images
- Each image is 32x32 grayscale (flattened into 1024 features in CSV files)
- 28 Arabic letter classes
- Files used:
  - `csvTrainImages 13440x1024.csv`
  - `csvTrainLabel 13440x1.csv`
  - `csvTestImages 3360x1024.csv`
  - `csvTestLabel 3360x1.csv`

## Models and Experiments

### Support Vector Machine (SVM)

- Trained directly on pixel data
- Accuracy: 65.3%

### SVM with HOG Features

- Applied Histogram of Oriented Gradients (HOG) on each image
- SVM trained on HOG-transformed data
- Accuracy improved to 73.0%

### Convolutional Neural Networks (CNN)

Two models were built:

#### Base CNN

- 2 Convolutional layers + MaxPooling + Dense
- Trained with K=5 cross-validation
- Achieved average validation accuracy around 92.4%

#### Deeper CNN

- 3 Convolutional layers, larger filters, and deeper dense layers
- Average validation accuracy: approximately 94.1%
- Final test accuracy: 95.03%

### Model Architecture (Deeper CNN)

```python
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(32,32,1)),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(256, (3,3), activation='relu'),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(28, activation='softmax')
])

## Preprocessing

- Normalized image pixel values to range `[0, 1]`
- Visualized random training samples using Matplotlib
- Encoded class labels using one-hot encoding (`to_categorical`)
- Applied **K-Fold cross-validation** to ensure generalization
- Implemented **early stopping** to prevent overfitting during CNN training

---

## Evaluation Summary

| Model        | Accuracy |
|--------------|----------|
| SVM          | 65.3%    |
| SVM + HOG    | 73.0%    |
| Base CNN     | ~92.4%   |
| Deeper CNN   | 95.0%    |

---
