# Advanced Machine Learning

This project contains various advanced machine learning techniques and experiments. The project is organized into several sections, each focusing on different topics and methods. Below is an overview of the main sections and their contents.

## Overview

1. **[LOWESS and KNN Implementation Using Usearch Library](https://github.com/akhipath03/AdvMachineLearning/tree/main/HW2)**: This section implements and compares various machine learning algorithms for regression and classification tasks. It includes custom implementations of LOWESS (Locally Weighted Scatterplot Smoothing), Gradient Boosting, and K-Nearest Neighbors, as well as comparisons with XGBoost.

2. **SCAD, Elastic Net, and Square Root Lasso Comparison**: This section implements and compares various regularization techniques in machine learning, focusing on SCAD (Smoothly Clipped Absolute Deviation), Elastic Net, and Square Root Lasso. It includes custom PyTorch implementations of these methods and a comparison of their performance on simulated data with a strong correlation structure.

3. **CNN Optimization with Particle Swarm Optimization**: This section implements a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset and uses Particle Swarm Optimization (PSO) to find optimal hyperparameters for the network architecture.

4. **Data Sampling and Classification with FastKDE, SMOTE, and ADASYN**: This section demonstrates data sampling techniques using FastKDE, SMOTE, and ADASYN, and tests the effectiveness of these techniques on a K-Nearest Neighbors (KNN) classifier.

---


## 1. LOWESS and KNN Implementation Using Usearch Library

This project implements and compares various machine learning algorithms for regression and classification tasks. It includes custom implementations of LOWESS (Locally Weighted Scatterplot Smoothing), Gradient Boosting, and K-Nearest Neighbors, as well as comparisons with XGBoost.

## Table of Contents
- Requirements
- Installation
- Usage
- Algorithms
- Results

## Requirements

- Python 3.9
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost
- USearch

## Installation

1. Install the required packages:
   ```
   pip install numpy pandas matplotlib scikit-learn xgboost usearch
   ```

## Usage

To run the project:

1. Ensure you have the `concrete.csv` dataset in the project directory.
2. Run the notebook

## Algorithms

1. **LOWESS (Locally Weighted Scatterplot Smoothing)**: A non-parametric regression method.
2. **Gradient Boosting with LOWESS**: A custom implementation of gradient boosting using LOWESS as the base learner.
3. **K-Nearest Neighbors**: A custom implementation for classification tasks.
4. **XGBoost**: Used for comparison in regression tasks.

## Results

The project compares the performance of different algorithms:

1. Gradient Boosting with LOWESS vs XGBoost for regression on the concrete dataset.
2. K-Nearest Neighbors for classification on the Iris dataset, with an analysis of accuracy vs. k value.

Detailed results and visualizations are generated when running the main script.

---










## 2. SCAD, Elastic Net, and Square Root Lasso Comparison

This project implements and compares various regularization techniques in machine learning, focusing on SCAD (Smoothly Clipped Absolute Deviation), Elastic Net, and Square Root Lasso. It includes custom PyTorch implementations of these methods and a comparison of their performance on simulated data with a strong correlation structure.

## Table of Contents
- Requirements
- Installation
- Implemented Methods
- Usage
- Results

## Requirements

- Python 3.9
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Pytorch-ignite

## Installation

1. Install the required packages:
   ```
   pip install torch numpy pandas matplotlib seaborn scikit-learn scipy pytorch-ignite
   ```

## Implemented Methods

1. **SCAD (Smoothly Clipped Absolute Deviation)**: A regularization method that addresses the bias problem in the LASSO.
2. **Elastic Net**: A regularization method that linearly combines the L1 and L2 penalties of the LASSO and ridge methods.
3. **Square Root Lasso**: A method that uses the square root of the usual LASSO criterion, making it pivotal and easier to tune.

## Usage

To run the project:

1. Ensure you have all required packages installed.
2. Run the notebook

This script will:
- Generate correlated data
- Train models using SCAD, Elastic Net, and Square Root Lasso
- Compare the results with the true beta coefficients

## Results

The project compares the performance of SCAD, Elastic Net, and Square Root Lasso on simulated data with a strong correlation structure. It provides:

1. Estimated coefficients for each method
2. Comparison with true beta coefficients
3. Mean Squared Error (MSE) for each method

The results demonstrate how each method handles feature selection and coefficient estimation in the presence of multicollinearity.

---













## 3. CNN Optimization with Particle Swarm Optimization

This project implements a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset and uses Particle Swarm Optimization (PSO) to find optimal hyperparameters for the network architecture.

## Table of Contents
- Requirements
- Installation
- Model Architecture
- Hyperparameter Optimization
- Usage
- Results

## Requirements

- Python 3.9
- PyTorch
- torchvision
- pyswarms
- CUDA-capable GPU (recommended)

## Installation

1. Install the required packages:
   ```
   pip install torch torchvision pyswarms
   ```

## Model Architecture

The CNN model (`MyModel`) has the following structure:
- Initial convolutional layer with batch normalization, ReLU activation, and max pooling
- Variable number of convolutional layers (determined by PSO)
- Fully connected layers

The model is designed to be flexible, with the number of layers and neurons per layer as optimizable parameters.

## Hyperparameter Optimization

Particle Swarm Optimization (PSO) is used to find the optimal hyperparameters:
- Number of convolutional layers (1-5)
- Number of neurons in the fully connected layer (16-256)

The optimization process aims to maximize the validation accuracy on the CIFAR-10 dataset.

## Usage

To run the project:

1. Ensure you have all required packages installed.
2. Run the notebook

This script will:
- Define the CNN model
- Set up the PSO optimizer
- Train multiple models with different hyperparameters
- Output the best hyperparameters found

## Results

The script will output:
- Training progress (loss and accuracy) for each epoch
- The best hyperparameters found by PSO

Example output:
```
Epoch [20/20], Loss: 1.5432, Accuracy: 85.67 %
Best hyperparameters: [3, 128]
```

This indicates that the best model found has 3 convolutional layers and 128 neurons in the fully connected layer.

---










## 4. Data Sampling and Classification with FastKDE, SMOTE, and ADASYN

This project demonstrates data sampling techniques using FastKDE, SMOTE, and ADASYN, and tests the effectiveness of these techniques on a K-Nearest Neighbors (KNN) classifier.

## Libraries Used

- scikit-learn-intelex
- numpy
- usearch
- fastKDE

### Installation

```bash
!pip install -q scikit-learn-intelex
!pip install numpy usearch
!pip install -q fastKDE
```

### Imports

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from usearch.index import search, MetricKind
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
import fastkde
```

## Project Overview

### 1. Creating FastKDE Function and Showing Sampling

We start by creating a custom sampling function using the `fastkde` library. This function helps us understand and visualize the sampling process.

### 2. Creating Dataset and Sampling Data with SMOTE, ADASYN, and FastKDE

We generate a synthetic dataset using `make_classification` from `scikit-learn`. The dataset is then sampled using three different techniques:
- **SMOTE (Synthetic Minority Over-sampling Technique)**
- **ADASYN (Adaptive Synthetic Sampling)**
- **FastKDE (Kernel Density Estimation)**

### 3. Testing Data Sampling on KNN Classifier

To evaluate the effectiveness of the sampling techniques, we train a K-Nearest Neighbors (KNN) classifier on the original and sampled datasets. Cross-validation scores are used to compare the performance of each sampling method.

## Custom FastKDE Sampling Function

The custom sampling function was created using the `fastkde` library to enhance the data distribution and improve classifier performance.

## Usage

1. Install the required libraries
2. Run the notebook

## Results

The project provides a detailed comparison of different sampling techniques and their impact on the KNN classifier performance. The results are visualized using matplotlib.

## Conclusion

This project highlights the importance of data sampling in machine learning, especially for imbalanced datasets. By leveraging advanced sampling techniques like FastKDE, we can significantly improve the performance of machine learning models.

---

