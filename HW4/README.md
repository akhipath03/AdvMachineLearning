# CNN Optimization with Particle Swarm Optimization

This project implements a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset and uses Particle Swarm Optimization (PSO) to find optimal hyperparameters for the network architecture.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Usage](#usage)
- [Results](#results)

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
