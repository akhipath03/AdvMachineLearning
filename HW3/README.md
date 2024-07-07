#  SCAD, Elastic Net, and Square Root Lasso Comparison

This project implements and compares various regularization techniques in machine learning, focusing on SCAD (Smoothly Clipped Absolute Deviation), Elastic Net, and Square Root Lasso. It includes custom PyTorch implementations of these methods and a comparison of their performance on simulated data with a strong correlation structure.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Implemented Methods](#implemented-methods)
- [Usage](#usage)
- [Results](#results)

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
