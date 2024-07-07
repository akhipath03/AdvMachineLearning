# Advanced Machine Learning Project

This project implements and compares various machine learning algorithms for regression and classification tasks. It includes custom implementations of LOWESS (Locally Weighted Scatterplot Smoothing), Gradient Boosting, and K-Nearest Neighbors, as well as comparisons with XGBoost.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Algorithms](#algorithms)
- [Results](#results)

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
2. Run the main script:
   ```
   python main.py
   ```

## Project Structure

- `main.py`: The main script that runs all experiments
- `lowess.py`: Implementation of LOWESS algorithm
- `gradient_boosting.py`: Implementation of Gradient Boosting with LOWESS
- `knn.py`: Implementation of K-Nearest Neighbors
- `concrete.csv`: Dataset for regression task
- `README.md`: This file

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
