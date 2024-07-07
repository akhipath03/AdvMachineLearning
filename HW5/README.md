
# Data Sampling and Classification with FastKDE, SMOTE, and ADASYN

This project demonstrates data sampling techniques using FastKDE, SMOTE, and ADASYN, and tests the effectiveness of these techniques on a K-Nearest Neighbors (KNN) classifier.

## Libraries Used

- [scikit-learn-intelex](https://github.com/intel/scikit-learn-intelex)
- [numpy](https://numpy.org/)
- [usearch](https://usearch.readthedocs.io/en/latest/)
- [fastKDE](https://pypi.org/project/fastKDE/)

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
