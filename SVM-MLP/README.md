# Model Interpretability and Robustness (SVM vs. MLP)

## Overview
This project compares the performance, interpretability, and robustness of Support Vector Machines (SVM) and Multi-Layer Perceptrons (MLP) when subjected to feature ablation. The goal is to analyze how the removal of features—both random and based on importance—affects each model's predictive accuracy.

## Dataset
- **California Housing Dataset**: Sourced from `sklearn.datasets`. The target variable is treated as a classification problem in this context (e.g., above/below a threshold), evaluated using metrics like F1 and AUC.

## Models Implemented

### 1. Support Vector Machine (SVM)
- **Implementation**: Uses `scikit-learn`'s `SVC` within a custom `SVMManager` class.
- **Hyperparameter Tuning**: Utilizes `GridSearchCV` to find the optimal configuration.
- **Best Parameters**: $C = 10$, `gamma` = 'auto', `kernel` = 'rbf'.
- **Baseline Performance**: Achieved a Test F1-Score of **0.8640** and an AUC of **0.9439**.

### 2. Multi-Layer Perceptron (MLP)
- **Implementation**: A custom neural network built from scratch using **PyTorch** (`nn.Sequential`).
- **Architecture**: Includes at least one hidden layer with 128 neurons.
- **Integration**: Wrapped in a custom `PyTorchMLPWrapper` (inheriting from `BaseEstimator`, `ClassifierMixin`) to maintain compatibility with `scikit-learn` pipelines.
- **Training Details**: Uses `BCEWithLogitsLoss`, an Adam/SGD optimizer, default learning rate of $0.01$, batch size of $64$, and runs for up to $100$ epochs.

## Experiments & Analysis

The core of the notebook focuses on feature importance and model degradation:

1. **Permutation Importance**: 
   - Evaluates the impact of each feature by shuffling its values and measuring the drop in performance (run with 10 repeats).
2. **Importance-Based Feature Ablation**: 
   - Systematically removes features one by one, starting from the least important to the most important.
   - Plots the "Test Accuracy" against the "Number of Features Remaining" for both SVM and MLP to visualize which model degrades faster.
3. **Random Feature Removal**: 
   - Drops features entirely at random across 10 independent trials.
   - Plots the Mean $\pm$ Standard Deviation of test accuracy to establish a baseline of robustness compared to the importance-based ablation.
