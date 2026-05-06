# Support Vector Machine (SVM) Kernel and Parameter Analysis

This repository contains two projects exploring the behavior of Support Vector Machines (SVMs) using `scikit-learn`. The experiments investigate the impact of the regularization parameter `C` and compare the performance of different kernel functions (Linear, Polynomial, and RBF).

## Project 1: Linear SVM & Regularization Analysis

This experiment performs a deep dive into the role of the regularization parameter `C` in a Linear SVM.

### Objective
To empirically demonstrate and visualize how changing the `C` parameter affects:
*   The model's decision boundary and margin width.
*   The number of support vectors.
*   Train and test accuracy.
*   Training time.

### Dataset
*   **Iris Dataset**: Filtered to use only the **petal length** and **petal width** features for 2D visualization. This is a multi-class (3 classes) classification problem.

### Methodology
The core of the experiment is a `scikit-learn` `SVC` with a linear kernel:
`clf = SVC(kernel='linear', C=C)`
The model is evaluated over a sweep of `C` values: `[0.01, 0.1, 1, 10, 100]`. To ensure robust results, each `C` value is tested across **10 different random seeds**.

### Key Findings & Visualizations
The project tracks and compares accuracy, training time, and the number of support vectors for each `C` value. Visualizations of the decision boundary clearly show that:
*   **Small `C`** values lead to a **wider margin** and more support vectors (stronger regularization).
*   **Large `C`** values result in a **narrower margin** and fewer support vectors (weaker regularization), fitting the training data more tightly.

---

## Project 2: SVM Kernel Comparison

This experiment compares the decision boundaries and performance of three fundamental SVM kernels: Linear, Polynomial, and Radial Basis Function (RBF).

### Objective
To analyze and visualize the differences between SVM kernels on a non-linearly separable dataset, focusing on how each kernel's unique hyperparameters shape the final model.

### Dataset
*   **Breast Cancer Dataset**: Filtered to use only the **first two features**. This is a binary classification problem.

### Kernels Explored

1.  **Linear Kernel**
*   `kernel='linear'`
*   Separates data with a hyperplane.
*   Controlled primarily by the `C` parameter.

2.  **Polynomial Kernel**
*   `kernel='poly'`
*   Maps data to a higher-dimensional space to find a non-linear separator.
*   Key Hyperparameters: `C`, `degree`.

3.  **RBF (Radial Basis Function) Kernel**
*   `kernel='rbf'`
*   A powerful kernel that can create complex, localized decision boundaries.
*   Key Hyperparameters: `C`, `gamma`.

### Methodology
For each kernel, the notebook trains an `SVC` model, evaluates its performance, and plots the resulting decision boundary. The experiment measures the same key metrics: accuracy, training time, and the number of support vectors, allowing for a direct comparison of the kernels' effectiveness and complexity.

### Key Findings & Visualizations
The visualizations highlight the distinct capabilities of each kernel:
*   The **linear** kernel provides a straight-line boundary.
*   The **polynomial** kernel creates a curved, global boundary.
*   The **RBF** kernel produces intricate, localized boundaries that can closely fit complex data distributions.

## Common Implementation Details

*   **Framework**: Both projects are built using `scikit-learn`, `numpy`, and `matplotlib`.
*   **Evaluation Metrics**:
*   **Accuracy Score**: The primary metric for classification performance.
*   **Training Time**: Measured in milliseconds.
*   **Support Vector Count**: The number of data points that lie on the margin.
*   **Data Preprocessing**: A standard train-test split and `StandardScaler` are used to scale the features before training.
