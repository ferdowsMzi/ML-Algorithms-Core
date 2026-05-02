# Multi-Class Logistic Regression (OvR & OvO)

This project implements **Logistic Regression** entirely from scratch, extending a binary classifier to handle multi-class predictions using **One-vs-Rest (OvR)** and **One-vs-One (OvO)** strategies. The model is applied to a medical dataset to predict the severity of **Heart Disease across 5 distinct levels** (Classes 0 through 4).

## Dataset & Task

The objective is to classify a patient's heart disease severity into one of 5 levels. Because standard logistic regression is inherently a binary classifier, this notebook builds the necessary wrappers to support multi-class classification, treating the target as levels 0–4.

## Mathematical Foundations

The core binary `LogisticRegression` class is built on the following mathematical principles:

### 1. Hypothesis (Sigmoid Activation)
The model outputs probabilities using the Sigmoid function, which is clipped in the code to prevent numerical overflow:
`$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$`
Where `$z = XW$`.

### 2. Loss Function (Binary Cross-Entropy + L2)
To prevent overfitting, the model uses **L2 Regularization (Ridge)** combined with the standard log-loss:
`$$ \text{Loss} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] + \frac{\lambda}{2m} \sum_{j} w_j^2 $$`
*(Where $m$ is the number of samples, and $\lambda$ is the regularization parameter `reg_lambda`)*.

### 3. Full-Batch Gradient Descent Update
The weights are updated iteratively using full-batch gradient descent:
`$$ \nabla = \frac{1}{m} X^T(\hat{y} - y) + \frac{\lambda}{m}w $$`
`$$ W = W - \eta \nabla $$`
*(Where `$\eta$` is the learning rate).*

## Multi-Class Strategies Implemented

### One-vs-Rest (OvR) Classifier
The `OneVsRestClassifier` trains **5 independent binary models**. For each class $c$ (from 0 to 4), a model is trained to predict whether a sample belongs to class $c$ or "the rest" (all other classes combined). 

### One-vs-One (OvO) Classifier
The `OneVsOneClassifier` trains a model for **every unique pair of classes**. For 5 classes, this results in $\frac{5 \times 4}{2} = 10$ individual binary classifiers. Each model is trained on a subset of the data containing only the two classes in question.

## Advanced Training Features

*   **Early Stopping:** The custom fit method tracks validation loss and stops training if the model fails to improve after a set `patience` limit, restoring the best weights. Logs show this triggering successfully (e.g., stopping around epochs 157 and 173).

## Experiments & Results

The notebook includes hyperparameter tuning for the OvR model, specifically testing different Learning Rates (`0.01` and `0.1`) with a fixed $\lambda = 0.1$:

*   **Learning Rate 0.01:** Achieved an accuracy of **64.52%** (Best configuration)
*   **Learning Rate 0.1:** Achieved an accuracy of **61.29%** (Converged faster, but at a lower accuracy)
