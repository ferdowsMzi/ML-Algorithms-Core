# Linear Regression with SGD & Regularization

This project implements a custom **Linear Regression** model trained via **Stochastic / Mini-batch Gradient Descent (SGD)**, built entirely from scratch. The primary objective is to predict housing prices using the **AmesHousing** dataset while exploring the mathematical mechanics of gradient descent, hyperparameter tuning, and weight regularization techniques (`L1` and `L2`).

## Dataset & Exploratory Data Analysis (EDA)

The model is trained and evaluated on `AmesHousing.csv`, a comprehensive dataset commonly used for regression tasks.

Before training, the project performs visual exploratory data analysis. A custom `plot_relations` function generates scatter plots to inspect the linear relationships between individual features and the target variable, helping identify informative predictors and potential outliers before modeling.

## Mathematical Foundations

The core model predicts the target using a linear combination of weights `W`, features `X`, and a bias term `b`:

- **Linear Prediction Rule:**  
  `y_hat = XW + b`

### 1. Cost Function & Regularization

The base objective is to minimize the Mean Squared Error (MSE). To reduce overfitting and handle multicollinearity, the model supports adding penalty terms to the loss function `J(W, b)`.

- **L1 Regularization (Lasso):** Encourages sparsity by penalizing the absolute magnitude of the weights.  
  `J(W, b) = (1 / N) * sum((y_hat_i - y_i)^2) + lambda * sum(|w_j|)`

- **L2 Regularization (Ridge):** Penalizes large weights smoothly, keeping all features but shrinking their influence.  
  `J(W, b) = (1 / N) * sum((y_hat_i - y_i)^2) + lambda * sum(w_j^2)`

### 2. Gradient Descent Updates

During training, the weights and bias are updated iteratively in the opposite direction of the gradient, scaled by the learning rate `eta`.

- **Weight Update Rule:**  
  `W = W - eta * (dJ / dW)`

## Experiments & Hyperparameter Tuning

A major focus of this notebook is understanding how different training configurations affect convergence and final performance. The project systematically experiments with:

- **Learning Rate Sweep:** Testing different step sizes (`eta`) to find the balance between slow convergence and gradient explosion.
- **Batch Size Variations:** Comparing Pure SGD (`batch size = 1`), Mini-batch SGD, and Full Batch Gradient Descent to observe trade-offs in computational speed and optimization noise.
- **Regularization Effects:** Comparing the impact of `L1` vs. `L2` penalties on the learned weight distributions and validation metrics.

## Evaluation Metrics

The model's performance is rigorously evaluated using standard regression metrics:

- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **MAE:** Mean Absolute Error
- **R^2 Score:** Coefficient of Determination
