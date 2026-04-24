# K-Nearest Neighbors (KNN)

This repository contains a complete from-scratch implementation of the K-Nearest Neighbors (KNN) algorithm in Python. Beyond a basic classifier, the project functions as an experimental suite that investigates how KNN behaves under different conditions, including feature scaling, feature selection, alternative distance metrics, and noisy data.

The model is evaluated on the **Breast Cancer Wisconsin (Diagnostic) Dataset** for binary classification.

---

## Extended Mathematical Foundations

A core part of this project involves implementing and comparing different distance metrics to evaluate how the definition of "proximity" affects model accuracy.

### 1. Distance Metrics

- **Euclidean Distance (`L2` norm):** The straight-line distance between two points.  
  `d(x, x_train) = ||x - x_train||_2 = sqrt(sum((x_i - x_train,i)^2))`  
  *Code snippet:* `np.linalg.norm(self.X_train - x, axis=1, ord=2)`

- **Manhattan Distance (`L1` norm):** The distance between two points measured along axes at right angles.  
  `d(x, x_train) = ||x - x_train||_1 = sum(|x_i - x_train,i|)`  
  *Code snippet:* `np.linalg.norm(self.X_train - x, axis=1, ord=1)`

- **Chebyshev Distance (`L∞` norm):** The maximum absolute difference between any single coordinate dimension.  
  `d(x, x_train) = ||x - x_train||_∞ = max(|x_i - x_train,i|)`  
  *Code snippet:* `np.linalg.norm(self.X_train - x, axis=1, ord=np.inf)`

### 2. Majority Voting

Predictions are made by finding the most common target label among the `k` nearest neighbors:

`y_hat = argmax_c sum(I(y_i = c)) for i = 1 to k`

Where `I(...)` is an indicator function that returns `1` if neighbor `i` belongs to class `c`, and `0` otherwise.

---

## Experimental Phases

This project evaluates the KNN algorithm through several progressive stages:

### Phase 1: Pure KNN

The raw algorithm is applied directly to the dataset. This serves as a baseline, demonstrating how the basic Euclidean distance and majority voting perform on unmodified data.

### Phase 2: KNN + Standard Scaler

Distance-based algorithms are highly sensitive to the scale of features. In this phase, a `StandardScaler` is applied to transform features such that they have a mean of `0` and a standard deviation of `1`:

`z = (x - μ) / σ`

This ensures that features with larger numerical ranges do not artificially dominate the distance calculations.

### Phase 3: KNN + Feature Selection

KNN is susceptible to the "curse of dimensionality." As the number of features increases, the distance between any two points becomes increasingly similar, degrading the model's predictive power. This phase introduces a feature selection step to isolate the most critical predictors, followed by an evaluation of Accuracy vs. `K` on the reduced feature set.

### Phase 4: Noise Injection & Robustness Testing

To test the resilience of the KNN algorithm, a custom `inject_noise` function was implemented to simulate real-world data corruption at varying rates (e.g., `10%`, `20%`):

1. **Label Flipping:** Randomly selecting a percentage of the targets and flipping their binary classes (`1 - y`)
2. **Feature Noise:** Adding Gaussian noise to the feature vectors:  
   `X_noisy = X + N(0, noise_rate)`

**Key Finding:** The experiments demonstrate that a higher value of `K` (for example, `k = 16`) is significantly more robust to noise than a lower value (`k = 1`). For instance, at a `10%` noise rate, `k = 1` accuracy dropped heavily, while `k = 16` maintained high accuracy (approximately `95%`). This shows that relying on a broader neighborhood helps smooth out anomalous data points.
