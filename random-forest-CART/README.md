# Random Forest & CART Decision Tree from Scratch

This project implements a **Random Forest** ensemble for binary classification entirely from scratch. It builds upon a custom implementation of the **CART (Classification and Regression Trees)** algorithm, combining multiple decision trees using bootstrap aggregating (bagging) and majority voting.

## Core Components

### 1. CART Decision Tree
The base learner is a Decision Tree that builds a binary tree through recursive partitioning. 
*   **Splitting Criterion:** The tree evaluates potential splits by scanning thresholds and maximizing **Information Gain** based on Gini Impurity reduction.
*   **Stopping Criteria:** The recursive growth stops when the maximum depth (`max_depth=100`) is reached, a node is pure (only one label remains), or the number of samples is less than `min_samples_split`.

### 2. Random Forest Ensemble
The forest combines multiple independent decision trees to reduce variance and prevent overfitting.
*   **Bootstrap Resampling:** For each tree, a random subset of the training data is drawn with replacement (bagging).
*   **Majority Vote:** During prediction, each tree casts a "vote" for the class label. The forest aggregates these predictions and outputs the most common label across all trees.

## Mathematical Foundations

### Gini Impurity
The impurity of a node is calculated using the Gini Index:
`$$ Gini(y) = 1 - \sum_{i} p_i^2 $$`
*(Where `$p_i$` represents the proportion of samples belonging to class `$i$` in a particular node).*

### Information Gain (Gini Reduction)
To find the optimal split, the algorithm calculates the reduction in Gini impurity from the parent node to the weighted child nodes:
`$$ IG = Gini(parent) - \left( \frac{n_{left}}{n} Gini(left) + \frac{n_{right}}{n} Gini(right) \right) $$`
*(Where $n$ is the total number of samples in the parent node, and `$n_{left}, n_{right}$` are the number of samples in the child nodes).*

## Evaluation Metrics Implemented

Standard classification metrics are implemented from scratch to evaluate the model's performance:
*   **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$
*   **Precision:** $\frac{TP}{TP + FP}$
*   **Recall:** $\frac{TP}{TP + FN}$
*   **F1-Score:** $2 \times \frac{Precision \times Recall}{Precision + Recall}$
*   **Confusion Matrix**

## Results & Usage

The notebook evaluates a baseline **Single Decision Tree**, achieving an accuracy of **~77.42%** on the test set. 

The full Random Forest model expands on this baseline. You can configure the ensemble using hyperparameters such as:
*   `n_estimators`: Number of trees in the forest (default: 10).
*   `max_depth`: Maximum depth of each individual tree.
*   `min_samples_split`: Minimum samples required to split an internal node.
*   `n_features`: Number of features to consider when looking for the best split.
