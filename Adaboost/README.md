# AdaBoost vs. Random Forest

## Overview
This project focuses on the implementation and comparative analysis of two powerful ensemble learning techniques: **AdaBoost** and **Random Forest**. The core objective is to build an AdaBoost classifier from scratch using decision stumps/trees based on a weighted Gini index, and then benchmark its performance and scaling behavior against a Random Forest model.

## Features
- Custom **AdaBoost** implementation from scratch.
- Integration of **sample weights** into a custom Decision Tree using a **Weighted Gini Index**.
- Early stopping mechanism (patience) implemented for AdaBoost to prevent overfitting.
- Extensive hyperparameter tuning: varying `max_depth` and `n_estimators`.
- Comparative visualizations mapping Accuracy against Number of Trees and Tree Depth.

## Mathematical Foundation: Weighted Gini Index
In standard decision trees, the Gini impurity measures how often a randomly chosen element would be incorrectly labeled. For AdaBoost, the Gini calculation must account for the dynamically updating sample weights.

The weighted Gini impurity for a node is implemented as:
`$$ Gini = 1 - \sum_{c} p_c^2 $$`
Where $p_c$ is the weighted proportion of class `$c$`:
`$$ p_c = \frac{\sum_{i \in c} w_i}{\sum w_{total}} $$`
The best split is chosen by minimizing this weighted Gini index across candidate thresholds.

## Experiments & Hyperparameter Sweep
The notebook runs a comprehensive grid search over two key dimensions for both models:
1. **Number of Estimators (`n_estimators`)**: Evaluating how the ensemble size impacts training and validation accuracy.
2. **Tree Depth (`max_depth`)**: Evaluating how the complexity of the base learners affects the final ensemble's performance.

### Model Evaluation Metrics
Both models are evaluated on a holdout test set using the following metrics (calculated via custom functions):
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

## Results
The hyperparameter sweep identifies the optimal configuration for the AdaBoost classifier, validating it against early stopping criteria. 

*Sample Output for the Best Model:*
- **Accuracy:** 0.7742 (with validation reaching ~0.8033 during search)
- **Precision:** 0.8462
- **Recall:** 0.6875

Comparative plots generated in the notebook clearly illustrate the trade-offs:
- **Accuracy vs. Number of Trees:** Shows how AdaBoost and Random Forest plateau or overfit as the ensemble grows.
- **Accuracy vs. Tree Depth:** Highlights the differing base-learner requirements (AdaBoost typically favors shallow stumps, while Random Forest utilizes deeper, fully grown trees).
