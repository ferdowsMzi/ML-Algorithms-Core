# Bayesian Network Structure Learning with K2 Algorithm

This project implements a complete pipeline for learning the structure of a Bayesian Network from data using the **K2 Algorithm**. Built from scratch without relying on high-level probabilistic graphical model libraries, this repository focuses on greedy structure search, K2 scoring, and model evaluation using Average Log-Likelihood.

The project is applied to the famous **ASIA dataset**, a synthetic dataset often used in Bayesian network research to model diagnostic probabilities.

## Dataset & Preprocessing

The model uses the `asia.csv` dataset, which consists of 5,000 samples and 8 binary variables: `['A', 'S', 'T', 'L', 'B', 'E', 'X', 'D']` (*`e04.ipynb`, lines 23–37*).

Since the K2 algorithm relies on discrete frequency counts, the preprocessing step converts the categorical string values into a binary numeric format (*`e04.ipynb`, lines 76–81*):

- `'yes' -> 1`
- `'no' -> 0`

Here is the real structure:

<img width="527" height="576" alt="ASIA network structure" src="https://github.com/user-attachments/assets/0bbbbc46-66c7-486e-91da-4de24ffe8286" />

## Mathematical Foundations

### 1. The K2 Score

To evaluate how well a set of parent nodes explains a child node, the project implements the log-space K2 scoring metric (*`e04.ipynb`, lines 138–175*). Since the variables are binary, the number of possible values for a node is `r = 2`.

- **K2 Score:** Computes the score of a node given a candidate parent set using frequency counts over parent configurations.  
  `Score = sum_j [ log((r - 1)!) - log((N_ij + r - 1)!) + sum_k log(N_ijk!) ]`

*Implementation Note:* To optimize the search process, K2 scores for node-parent combinations are dynamically stored in a `score_cache` dictionary, which significantly speeds up the greedy search.

### 2. Average Log-Likelihood

To compare different learned network structures, the model's overall fit is evaluated on validation data using the Average Log-Likelihood (*`e04.ipynb`, lines 185–199*). This is approximated by summing the K2 scores for all nodes under the candidate graph structure and normalizing by the dataset size `N`.

- **Average Log-Likelihood Approximation:**  
  `Avg LL ≈ (1 / N) * sum_over_nodes K2Score(node, Parents(node), data)`

## Structure Search Strategy

Finding the optimal Directed Acyclic Graph (DAG) is computationally expensive. This implementation uses a constrained **greedy K2-style search** (*`e04.ipynb`, lines 327–390*):

1. **Node Ordering:** The algorithm generates 5 random topological orderings of the nodes to explore different network spaces.
2. **Parent Capacity Bound:** The search iterates through maximum allowed parents using `max_parents = [1, 2, 3, 4]`.
3. **Greedy Addition:** For each node in a given order, the algorithm considers its predecessors. It iteratively adds the parent that yields the highest localized K2 score improvement. If adding a parent decreases the score, the addition stops.

## Experiments & Metrics

The project evaluates candidate networks by plotting **Validation Log-Likelihood vs. Max Parents** to determine the optimal network complexity without overfitting.

A custom evaluation suite is also included to test downstream classification performance (*`e04.ipynb`, lines 394–423*), implementing standard metrics from scratch:

- **Precision:** Measures the fraction of predicted positives that are correct.  
  `Precision = TP / (TP + FP + epsilon)`

- **Recall:** Measures the fraction of actual positives that are correctly identified.  
  `Recall = TP / (TP + FN + epsilon)`

- **F1-Score:** Balances precision and recall using their harmonic mean.  
  `F1-Score = 2 * (Precision * Recall) / (Precision + Recall + epsilon)`
