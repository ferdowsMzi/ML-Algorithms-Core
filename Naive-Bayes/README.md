# Emotion Extraction & Text Classification with Naive Bayes

This repository contains a full pipeline for Natural Language Processing (NLP) and text classification, built entirely from scratch. The primary goal of this project is to extract and classify emotions from textual data using a custom implementation of the **Multinomial Naive Bayes** algorithm.

This project explores the raw mechanics of text preprocessing, Bag-of-Words (BoW) vectorization, probability calculations, and the impact of hyperparameter tuning — especially Laplace smoothing and vocabulary size — on model performance.

## Text Preprocessing Pipeline

Raw text data is messy and requires normalization before it can be fed into a statistical model. The custom `clean_text` function (*`e3.ipynb`, lines 149–170*) applies a rigorous preprocessing pipeline:

- **Lowercasing:** Standardizes all text to lowercase.
- **HTML Tag Removal:** Strips out web-scraped artifacts using regex (`<.*?>`).
- **Punctuation / Noise Removal:** Filters out all non-alphabetic characters.
- **Stop Words Removal:** Discards common, low-information words.
- **Light Stemming:** Reduces words to their base forms by stripping common English suffixes such as `-ing`, `-ed`, and `-s`.

## Mathematical Foundations: Naive Bayes & Smoothing

The core algorithm is a probabilistic classifier based on Bayes' Theorem. To prevent numerical underflow when multiplying many small probabilities, the entire algorithm is implemented in **log-space**.

### 1. Class Prior Probability

- **Class Prior:** Represents the baseline frequency of each emotion class `c` in the training set.  
  `P(c) = count(X_c) / N_samples`

### 2. Likelihood & Laplace (Alpha) Smoothing

To calculate the probability of a word `w` occurring given a specific class `c`, the model counts word frequencies within each class. However, if a word in the test set never appeared in the training data for a given class, its probability becomes zero, which breaks the full probability product.

- **Laplace-Smoothed Likelihood:** Solves the zero-probability problem by adding a smoothing parameter `alpha`.  
  `P(w | c) = (count(w, c) + alpha) / (count(c) + alpha * V)`  
  *Where `V` is the total number of features or words in the vocabulary.*

### 3. Log-Probability Prediction

For a new text sample represented by a feature vector `x`, the model computes a score for each class and predicts the label with the highest score (*`e3.ipynb`, lines 232–268*).

- **Log-Space Prediction Rule:**  
  `log P(c | x) ∝ log P(c) + sum(x_i * log P(w_i | c))`

## Implementation Details

The `Naive_Bayes` class stores the computed probabilities during training (`fit`) and uses them during inference (`predict`):

- `class_log_prior`: Stores `log P(c)` for each emotion class.
- `feature_log_prob`: Stores the smoothed `log P(w | c)` matrix.
- Uses `numpy` matrix dot products such as `np.dot(x, self.feature_log_prob[c])` for efficient and vectorized log-probability computation.

## Experiments & Analysis

The project evaluates the model using the **F1-score** and conducts two major experiments to understand model behavior.

### Experiment 1: Impact of Feature Vector Length

Does a larger vocabulary always improve performance? The model builds vocabularies of varying maximum sizes (`100`, `500`, `1000`, `2000`, `5000`) and evaluates the resulting F1-score (*`e3.ipynb`, lines 337–361*).

- *Example Result:* Increasing the vocabulary size from `100` to `500` words produced a notable improvement in F1-score, from approximately `0.7150` to `0.8218`.

### Experiment 2: Laplace Smoothing Optimization

The model is tested across a range of alpha values (`[0.001, 0.1, 1.0, 5.0, 10.0, 50.0]`) to find the best balance between preventing zero-probabilities and preserving the learned distributions (*`e3.ipynb`, lines 371–396*).

- A plot of **F1-Score vs. Laplace Alpha** is generated to visualize how aggressive smoothing affects predictive accuracy.
