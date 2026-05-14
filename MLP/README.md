# Encoding Strategies & MLP Architectures

This repository contains a PyTorch-based project that explores and compares different categorical feature encoding strategies and neural network architectures for tabular classification tasks. 

The project investigates how data representation (One-Hot, Target Encoding, and Learned Embeddings) and training configurations (Optimizers, Regularization, Learning Rate Schedules) impact the performance of Multi-Layer Perceptrons (MLPs).

## Features

* **Multiple Categorical Encodings**: Evaluates One-Hot Encoding, Target Encoding, and deep Learned Embeddings.
* **Custom PyTorch Models**: Includes a standard MLP and a specialized Embedding-MLP for mixed continuous/categorical data.
* **Optimizer Comparison**: Supports training with `Adam`, `SGD`, and `RMSProp`.
* **Advanced Training Dynamics**: Implements Dropout, Early Stopping, and Learning Rate Decay schedules across different experiment configurations (Configs A, B, C, D).

## Data Preprocessing & Encodings

The notebook processes categorical variables using three distinct pipelines to compare their effectiveness:

1. **One-Hot Encoding (`X_train_ohe`)**: 
   * Uses `pd.get_dummies`.
   * Expands the feature space significantly (e.g., to 66 features).
   * Best for low-cardinality categorical variables.
2. **Target Encoding (`X_train_te`)**: 
   * Replaces categories with target-dependent statistical probabilities.
   * Keeps the feature space compact (e.g., 9 features).
3. **Label Encoding for Embeddings (`X_train_le`)**: 
   * Maps categories to integer IDs.
   * Feeds directly into PyTorch `nn.Embedding` layers to learn dense, semantic representations during training.

## Model Architectures

The project implements two distinct PyTorch `nn.Module` classes:

### 1. `MLP` (Standard)
A standard feed-forward neural network designed for purely numerical, one-hot encoded, or target-encoded inputs.
* Layers: Fully connected `nn.Linear` layers.
* Activations: `nn.ReLU()`.
* Regularization: `nn.Dropout(dropout_rate)`.
* Output: Single-node linear layer `nn.Linear(32, 1)` for binary classification logits.

### 2. `MLPEmbedded`
A specialized architecture for handling categorical variables via learned embeddings.
* **Embedding Layers**: Uses `nn.Embedding(categories, size)` for each categorical feature.
* **Concatenation**: Flattens and concatenates the learned categorical embeddings with the continuous features.
* **Dense Head**: Passes the combined representations through a standard MLP sequence (Linear -> ReLU -> Dropout -> Linear).

## Training Configurations

The training loop is highly configurable and includes several experimental setups to find the best convergence strategy:

* **Optimizers**: Choose between **SGD**, **Adam**, and **RMSProp**.
* **Regularization (Dropout)**: Configurable dropout rates to prevent overfitting.
* **Config C (Early Stopping)**: Halts training if the validation loss (`best_val_loss`) fails to improve for 10 consecutive epochs (`patience_counter`).
* **Config D (Learning Rate Decay)**: Reduces the learning rate dynamically after 5 stagnant epochs, allowing up to 3 total decays to fine-tune the weights as the model converges.
ls.
