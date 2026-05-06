# Neural Network Implementation from Scratch

## Overview
This project involves building a fully connected Neural Network from scratch using Python and NumPy. The primary objective is to understand the mathematical foundations of deep learning by implementing forward propagation, backpropagation, and weight updates without relying on high-level frameworks like TensorFlow or PyTorch.

## Features
- Complete Neural Network implementation from scratch.
- Custom dataset preprocessing class.
- Support for multiple activation functions: **ReLU**, **Tanh**, and **Softmax** (for output layer probabilities).
- Configurable network architectures (number of layers and neurons).

## Dataset
- **MNIST Dataset**: The standard benchmark dataset containing handwritten digits. The $28 \times 28$ pixel grayscale images are flattened into a 784-dimensional input vector.

## Model Architectures & Results
The notebook evaluates four different network configurations to compare the effects of width, depth, and activation functions. 

| Model | Architecture (Neurons per Layer) | Activation (Hidden) | Test Accuracy |
| :--- | :--- | :--- | :--- |
| **Model 1** | Input: 784 &rarr; Hidden: 256 &rarr; Output: 10 | ReLU | 95% |
| **Model 2** | Input: 784 &rarr; Hidden: 512 &rarr; Output: 10 | ReLU | 95% |
| **Model 3** | Input: 784 &rarr; Hidden: 256 &rarr; Hidden: 128 &rarr; Output: 10 | Tanh | **95.13%** |
| **Model 4** | Input: 784 &rarr; Hidden: 512 &rarr; Hidden: 256 &rarr; Output: 10 | Tanh | **95.49%** |

## Implementation Details
1. **Mathematical Foundation:**
   - **Linear Step:** `$Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}$`
   - **Activation Step:** `$A^{[l]} = g^{[l]}(Z^{[l]})$`
2. **Backpropagation:** Uses the chain rule to compute gradients `$dW$` and `$db$` for optimization.
3. **Loss Function:** Cross-entropy loss is used alongside the Softmax activation in the final layer for multi-class classification.
