# Lecture 3: Deep Learning Basics  

**Date:** 2025-10-06

---

## 1. Introduction
- Deep learning is a subset of machine learning that uses **artificial neural networks** with multiple layers.  
- It has revolutionized applications in **image analysis, genomics, drug discovery, and medical diagnostics**.  
- Although inspired by biology, modern deep learning models are **mathematical and computational constructs**.  

---

## Neural Networks: Structure and Intuition 

###  Biological Inspiration  
- The term “neural network” comes from the analogy to **biological neurons**.  
- Each artificial neuron is a **computational unit** that sums weighted inputs and passes them through an **activation function**.  

### Perceptron  
- Proposed by **Frank Rosenblatt (1958)**.  
- Components:  
  - **Inputs (x₁, x₂, …, xn)**  
  - **Weights (w₁, w₂, …, wn)**  
  - **Summation and bias (Σ wᵢxᵢ + b)**  
  - **Activation function (e.g., step, sigmoid, ReLU)**  

###  Multi-Layer Perceptron (MLP)  
- Consists of:  
  - **Input layer**: raw features  
  - **Hidden layers**: feature transformations  
  - **Output layer**: prediction or classification  

### Common Activation Functions  
- **Sigmoid**: good for probabilities but suffers from vanishing gradients  
- **tanh**: centered at zero, smoother gradients  
- **ReLU**: efficient, avoids vanishing gradient but may suffer from dead neurons  
- **Softmax**: converts outputs into probabilities  

---

## Training Neural Networks

### Forward Propagation  
- Inputs are multiplied by weights, biases are added, activation functions applied → **output is produced**.  

### Loss Functions  
- Measure the error between prediction and target.  
- Examples:  
  - Mean Squared Error (MSE) for regression  
  - Cross-Entropy for classification  

### Backpropagation  
- Central algorithm for training.  
- Steps:  
  1. Compute error at output.  
  2. Propagate error backward through the network.  
  3. Update weights using **gradient descent**.  

### Gradient Descent Variants  
- **Batch GD**: entire dataset  
- **Stochastic GD (SGD)**: one sample at a time  
- **Mini-batch GD**: compromise, most commonly used  
- Optimizers: **Adam, RMSProp, Adagrad**  

---

## Deep Learning Frameworks

### TensorFlow  
- Developed by Google.  
- Computational graph-based.  
- Widely used in both research and production.  

### PyTorch  
- Developed by Facebook (Meta).  
- Dynamic computational graphs → more intuitive debugging.  
- Extremely popular in research.  

### Keras  
- High-level API (originally independent, now integrated with TensorFlow).  
- User-friendly, ideal for prototyping.  

### Comparison  
| Framework | Strengths | Weaknesses |  
|-----------|-----------|------------|  
| TensorFlow | Production, scalability | Steeper learning curve |  
| PyTorch    | Flexibility, research | Slightly less optimized for deployment |  
| Keras      | Simplicity | Less control for advanced users |  

---

## Applications in Biology

- **Genomics**: predicting gene expression, variant classification  
- **Proteomics**: protein structure prediction (e.g., AlphaFold)  
- **Medical Imaging**: tumor detection in MRI, X-ray classification  
- **Drug Discovery**: virtual screening, molecular property prediction  
- **Systems Biology**: modeling biological networks  

---

## Practical Notes and Challenges

- **Overfitting**:  
  - Networks memorize instead of generalizing.  
  - Solutions: dropout, data augmentation, regularization.  

- **Vanishing/Exploding Gradients**:  
  - Mitigation: better activations (ReLU), careful initialization, batch normalization.  

- **Computational Costs**:  
  - Requires GPUs/TPUs.  
  - High memory consumption.  

- **Interpretability**:  
  - Deep models often act as “black boxes.”  
  - Explainable AI (XAI) methods are needed in biomedical contexts.  

---

## Summary

- Neural networks are mathematical models inspired by biology.  
- Training relies on **backpropagation and gradient descent**.  
- Frameworks like TensorFlow and PyTorch enable scalable experimentation.  
- Applications in biology are vast but face challenges in **interpretability and data quality**.  

---

## Suggested Readings  

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.  
2. LeCun, Y., Bengio, Y., & Hinton, G. (2015). *Deep Learning*, Nature, 521, 436–444.  
3. Schmidhuber, J. (2015). *Deep learning in neural networks: An overview*, Neural Networks, 61, 85–117.  
4. Min, S., Lee, B., & Yoon, S. (2017). *Deep learning in bioinformatics*, Briefings in Bioinformatics, 18(5), 851–869.  

---

## Discussion Questions / Exercises  

1. Why is backpropagation considered the backbone of deep learning?  
2. Compare biological neurons with artificial neurons – what are the key similarities and differences?  
3. Implement a simple neural network using either TensorFlow or PyTorch to classify a small dataset (e.g., MNIST).  
4. In a biomedical application (e.g., cancer diagnosis), what risks do we face when applying deep learning models blindly?  

