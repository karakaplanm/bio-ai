
# 🧬 Gene Expression Analysis Using Autoencoders  
### A Biological Artificial Intelligence Application

## 1. Introduction
High-throughput omics technologies such as RNA-sequencing generate large-scale gene expression datasets with thousands of features per sample. The high dimensionality of such data introduces challenges including noise, redundancy, and computational complexity.  
Deep learning-based representation learning methods offer effective solutions for extracting meaningful patterns from biological data.

## 2. Aim of the Project
The aim of this project is to apply an unsupervised deep learning approach to gene expression data in order to:
- Perform dimensionality reduction
- Learn compact latent representations
- Reconstruct gene expression profiles with minimal information loss

## 3. Methodology
An Autoencoder neural network consisting of encoder and decoder components was implemented using TensorFlow and Keras.  
The model was trained using Mean Squared Error loss and the Adam optimizer.

## 4. Dataset
Synthetic gene expression data was generated for demonstration purposes:
- 500 samples
- 1000 genes per sample
Values were normalized between 0 and 1.

## 5. Technologies Used
- Python 3.x
- TensorFlow & Keras
- NumPy
- Matplotlib

## 6. Project Structure
gene_expression_autoencoder/
├── README.md
├── train_autoencoder.py
├── visualize.py

## 7. Usage
pip install tensorflow numpy matplotlib  
python train_autoencoder.py  
python visualize.py  

## 8. Conclusion
This project demonstrates the effectiveness of autoencoders for representation learning in biological data analysis and highlights the role of artificial intelligence in computational biology.
