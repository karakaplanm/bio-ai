# 🍄 Fungal Colony Classification  
A simple Convolutional Neural Network (CNN) model for classifying fungal colonies on Petri dish images.

This project demonstrates a complete machine learning workflow:

- data → preprocessing → model → training → visualization → prediction  
- synthetic dataset generator  
- heatmap (model interpretability)  
- RGB channel extraction  
- simple CNN architecture (PyTorch)

## 🔬 Biological Background

Different fungal species form characteristic colony morphologies:

| Species | Colony Color | Shape | Edge |
|--------|--------------|--------|-------|
| *Aspergillus niger* | Black | Round | Smooth |
| *Candida albicans* | White | Moist | Soft |
| *Penicillium* | Green | Spread | Irregular |

## 🧠 Model Architecture

- 2× Conv2D + ReLU + MaxPool  
- Fully Connected (256 units)  
- Output: 3 classes  
Input: 128×128 RGB
