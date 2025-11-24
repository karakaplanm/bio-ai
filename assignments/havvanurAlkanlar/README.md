# 🧬 Bacterial Colony Classification  
A simple Convolutional Neural Network (CNN) model for classifying bacterial colonies on Petri dish images.

This project demonstrates a complete machine learning workflow:

- data → preprocessing → model → training → visualization → prediction  
- synthetic dataset generator  
- heatmap (model interpretability)  
- RGB channel extraction  
- simple CNN architecture (PyTorch)

## 🔬 Biological Background

Different bacterial species form characteristic colony morphologies:

| Species | Colony Color | Shape | Edge |
|--------|--------------|--------|-------|
| *Staphylococcus aureus* | Yellow | Round | Smooth |
| *Escherichia coli* | White | Moist | Soft |
| *Pseudomonas aeruginosa* | Green | Spread | Irregular |

## 🧠 Model Architecture

- 2× Conv2D + ReLU + MaxPool  
- FC layer (256 units)  
- Output: 3 classes  

Input: 128×128 RGB
