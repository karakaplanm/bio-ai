
# 🔬 Microorganism Classification (using Transfer Learning)

This project uses Deep Learning to identify and classify 8 different types of microorganisms from microscope images.

The model is built with TensorFlow and Keras. It uses **Transfer Learning** with the `MobileNetV2` architecture, which is a powerful method for image classification.

## 🎯 Project Goal

The main goal is to create a model that can automatically look at an image of a microorganism (like an Amoeba or Bacteria) and correctly guess what it is. This can help speed up work in a lab by automating the identification process.

## 📁 The Dataset

The model was trained to identify 8 different classes of microorganisms:

1.  Amoeba
    
2.  Euglena
    
3.  Hydra
    
4.  Paramecium
    
5.  Rod Bacteria
    
6.  Spherical Bacteria
    
7.  Spiral Bacteria
    
8.  Yeast
    

The image data is organized into `data/train` (for training) and `data/val` (for validation) folders. The `ImageDataGenerator` in Keras uses this structure to load the images.

## 💻 Technologies Used

-   **Python 3.x**
    
-   **TensorFlow** and **Keras** (for building and training the model)
    
-   **MobileNetV2** (The pre-trained base model)
    
-   **NumPy** (for handling data)
    
-   **Matplotlib** (for plotting the training results)
    

## 🧠 How the Model Works (Methodology)

We trained the model in two main steps. This is a common and effective strategy.

1.  **Feature Extraction:**
    
    -   We loaded the `MobileNetV2` model, which was already trained on millions of general images (the ImageNet dataset).
        
    -   We "froze" the layers of this base model. This means we didn't let them change.
        
    -   We added our own new layers at the end: one `Dense` layer and one `softmax` (output) layer for our 8 classes.
        
    -   We trained _only_ these new layers. This lets the model learn to connect the features MobileNetV2 knows (like shapes and textures) to our specific microbe classes.
        
2.  **Fine-Tuning:**
    
    -   After the first step, we "un-froze" the top 30 layers of the `MobileNetV2` base model.
        
    -   We set a **very low learning rate** (like `1e-5`). This is important so we don't destroy the knowledge the model already has.
        
    -   We trained the model for a few more epochs. This step allows the model to slightly adjust its existing knowledge to better fit the details of _our_ microorganism images.
        

----------

## 🚀 How to Use This Project

### 1. Setup

You need to install the required libraries first:

Bash

```
pip install tensorflow numpy matplotlib

```

### 2. Training the Model

1.  Place your image files in the `data/train` and `data/val` folders. (See the [Folder Structure](https://www.google.com/search?q=%23-folder-structure) section below).
    
2.  Run the main training script:
    
    Bash
    
    ```
    python bacteria_classification.py
    
    ```
    
3.  When training is finished, the model will be saved as `bacteria_colony_model.h5`.
    

### 3. Making a Prediction

You can use the saved `.h5` model to predict a new, single image.

1.  Open the `predict.py` (or `ghg.py`) script.
    
2.  Change the `IMG_PATH` variable to the full path of your test image.
    
3.  Make sure the `CLASS_NAMES` list in the script is in the correct alphabetical order (matching the training data).
    
4.  Run the prediction script:
    
    Bash
    
    ```
    python predict.py
    
    ```
    

**Example Output:**

```
--- PREDICTION RESULT ---
Image: my_test_image.jpg
Predicted Class: Amoeba
Confidence: 98.72%

```

## 📂 Folder Structure

The project needs this folder structure to work correctly with `flow_from_directory`:

```
microbe-classification/
├── data/
│   ├── train/
│   │   ├── Amoeba/
│   │   │   ├── img1.jpg
│   │   │   └── ...
│   │   ├── Euglena/
│   │   │   └── ...
│   │   └── (and the other 6 classes)
│   └── val/
│       ├── Amoeba/
│       │   └── ...
│       ├── Euglena/
│       │   └── ...
│       └── (and the other 6 classes)
│
├── bacteria_classification.py   (Main script to train the model)
├── predict.py                   (Script to predict a new image)
├── bacteria_colony_model.h5     (The final, saved model)
└── README.md                    (This file)
```
