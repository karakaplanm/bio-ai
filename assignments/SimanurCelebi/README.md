Project Name: DNA Sequence Feature Recognition (Using LSTM)

Deep Learning – LSTM-Based DNA Sequence Classification Project

1. Project Goal

The primary goal of this project is to accurately classify the biological feature of a given DNA sequence (Gene Start, Intron, or Exon) using deep learning.

To achieve this, an LSTM (Long Short-Term Memory) model—well-known for its ability to capture long-term dependencies in sequential data—has been applied.

2. Project Components and Technologies Used

| Component               | Purpose                                                      | Technologies Used          |
| ----------------------- | ------------------------------------------------------------ | -------------------------- |
| **Model Type**          | Learn dependencies within sequential DNA data                | LSTM Layer                 |
| **Input Encoding**      | Convert nucleotides into numerical form and embedded vectors | Embedding Layer            |
| **Framework**           | Build, train, and evaluate the deep learning model           | TensorFlow / Keras         |
| **Data Processing**     | Sequence conversion, numerical encoding, one-hot encoding    | NumPy, to_categorical      |
| **Classification Type** | Distinguish between 3 or more classes                        | Multi-class Classification |


3. Dataset and Preprocessing
3.1 Dataset Characteristics

Input (X): DNA sequences with a fixed length of 20 nucleotides

Vocabulary Size: 4 (A, C, G, T)

Output Classes (Y):

0: Gene Start

1: Intron

2: Exon

3.2 Preprocessing Steps
✔ Numerical Encoding

Each nucleotide is mapped to an integer:

A → 0

C → 1

G → 2

T → 3

✔ Sequence Preparation

All sequences are adjusted to SEQ_LENGTH = 20, either by truncating or padding.
In this project, truncation is used.

✔ One-Hot Encoding

Output labels are converted into one-hot vectors using to_categorical.

Example:
Class 2 → [0, 0, 1]

4. Model Architecture and Configuration

The model is built using the Keras Sequential API, optimized for sequential biological data.

4.1 Layers

| Layer              | Parameters                                 | Function                                        |
| ------------------ | ------------------------------------------ | ----------------------------------------------- |
| **Embedding**      | input_dim=4, output_dim=8, input_length=20 | Transforms DNA bases into 8-dimensional vectors |
| **LSTM**           | units=32                                   | Learns temporal and sequential dependencies     |
| **Dense (hidden)** | units=16, activation='relu'                | Enhances abstract feature representation        |
| **Dense (output)** | units=3, activation='softmax'              | Computes class probabilities for 3 categories   |

4.2 Compilation Settings

Optimizer: adam

Loss Function: categorical_crossentropy

Metric: accuracy

5. Project Development and Testing Phases
✓ Data Collection & Labeling

Real biological DNA sequences are collected and manually labeled as Gene Start, Intron, or Exon.

✓ Train–Validation Split

The dataset is split into training and validation sets.

✓ Model Training

The model is trained using model.fit() for 50 epochs.

✓ Evaluation

Accuracy and loss values are monitored to optimize performance.

✓ Prediction

A new DNA sequence is fed into the trained model to predict its class.

Example Prediction Output

| Test Sequence        | Predicted Probabilities | Predicted Class    |
| -------------------- | ----------------------- | ------------------ |
| ACGTACGTACGTACGTACGT | [0.92, 0.05, 0.03]      | **0 – Gene Start** |
