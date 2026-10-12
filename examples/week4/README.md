# Week 4

This week applies the machine learning methods from the earlier weeks to biology: five short scripts, each showing one type of model on one type of biological question. All of them use made-up data, so they show how each workflow is put together rather than produce real results.

## Files

| File | Problem | Method | Library |
|:--|:--|:--|:--|
| `example1.py` | Classify samples as healthy or cancer from gene expression values | Random Forest (classification) | scikit-learn |
| `example2.py` | Group samples with similar gene expression | K-Means (clustering) | scikit-learn |
| `example3.py` | Decide whether a DNA sequence contains a promoter region | LSTM neural network (sequence classification) | TensorFlow/Keras |
| `example4.py` | Predict a protein's solubility from numeric features | Linear regression | scikit-learn |
| `example5.py` | Predict whether a compound binds to a target protein | Support Vector Machine (classification) | scikit-learn |

## Setup

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow --break-system-packages
```

Only `example3.py` needs TensorFlow. TensorFlow supports only certain Python versions; if `pip` cannot find a matching release, create a virtual environment with a supported Python version and install the packages there.

## Run

```bash
cd examples/week4

python3 example1.py
python3 example2.py
python3 example3.py
python3 example4.py
python3 example5.py
```

Examples 1 and 2 open a chart window; close it to end the script. To run them without a window, for example over SSH:

```bash
MPLBACKEND=Agg python3 example1.py
```

## The examples

### Example 1: Cancer classification from gene expression

1. Creates 100 samples with 5 gene expression values each, and a random label per sample (0 = healthy, 1 = cancer).
2. Splits the data into 70% training and 30% test.
3. Trains a Random Forest with 50 trees.
4. Prints the accuracy on the test set.
5. Plots the test samples by their first two genes, colored by the predicted class.

The accuracy is around 0.5, the same as guessing. That is the correct result: the labels are random, so there is no relation between the genes and the label for the model to learn. With real expression data, the same code would find whatever signal the data contains.

### Example 2: Clustering gene expression samples

1. Creates 100 samples with 2 gene expression values each.
2. Runs K-Means with 3 clusters. No labels are used; this is unsupervised learning.
3. Plots the samples, colored by cluster.

The data is uniformly random, so there are no natural groups, and K-Means simply cuts the square into three regions. K-Means always returns as many clusters as you ask for, whether or not the data has that structure.

### Example 3: DNA sequence classification

1. Encodes each base as a number: A=0, C=1, G=2, T=3.
2. Builds a dataset of four sequences of 10 bases, labeled 1 (promoter) or 0 (non-promoter).
3. Reshapes the data to the `[samples, timesteps, features]` form that an LSTM expects, here `(4, 10, 1)`.
4. Builds a network with an LSTM layer (16 units), a dense layer (8 neurons, ReLU), and a single sigmoid output.
5. Trains for 20 epochs and prints the prediction for the first sequence.

The output is a number between 0 and 1, the predicted probability that the sequence is a promoter. With four training sequences and 20 epochs it stays near 0.5 (0.48 in a test run with TensorFlow 2.22), although the correct label is 1. Raise `epochs` to see the model fit the four sequences.

### Example 4: Protein solubility

1. Builds a table of five proteins with three features (hydrophobicity, molecular weight, charge) and a measured solubility.
2. Fits a linear regression, which predicts a continuous value rather than a class.
3. Prints one coefficient per feature, then the predicted solubility of a new protein with hydrophobicity 0.4, molecular weight 15, and charge 0.

This is the only example with fixed data, so its output is the same on every run:

```
Coefficients: [ 0.51960784 -0.04034462 -0.07578728]
Predicted solubility: 0.6740641711229948
```

Each coefficient is the change in predicted solubility when that feature increases by one unit and the others stay fixed. Five proteins are far too few to trust these values.

### Example 5: Drug-target interaction

1. Creates 50 compounds with 10 molecular descriptors each, and a random label per compound (1 = binding, 0 = no binding).
2. Trains an SVM with an RBF kernel. `probability=True` makes it report probabilities as well as classes.
3. Creates one new random compound and prints its predicted binding probability.

As in Example 1, the labels are random, so the probability is close to 0.5.

## Notes

- **Different numbers on every run** — examples 1, 2, 3, and 5 generate random data or start from random network weights. For repeatable data, add `np.random.seed(42)` after the imports.
- **`UserWarning: X does not have valid feature names`** (example 4) — harmless. The model was trained on a DataFrame with column names and is asked to predict from a plain list.
- **`UserWarning: Do not pass an input_shape/input_dim argument to a layer`** (example 3) — harmless. Newer Keras versions prefer an `Input(shape=(10, 1))` layer as the first layer of the model.
- **`ModuleNotFoundError`** — see Setup above.
- **TensorFlow GPU errors in example 3** — run it on the CPU, which is fast enough for this model:

  ```bash
  CUDA_VISIBLE_DEVICES="" python3 example3.py
  ```

  See the Week 3 README for details.
