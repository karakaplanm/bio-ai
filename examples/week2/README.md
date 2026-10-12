
# Week 2

This week introduces supervised classification with scikit-learn on two small tabular datasets: the classic Iris flower measurements and a heart disease dataset.

## Files

| File | Description |
|:--|:--|
| `iris_flower_classification.py` | Full Iris workflow: data exploration, plots, label encoding, and a comparison of seven classifiers before and after hyperparameter tuning. |
| `Iris_Flower_Classification.ipynb` | The same workflow as a notebook, with the charts and explanations inline. |
| `Iris.csv` | 150 flowers, 4 measurements (sepal/petal length and width, in cm), 3 species. |
| `neural_network.py` | A minimal multi-layer perceptron (`MLPClassifier`) trained on the heart disease data, then used to predict one new patient. |
| `heart_disease.csv` | 303 patients, 13 clinical features, and a binary `target` column. |

## Setup

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost tabulate --break-system-packages
```

`tabulate` is needed because the Iris script prints its result tables with `DataFrame.to_markdown()`.

## Run

Run the scripts from this folder, since they read the CSV files by relative path.

```bash
cd examples/week2

# Iris classification (opens several chart windows; close each one to continue)
python3 iris_flower_classification.py

# Heart disease neural network
python3 neural_network.py
```

To run the Iris script without chart windows, for example over SSH:

```bash
MPLBACKEND=Agg python3 iris_flower_classification.py
```

## What the Iris script does

1. Loads `Iris.csv` and checks its shape, types, duplicates, and missing values.
2. Plots feature distributions, pairwise scatter plots by species, and a correlation heatmap.
3. Encodes the species names as the labels 0, 1, and 2.
4. Splits the data into 70% training and 30% test.
5. Trains and evaluates each model twice, once with default settings and once tuned by grid or randomized search:
   Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, Naive Bayes, and a Neural Network.
6. Prints a comparison table of precision, recall, accuracy, and F1 on the training and test sets.
7. Predicts the species of one sample flower with the tuned Random Forest.

All models reach roughly 91–96% accuracy on the test set. Exact numbers can vary slightly between runs and library versions.

## What the neural network script does

It trains an MLP with two hidden layers (15 and 13 neurons) on all 303 patients and prints the predicted class for one hand-written patient record (`Beklenen : [1]`). There is no train/test split, so it shows how to fit and predict, not how well the model generalizes.

## Troubleshooting

- **`ValueError: could not convert string to float: 'Iris-setosa'`** — pandas 2.0 and later no longer skip text columns in `DataFrame.corr()`. The script uses `data.corr(numeric_only=True)` for this reason; if you see the error, you are running an older copy.
- **`ImportError: Missing optional dependency 'tabulate'`** — run `pip install tabulate --break-system-packages`.
- **`FileNotFoundError: Iris.csv`** — run the script from inside `examples/week2`.
