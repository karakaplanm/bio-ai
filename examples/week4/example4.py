# Example 4: Predicting protein solubility
# Here we train a model to predict a protein’s solubility from simplified numeric features.

import pandas as pd
from sklearn.linear_model import LinearRegression

# Dummy dataset
data = {
    "hydrophobicity": [0.3, 0.6, 0.1, 0.8, 0.5],
    "molecular_weight": [10, 25, 8, 30, 20],
    "charge": [-1, 1, 0, 1, -1],
    "solubility": [0.9, 0.3, 0.8, 0.2, 0.6],
}
df = pd.DataFrame(data)

X = df[["hydrophobicity", "molecular_weight", "charge"]]
y = df["solubility"]

model = LinearRegression().fit(X, y)

print("Coefficients:", model.coef_)
print("Predicted solubility:", model.predict([[0.4, 15, 0]])[0])
