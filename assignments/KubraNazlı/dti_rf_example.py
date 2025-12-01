# Example: Drug-Target Interaction Prediction (Random Forest)
# AI for Drug Discovery: Predict if a compound interacts with a target protein
# This example uses RandomForestClassifier with different parameters.

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Simulated compound–protein interaction features
# 100 compounds, each with 15 features (e.g., descriptors, fingerprints, docking scores)
X = np.random.rand(100, 15)
y = np.random.randint(0, 2, 100)   # 1 = interaction, 0 = no interaction

# Random Forest model with modified parameters
model = RandomForestClassifier(
    n_estimators=150,     # number of trees
    max_depth=6,          # limit depth to avoid overfitting
    min_samples_split=4,  # minimum samples needed to split
    random_state=42
)

model.fit(X, y)

# New compound with 15 features
new_compound = np.random.rand(1, 15)
prob = model.predict_proba(new_compound)[0, 1]

print("Predicted binding probability:", prob)
