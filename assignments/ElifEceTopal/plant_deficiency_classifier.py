## Example: Predicting nutrient deficiency (e.g., Nitrogen deficiency) in plant leaves
## This example uses a Random Forest classifier with simulated leaf chemical data.

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --- Simulated leaf chemical measurements ---
# 120 samples × 4 features (e.g., chlorophyll level, moisture, pH, nitrate content)
X = np.random.rand(120, 4)

# Labels: 0 = healthy leaf, 1 = nitrogen deficiency
y = np.random.randint(0, 2, 120)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

# Train model
model = RandomForestClassifier(n_estimators=60, random_state=10)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Plot predictions (using feature 1 vs feature 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
plt.title("Predicted Leaf Nutrient Status")
plt.xlabel("Chlorophyll Level")
plt.ylabel("Moisture Level")
plt.colorbar(label='Prediction (0 = Healthy, 1 = Deficient)')
plt.show()
