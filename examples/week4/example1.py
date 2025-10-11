
## 1️⃣ Example: Classifying Cancer Samples from Gene Expression Data

## This example uses **scikit-learn** to predict whether a patient has cancer based on a few gene expression values.

# Example 1: Simple cancer classification
# This example uses a Random Forest classifier to predict cancer status based on simulated gene expression data.

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Simulated gene expression data
# 100 samples × 5 genes
X = np.random.rand(100, 5)
# Labels: 0 = healthy, 1 = cancer
y = np.random.randint(0, 2, 100)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Plot the test set predictions
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis')
plt.title("Random Forest Predictions (Test Set)")
plt.xlabel("Gene 1 Expression")
plt.ylabel("Gene 2 Expression")
plt.colorbar(label='Prediction (0=Healthy, 1=Cancer)')
plt.show()