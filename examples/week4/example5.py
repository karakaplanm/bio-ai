# Example 5: Drug-target interaction prediction
# AI for Drug Discovery: Predict if a compound binds to a target protein using simulated features.
# This example uses a Support Vector Machine (SVM) for classification.

import numpy as np
from sklearn.svm import SVC

# Simulated compound features (molecular descriptors)
X = np.random.rand(50, 10)
y = np.random.randint(0, 2, 50)  # 1 = binding, 0 = no binding

model = SVC(kernel='rbf', probability=True)
model.fit(X, y)

new_compound = np.random.rand(1, 10)
prob = model.predict_proba(new_compound)[0, 1]
print("Predicted binding probability:", prob)
