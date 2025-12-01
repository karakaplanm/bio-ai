"""
Simple Machine Learning Example using RandomForestClassifier
This script demonstrates how to create a simple dataset,
train a RandomForestClassifier on it, and make predictions. 

pip install scikit-learn pandas"""


import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Örnek veri
# X: ifade seviyeleri, y: pathway aktif mi?
X = pd.DataFrame({
    "EGFR": [4.2, 3.1, 8.4, 7.5],
    "KRAS": [6.1, 2.9, 7.8, 8.2],
    "RAF":  [2.1, 1.8, 6.0, 5.7]
})
y = [0, 0, 1, 1]

# 2. Model
clf = RandomForestClassifier()
clf.fit(X, y)

# 3. Predict
print(clf.predict([[5.5, 6.8, 4.0]]))
