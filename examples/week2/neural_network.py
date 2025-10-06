import pandas as pd
from sklearn.neural_network import MLPClassifier
import numpy as np
data = pd.read_csv('heart_disease.csv')
X = data.drop('target', axis=1)
y = data['target']
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
     hidden_layer_sizes=(15, 13),
     random_state=1, max_iter=1000)
clf.fit(X, y)
girdi=np.array([14,0,1,88,21,0,1,
               92,0,0.7,2,0,2]).reshape(1,-1)
beklenen = clf.predict(girdi)
print("Beklenen :", beklenen)
