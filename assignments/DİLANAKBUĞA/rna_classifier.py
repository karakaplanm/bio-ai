# RNA Sequence Classification with Deep Learning
# LSTM-based model to classify sequences as mRNA (1) or miRNA (0)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Encode RNA sequences: A=0, C=1, G=2, U=3
def encode_rna(seq):
    mapping = {'A':0, 'C':1, 'G':2, 'U':3}
    return [mapping[x] for x in seq]

# Dummy RNA dataset
seqs = [
    "AUGCGAUUGC",   # mRNA-like
    "UUUGCACUGA",   # miRNA-like
    "AUGAUGAUGA",   # mRNA-like
    "UGUGUGUGUG"    # miRNA-like
]

# Labels: 1 = mRNA, 0 = miRNA
y = np.array([1, 0, 1, 0])

# Encoding
X = np.array([encode_rna(s) for s in seqs])
X = X.reshape((X.shape[0], X.shape[1], 1))  # LSTM input shape

# Building the model
model = Sequential([
    LSTM(20, input_shape=(10, 1)),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
model.fit(X, y, epochs=25, verbose=0)

# Prediction example
print("Prediction for AUGCGAUUGC:", model.predict(X[:1])[0])
