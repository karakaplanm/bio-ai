# Example 3: DNA sequence classification
# Example: DNA Sequence Classification with Deep Learning

# Using TensorFlow/Keras to predict whether a DNA sequence contains a promoter region.
# pip install tensorflow --break-system-packag

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical

# Encode DNA sequences: A=0, C=1, G=2, T=3
def encode_seq(seq):
    mapping = {'A':0, 'C':1, 'G':2, 'T':3}
    return [mapping[x] for x in seq]

# Create dummy dataset
seqs = ["ACGTACGTAC", "TGCATGCATG", "AAAAACCCCC", "GGGGTTTTAA"]
X = np.array([encode_seq(s) for s in seqs])
y = np.array([1, 0, 1, 0])  # 1=promoter, 0=non-promoter

# Reshape for LSTM [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Model
model = Sequential([
    LSTM(16, input_shape=(10,1)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=20, verbose=0)

print("Prediction for ACGTACGTAC:", model.predict(X[:1])[0])