
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

X = np.random.rand(500, 1000)

input_dim = X.shape[1]
encoding_dim = 64

input_layer = Input(shape=(input_dim,))
encoded = Dense(256, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(256, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(1e-3), loss='mse')

autoencoder.fit(X, X, epochs=20, batch_size=32, validation_split=0.1)
autoencoder.save("gene_autoencoder.h5")
