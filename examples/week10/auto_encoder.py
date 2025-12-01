"""
Autoencoder for Dimensionality Reduction of Omics Data
This script demonstrates how to build and train a simple autoencoder
using TensorFlow/Keras to reduce the dimensionality of concatenated omics data.

pip install tensorflow
"""


"""Amaç:
300 boyutlu (genomik + transkriptomik + proteomik vb.) veriyi
👉 32 boyutlu ortak biyolojik temsile indirgemek (latent space)

Bu, multi-omics entegrasyonunda:

alt tip keşfi
hasta gruplama
biyolojik imza (signature) çıkarma
noise azaltma
için kullanılır.
"""

import tensorflow as tf
from tensorflow.keras import layers

input_dim = 300  # omics concatenated
input_layer = layers.Input(shape=(input_dim,))

encoded = layers.Dense(64, activation='relu')(input_layer)
encoded = layers.Dense(32, activation='relu')(encoded)

decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = tf.keras.Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# fake data
import numpy as np
X = np.random.rand(100, 300)

autoencoder.fit(X, X, epochs=10)

latent_model = tf.keras.Model(input_layer, encoded)
latent_features = latent_model.predict(X)
print(latent_features.shape)  # (100, 32)
