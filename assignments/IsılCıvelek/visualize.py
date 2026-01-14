
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model("gene_autoencoder.h5")
X = np.random.rand(1, 1000)
reconstructed = model.predict(X)

plt.plot(X.flatten()[:100], label="Original")
plt.plot(reconstructed.flatten()[:100], label="Reconstructed")
plt.legend()
plt.show()
