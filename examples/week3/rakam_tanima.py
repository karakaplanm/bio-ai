# TensorFlow ve Keras'ı içe aktaralım
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Veriyi yükle (MNIST: 28x28 piksel el yazısı rakamlar)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Veriyi normalize et (0-255 arası pikselleri 0-1 aralığına ölçekle)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Veriyi düzleştir (28x28 → 784)
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Modeli tanımla
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(784,)),
    layers.Dropout(0.2),  # overfitting'i azaltmak için dropout
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")  # 10 sınıf (0–9 rakamları)
])

# Modeli derle
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Eğitimi başlat
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Test verisinde değerlendirme
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest doğruluğu: {test_acc:.4f}")

# Birkaç tahmin görelim
predictions = model.predict(x_test[:5])
print("\nGerçek etiketler:", y_test[:5])
print("Tahmin edilen etiketler:", predictions.argmax(axis=1)[:5])

