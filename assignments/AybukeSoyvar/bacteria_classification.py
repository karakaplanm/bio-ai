import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

# --- 1. Parametreler ---
IMG_SIZE = (224, 224)
BATCH = 16
EPOCHS = 10
train_dir = "C:/Users/yugur/OneDrive/Belgeler/AybukeSoyvar/data/train"
val_dir = "C:/Users/yugur/OneDrive/Belgeler/AybukeSoyvar/data/val"

# --- 2. Veri Yükleme ve Artırma (Augmentation) ---
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical"
)

# --- 3. Model (MobileNetV2 Transfer Learning) ---
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
    pooling="avg"
)
base_model.trainable = False  # önceden eğitilmiş ağı dondur

model = models.Sequential([
    base_model,
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer=optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# --- 4. Eğitim ---
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# --- 5. Sonuç Görselleştirme ---
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.legend()
plt.tight_layout()
plt.show()

# --- 6. İnce Ayar (Fine-tuning) ---
base_model.trainable = True
for layer in base_model.layers[:-30]:  # sadece son 30 katmanı eğit
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_epochs = 5
model.fit(train_data, validation_data=val_data, epochs=fine_tune_epochs)
model.save("bacteria_colony_model.h5")
print("Model başarıyla kaydedildi ✅")
