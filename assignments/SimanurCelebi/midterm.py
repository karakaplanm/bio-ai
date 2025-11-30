
# Hyperparameters
SEQ_LENGTH = 20  # Sequence length
NUM_CLASSES = 3  # 0: Gene Start, 1: Intron, 2: Exon
VOCAB_SIZE = 4   # DNA bases: A, C, G, T

# --- Data Preprocessing ---

# Encode DNA sequences: A=0, C=1, G=2, T=3
def encode_seq(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return [mapping[x] for x in seq]

# 🧪 Example Dataset (Length 20, 3 different classes)
seqs = [
    "ACGTACGTACGTACGTACGT",  # Class 0: Gene Start
    "TGCATGCATGCATGCATGCA",  # Class 1: Intron
    "AAAAACCCCCGGGGTTTTTT",  # Class 2: Exon
    "TTTTTAAAAACCCCCGGGGG",  # Class 0: Gene Start
    "GTGTGTGTGTGTGTGTGTGT",  # Class 1: Intron
    "GATTACGATTACGATTACGA"   # Class 2: Exon
]

# Sequences are truncated to SEQ_LENGTH (20)
X = np.array([encode_seq(s[:SEQ_LENGTH]) for s in seqs])

# Labels: 0=Gene Start, 1=Intron, 2=Exon
y = np.array([0, 1, 2, 0, 1, 2])

# One-Hot Encode the output
y = to_categorical(y, num_classes=NUM_CLASSES)

# ⚙️ Model Architecture (Embedding Layer Added)
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=8, input_length=SEQ_LENGTH),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

# 🛠️ Model Compilation
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

def train_model(epochs=50, verbose=0):
    print("🔧 Training model...")
    model.fit(X, y, epochs=epochs, verbose=verbose)
    print("✅ Model Training Completed.\n")

def predict_sequence(seq):
    encoded_test = np.array([encode_seq(seq[:SEQ_LENGTH])])
    prediction = model.predict(encoded_test)[0]
    predicted_class = np.argmax(prediction)
    class_names = {0: "Gene Start", 1: "Intron", 2: "Exon"}
    return prediction, predicted_class, class_names[predicted_class]

if __name__ == "__main__":
    # Train
    train_model(epochs=50, verbose=0)

    # --- Testing and Prediction ---
    test_seq = "ACGTACGTACGTACGTACGT"
    prediction, predicted_class, predicted_name = predict_sequence(test_seq)

    print(f"Test Sequence: {test_seq}")
    print(f"Predicted Probabilities (Softmax): {prediction}")
    print(f"Predicted Class: {predicted_class} ({predicted_name})")
