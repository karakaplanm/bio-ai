import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization

# One-hot encode DNA
def one_hot_encode(seq):
    mapping = {'A':0,'C':1,'G':2,'T':3}
    out=[]
    for base in seq:
        vec=np.zeros(4)
        vec[mapping[base]]=1
        out.append(vec)
    return np.array(out)

# Dataset
seqs=[
    "ACGTGCGCGC", "GGGCGCGGCG", "ATATATATAT",
    "AATTATAATT","CGCGTTCGCG","TATATACACA"
]
y=np.array([1,1,0,0,1,0])

X=np.array([one_hot_encode(s) for s in seqs])

# Model
model=Sequential([
    GRU(32,input_shape=(10,4)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16,activation='relu'),
    Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X,y,epochs=15,batch_size=2,verbose=1)

# Test example
test="GCGCGCGTTA"
test_encoded=one_hot_encode(test).reshape(1,10,4)
print(model.predict(test_encoded))
