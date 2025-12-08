from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

yorumlar = [
    "Bu film harikaydı, bayıldım!",
    "Gerçekten çok kötüydü.",
    "Mükemmel bir deneyimdi.",
    "Hiç beğenmedim.",
    "Harika oyunculuk.",
    "Tam bir zaman kaybı.",
    "Film çok güzeldi.",
    "Berbat bir yapım."
]

duygular = [
    "pozitif",
    "negatif",
    "pozitif",
    "negatif",
    "pozitif",
    "negatif",
    "pozitif",
    "negatif"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(yorumlar)

model = MultinomialNB()
model.fit(X, duygular)

def duygu_tahmin(cumle):
    X_test = vectorizer.transform([cumle])
    sonuc = model.predict(X_test)[0]
    return sonuc

girdi = "Film oldukça güzeldi ve çok keyif aldım."
print("Duygu Analizi:", duygu_tahmin(girdi))
