# 🤖 Yapay Zeka Projesi: Gen Ekspresyonu ve Rakam Tanıma

Bu proje, gen ekspresyon verileri ile biyolojik örneklerin kümelenmesi (GEO analizi) ve yapay sinir ağı kullanarak rakam tanıma (MLP Classifier – MNIST) olmak üzere iki makine öğrenimi uygulamasını içerir.

## 📊 1. Gen Ekspresyonu Analizi (GEO)
Bu bölümde:
- GEOparse ile veri indirildi
- Ekspresyon matrisi oluşturuldu
- Normalizasyon: StandardScaler
- K-Means ile kümeleme (n=3)
- PCA ile boyut indirgeme (2D)
- Görselleştirme yapıldı

Çıktılar `plots/` klasörüne kaydedilir.

## ✏️ 2. Rakam Tanıma (MLP Classifier)
Bu bölümde:
- sklearn "digits" dataset yüklendi
- Eğitim/Test ayrımı yapıldı (%80 - %20)
- MLPClassifier ile model eğitildi
- Doğruluk oranı hesaplandı

## 🚀 Çalıştırma
```
pip install -r requirements.txt
python run_analysis.py
```
