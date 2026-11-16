# 🧬 Likit Biyopside Hücre Dışı DNA (cfDNA) Fragmantasyon Profilleri ile Erken Evre Kanser Tespiti

**Öğrenci:** Sude Merve Bulut  
**Öğrenci No:** 03190321083  
**Ders:** Biyolojide Yapay Zeka (*Artificial Intelligence in Biology - Bio-AI*)  
**Ödev Türü:** Vize Projesi (Midterm Project)  
**Alan:** Kanser Genomikleri, Epigenetik & Likit Biyopsi Biyoinformatiği  
**Programlama Dili & Ortam:** Python 3 (Google Colab / VS Code)  
**Klasör Konumu:** `assignments/SudeMerveBulut/`  
**Ana Dosyalar:** `cfdna_fragmentomics_classifier.py` | `cfdna_fragmentomics_classifier.ipynb` | `requirements.txt`  

---

## 📌 1. Biyolojik Arka Plan & Problem Tanımı

### 1.1. Erken Evre Kanser Teşhisinde Likit Biyopsinin Önemi
Onkolojide sağkalım oranlarını belirleyen en kritik parametre tümörün **Evre I veya Evre II** aşamasında tespit edilmesidir. Geleneksel doku biyopsileri cerrahi müdahale gerektirir ve tekrarlanabilirliği düşüktür. **Likit biyopsi**, periferik kandan alınan plazma örneğinde serbest dolaşan **hücre dışı DNA (cell-free DNA - cfDNA)** moleküllerini inceleyerek non-invaziv erken tanı imkanı sunar.

* **Sağlıklı Nükleozom Mimarisi:** Sağlıklı bireylerde hematopoietik hücrelerin apoptozu sırasında DNA, histon oktamerlerine sarılı olduğu bölgelerden korunur ve nükleozomlar arası linker bölgelerinden kesilir. Bu durum plazmada **~167 baz çifti (bp)** uzunluğunda karakteristik mononükleozomal bir tepe noktası yaratır.
* **Tümör Kromatin Düzensizliği:** Kanser hücrelerinden kana salınan dolaşımdaki tümör DNA'sı (**ctDNA**), yoğun transkripsiyonel aktivite ve açık kromatin yapısı nedeniyle nükleaz enzimlerine daha açıktır. Sonuç olarak tümör kaynaklı fragmanlar **100–150 bp** aralığında yoğunlaşır ve kromozom kolları boyunca heterojen dağılım gösterir.

```
          [ Periferik Kan Örneği (10 mL) ]
                         │
            [ Plazma İzolasyonu & cfDNA ]
                         │
       [ Düşük Kapsamlı Tüm Genom Dizileme (lpWGS) ]
                         │
   [ 39 Kromozom Kolu Kısa/Uzun (100-150 / 151-220 bp) Oranları ]
                         │
                         ▼
 ┌────────────────────────────────────────────────────────┐
 │            YAPAY ZEKA / MAKİNE ÖĞRENMESİ               │
 │  (Extra Trees, Gradient Boosting, Random Forest, LR)   │
 └───────────────────────────┬────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   [ Sağlıklı ]    [ Akciğer (NSCLC) ]  [ Kolorektal (CRC) ]
  Rutin Tarama        Toraks BT (LDCT)       Kolonoskopi
```

### 1.2. DELFI (DNA Evaluation of Fragments for Early Interception) Yaklaşımı
Bu projede Cristiano et al. (*Nature*, 2019) tarafından geliştirilen **DELFI** fragmanomik prensipleri temel alınmıştır:
1. Genom 39 otozomal kromozom koluna (`1p`–`22q`) ayrılır.
2. Her koldaki **kısa fragmanların (100–150 bp)** **uzun fragmanlara (151–220 bp)** oranı (*Short-to-Long Ratio - SL Ratio*) hesaplanır.
3. Ek olarak plazma cfDNA konsantrasyonu (`ng/mL`), modal fragman boyutu (`bp`) ve **DNASE1L3** nükleaz aktivitesinin göstergesi olan **5' CCCA uç motif frekansı** modele dahil edilir.

---

## 🧬 2. Biyoinformatik & Fragmanomik Öznitelik Mühendisliği

Ham 39 kromozom kolu oranlarından **50 adet biyoinformatik ve klinik öznitelik** türetilmiştir:

1. **Kromozom Kolu SL Oranları (39 Boyut):** Sağlıklı bireylerde ~0.18 bandında seyrederken, tümör varlığında 0.25–0.38 aralığına yükselir.
2. **Shannon Fragmantasyon Entropisi ($H$):**
   $$H = -\sum_{i=1}^{39} p_i \log(p_i)$$
   Genom genelindeki fragman dağılımının düzensizliğini (kromatin heterojenitesini) ölçer.
3. **Sağlıklı Referans Kosinüs Benzerliği:** Bireyin 39 kollu fragman vektörünün ideal sağlıklı referans profiline olan açısal yakınlığını hesaplar.
4. **Doku Kökeni Spesifik İmza İndeksleri (`nsclc_signature_index` & `crc_signature_index`):** Akciğer kanserinde sık görülen `1q, 3q, 5p, 19q` sapmaları ile kolorektal kanserde öne çıkan `13q, 18q, 20p, 20q` sapmalarını birbirinden ayırt eder.
5. **5' Uç Motifi (`motif_CCCA_freq`) ve Modal Fragman Boyutu (`modal_fragment_bp`):** Epigenetik kesim örüntülerini yakalar.

---

## ⚙️ 3. Yapay Zeka Mimarisi ve Doğrulama Stratejisi

Sistemde 4 farklı sınıflandırma algoritması eğitilmiş ve kıyaslanmıştır:
* **Extra Trees Classifier (Extremely Randomized Trees):** Yüksek boyutlu fragmanomik özelliklerde varyansı düşürerek karar sınırlarını optimize eder.
* **Gradient Boosting Classifier:** Hata artıklarını ardışık olarak minimize ederek erken evre düşük ctDNA sinyallerini yakalar.
* **Random Forest Classifier:** Biyobelirteç önem sıralaması (*Gini Importance*) ve kararlı sınıflandırma sağlar.
* **Logistic Regression (L2 Regularization):** Doğrusal referans taban modeli olarak kullanılmıştır.

Veri seti ($N=720$) **%80 Eğitim** ($N=576$) ve **%20 Bağımsız Test** ($N=144$) olarak tabakalı şekilde ayrılmış; aşırı öğrenmeyi engellemek için **5-Katlı Tabakalı Çapraz Doğrulama (Stratified 5-Fold CV)** uygulanmıştır.

---

## 📊 4. Deneysel Bulgular ve Performans Metrikleri

Kodu çalıştırarak doğrulanan bağımsız test kümesi sonuçları aşağıdadır:

| Model Mimarisi | 5-Fold Macro F1 | Test Doğruluk (*Accuracy*) | Macro Precision | Macro Recall | Macro F1 | Macro ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Extra Trees Classifier** 🏆 | **0.9851 (±0.008)** | **%99.31** | **0.9924** | **0.9907** | **0.9915** | **0.9998** |
| **Random Forest Classifier** | 0.9894 (±0.007) | %98.61 | 0.9830 | 0.9830 | 0.9830 | 0.9997 |
| **Gradient Boosting Classifier** | 0.9894 (±0.007) | %97.92 | 0.9740 | 0.9752 | 0.9745 | 0.9997 |
| **Logistic Regression (L2)** | 0.9851 (±0.011) | %100.00 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

---

## 📈 5. Görselleştirme Çıktıları (`cfdna_fragmentomics_report.png`)

1. **Karışıklık Matrisi:** Sağlıklı, Akciğer (NSCLC) ve Kolorektal (CRC) sınıfları arasındaki ayrım başarısını gösterir.
2. **One-vs-Rest ROC-AUC Eğrileri:** Tüm sınıflar için $>0.998$ AUC değerini görselleştirir.
3. **En Belirleyici 12 cfDNA Biyobelirteci:** `nsclc_signature_index`, `crc_signature_index`, `aberrant_arms_count` ve `motif_CCCA_freq` özniteliklerinin tanıdaki baskın rolünü ortaya koyar.
4. **Genom Boyu DELFI Fragmantasyon Profili:** 39 kromozom kolu boyunca sağlıklı bireylerin düz çizgisi ile kanser hastalarının karakteristik zirvelerini karşılaştırır.

---

## 🚀 6. Kurulum ve Çalıştırma

### Google Colab Üzerinden
1. `cfdna_fragmentomics_classifier.ipynb` dosyasını [Google Colab](https://colab.research.google.com/) ortamına yükleyin.
2. **Çalışma Zamanı ➔ Tümünü Çalıştır** adımıyla tüm hücreleri yürütün.

### Yerel Ortamda (Terminal / VS Code)
```bash
cd assignments/SudeMerveBulut
pip install -r requirements.txt
python cfdna_fragmentomics_classifier.py
```

---

## 📚 7. Kaynakça
1. **Cristiano, S., et al. (2019).** Genome-wide cell-free DNA fragmentation in patients with cancer. *Nature*, 570(7761), 385-389.
2. **Jiang, P., et al. (2020).** Preferred end coordinates and somatic variants as signatures of circulating tumor DNA associated with hepatocellular carcinoma. *PNAS*, 117(25), 14411-14420.
3. **Lo, Y. M. D., et al. (2021).** Epigenetics, fragmentomics, and topology of cell-free DNA in liquid biopsies. *Science*, 372(6538), eaaw3616.
