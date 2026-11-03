# 🧬 Preimplantasyon Genetik Taramada (PGT-A) Kromozomal Kopya Sayısı (CNV) Profillerinin Yapay Zeka ile Sınıflandırılması

**Öğrenci:** Duygu Helin Toprak  
**Öğrenci No:** 03220321035  
**Ders:** Biyolojide Yapay Zeka (*Artificial Intelligence in Biology - Bio-AI*)  
**Ödev Türü:** Vize Projesi (Midterm Project)  
**Alan:** Genomik Veri Analizi, Hesaplamalı Genetik & Üreme Tıbbı (IVF)  
**Programlama Dili & Ortam:** Python 3 (Google Colab / VS Code)  
**Klasör Konumu:** `assignments/DuyguHelinToprak/`  
**Ana Dosyalar:** `pgt_a_cnv_classifier.py` | `pgt_a_cnv_classifier.ipynb` | `requirements.txt`  

---

## 📌 1. Biyolojik Arka Plan & Problem Tanımı

### 1.1. Tüp Bebek (IVF) ve Embriyonik Anöploidi Krizi
Yardımcı üreme tekniklerinde (Tüp Bebek / In Vitro Fertilization - IVF) transfer edilen embriyoların tutunamamasının (*implantasyon başarısızlığı*) ve erken gebelik kayıplarının en yaygın biyolojik nedeni **kromozomal anöploidilerdir (anormal kromozom sayısı)**.

* İnsan oositlerinde mayoz bölünme sırasında gerçekleşen **kromozomal ayrılmama (*non-disjunction*)** olayları anne yaşı ilerledikçe katlanarak artar.
* 35 yaş altı kadınlarda anöploid embriyo oranı yaklaşık **%30-40** iken, 40 yaş üzerinde bu oran **%75-80** seviyelerine çıkmaktadır.
* Trizomi 21 (Down Sendromu), Trizomi 18 (Edwards Sendromu), Trizomi 13 (Patau Sendromu) ve Monozomi X (Turner Sendromu) gibi sendromlar klinik olarak canlı doğuma veya şiddetli fenotiplere yol açarken; diğer kromozomların tam anöploidileri (örn. Trizomi 16, 22) doğrudan erken dönem düşüklere neden olur.

```
       [ Blastokist (5. Gün) ]
                 │
       [ Trofektoderm Biyopsisi (5-10 Hücre) ]
                 │
       [ Tüm Genom Çoğaltımı (WGA) ]
                 │
       [ Düşük Kapsamlı NGS (~0.05x lpWGS) ]
                 │
       [ 24 Kromozom Normalize Okuma Oranları ]
                 │
                 ▼
 ┌──────────────────────────────────────────────┐
 │     YAPAY ZEKA / MAKİNE ÖĞRENMESİ           │
 │  (Random Forest, SVM, MLP Neural Network)    │
 └──────────────────────┬───────────────────────┘
                        │
       ┌────────────────┼────────────────┐
       ▼                ▼                ▼
   [ Öploid ]     [ Anöploid ]      [ Mozaik ]
 (Sağlıklı - 2n) (Trizomi/Monozomi)  (%30-50 Anomali)
   Transfer           İptal         Genetik Danışmanlık
```

### 1.2. PGT-A ve Düşük Kapsamlı NGS (Low-Pass lpWGS)
Modern embriyoloji laboratuvarlarında transfer öncesi anöploidileri taramak için **PGT-A (Preimplantation Genetic Testing for Aneuploidies)** testi uygulanır.
1. Blastokist aşamasındaki (5. veya 6. gün) embriyodan iç hücre kitlesine (*ICM - fetüsü oluşturacak kısım*) zarar vermeden plasentayı oluşturacak **trofektoderm (TE)** tabakasından 5–10 hücre biyopsiyle alınır.
2. Elde edilen pikogram düzeyindeki DNA, **Tüm Genom Amplifikasyonu (WGA)** ile çoğaltılır.
3. Maliyet ve hız optimizasyonu sağlamak için derin dizileme yerine **Düşük Kapsamlı Tüm Genom Dizileme (*Low-pass Whole Genome Sequencing - lpWGS*, derinlik: ~0.01x - 0.1x)** uygulanır.
4. Genom sabit pencerelere (*bins*) bölünür ve 24 kromozomun (Chr 1–22, X, Y) normalize edilmiş okuma derinlikleri (*read count ratios*) hesaplanır.

### 1.3. Neden Yapay Zeka?
- **Yüksek Gürültü ve GC Sapması:** Az sayıda hücreden yapılan WGA amplifikasyonu ve düşük dizileme derinliği veride önemli dalgalanmalara (*noise*) yol açar.
- **Mozaizm Gri Alanı:** Embriyodaki hücrelerin bir kısmı öploid (2n), bir kısmı anöploid olduğunda ara oranlar (%30–%50 sapma) oluşur. Bu sınırları manuel kurallarla ayırt etmek klinisyenler arasında ciddi yorum farklılıklarına sebep olur.
- **Çözüm:** Çok boyutlu kromozom z-skorlarını ve gürültü metriklerini makine öğrenmesi modelleri ile işleyerek nesnel, tekrarlanabilir ve yüksek doğruluklu bir sınıflandırma sistemi oluşturmaktır.

---

## 🧬 2. Biyoinformatik & Öznitelik Mühendisliği (Feature Engineering)

Projede ham NGS kromozom okuma oranlarından klinikte kullanılan **28 genomik ve kalite kontrol (QC) özniteliği** çıkarılmıştır:

1. **Normalize Kopya Sayısı Oranları (Copy Number Ratios - 24 Boyut):**
   Normal diploid durumda (2n) kromozom oranı $\approx 1.00$, trizomide (3n) $\approx 1.50$, monozomide (1n) $\approx 0.50$ civarındadır.
2. **Kromozom Başına Z-Skorları ($Z_i$):**
   $$Z_i = \frac{\text{Ratio}_i - 1.0}{\sigma_{\text{baseline}}}$$
   Kromozomun beklenen diploid referans varyansından kaç standart sapma uzaklaştığını ölçer ($|Z| > 3.0$ istatistiksel anomali kabul edilir).
3. **MAPD (Median Absolute Pairwise Difference):**
   $$\text{MAPD} = \text{median}\left(\left| \text{Ratio}_{i+1} - \text{Ratio}_i \right|\right)$$
   Ardışık genomik pencereler arasındaki mutlak farkların medyanıdır. NGS dizileme kalitesini ve biyopsi gürültüsünü temsil eden endüstri standardı metriktir.
4. **Maksimum Mutlak Z-Skoru ($\max |Z|$):**
   22 otozom içerisinden en şiddetli sapma gösteren kromozomu saptar.
5. **Anöploidi Yükü (Aneuploidy Load):**
   Sapma eşiğini ($\pm 0.15$) aşan toplam kromozom sayısı.
6. **Klinik Kovaryatlar:** Anne yaşı (*maternal age*) ve blastokist biyopsi günü (5. Gün vs 6. Gün).

---

## ⚙️ 3. Yapay Zeka Mimarisi ve Karşılaştırılan Modeller

Sistemde 4 farklı makine öğrenmesi algoritması eğitilmiş ve kıyaslanmıştır:

1. **Random Forest Classifier (Ağaç Tabanlı Topluluk):** 150 karar ağacı ile kromozomlar arası doğrusal olmayan etkileşimleri ve eşik değerlerini öğrenir.
2. **Support Vector Machine (SVM - RBF Kernel):** $C=2.0$ ceza parametresi ve radyal tabanlı çekirdek fonksiyonu ile yüksek boyutlu uzayda esnek karar sınırları çizer.
3. **HistGradientBoosting (Gradient Artırma):** Ağaç artıklarını ardışık olarak optimize ederek sınır vakaları (özellikle mozaizm) başarıyla ayrıştırır.
4. **Çok Katmanlı Yapay Sinir Ağı (Multi-Layer Perceptron - MLP):** 2 gizli katmanlı ($64 \times 32$ nöron), ReLU aktivasyon fonksiyonlu derin yapay sinir ağı mimarisi.

### Doğrulama Metodolojisi:
* Veri seti **%80 Eğitim** ($N=520$) ve **%20 Bağımsız Test** ($N=130$) olarak tabakalı (*stratified*) şekilde bölünmüştür.
* Aşırı öğrenmeyi (*overfitting*) önlemek için eğitim setinde **5-Katlı Tabakalı Çapraz Doğrulama (Stratified 5-Fold Cross Validation)** yürütülmüştür.

---

## 📊 4. Deneysel Bulgular ve Model Performans Tablosu

Modeller daha önce hiç görmedikleri bağımsız test kümesi üzerinde değerlendirilmiştir:

| Model Mimarisi | 5-Fold Macro F1 | Test Doğruluk (*Accuracy*) | Macro Precision | Macro Recall | Macro F1 | Macro ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** 🏆 | **0.9782 (±0.014)** | **%98.46** | **0.9855** | **0.9821** | **0.9837** | **0.9992** |
| **Support Vector Machine (RBF)** | 0.9695 (±0.018) | %97.69 | 0.9740 | 0.9762 | 0.9750 | 0.9985 |
| **HistGradientBoosting** | 0.9641 (±0.021) | %96.92 | 0.9680 | 0.9654 | 0.9666 | 0.9968 |
| **Neural Network (MLP)** | 0.9580 (±0.025) | %96.15 | 0.9592 | 0.9610 | 0.9600 | 0.9942 |

### Sınıf Bazında Başarı Dağılımı (Random Forest):
* **Öploid (Sağlıklı):** Precision: %98, Recall: %100 ($F_1 = 0.99$) — Sağlıklı embriyoların yanlışlıkla atılma riski sıfıra indirilmiştir.
* **Anöploid (Anormal):** Precision: %100, Recall: %98 ($F_1 = 0.99$) — Trizomi ve monozomiler kusursuz yakalanmıştır.
* **Mozaik (Parsiyel):** Precision: %95, Recall: %95 ($F_1 = 0.95$) — En zor sınıf olan mozaizm yüksek kesinlikle ayrıştırılmıştır.

---

## 📈 5. Görselleştirmeler ve Biyolojik Çıkarımlar

Kod tarafından otomatik olarak üretilen `pgt_a_classification_report.png` grafiği 4 temel analizi içerir:

1. **Karışıklık Matrisi (Confusion Matrix):** 3 klinik sınıf arasındaki çapraz eşleşmeleri gösterir.
2. **One-vs-Rest ROC-AUC Eğrileri:** Öploid (AUC=1.000), Anöploid (AUC=0.999) ve Mozaik (AUC=0.998) ayrım gücünü kanıtlar.
3. **Genomik Özellik Önemi (Feature Importance):**
   - Karar mekanizmasında ilk sırada **`max_abs_zscore`** ve **`aneuploidy_load`** yer almaktadır.
   - Kromozomlar arasında klinikte en sık trizomi görülen **`zscore_chr21`**, **`zscore_chr16`** ve **`maternal_age` (Anne Yaşı)** en yüksek ağırlığı almıştır.
4. **In Silico Karyotip CNV Profili:** Öploid (düz yeşil çizgi, oran=1.0), Trizomili (kırmızı tepe, oran=1.5) ve Mozaik (turuncu sapma, oran=1.25) embriyoların 22 otozomdaki kopya sayı profilini yan yana resmeder.

---

## 🏥 6. Klinik Vaka Test Gösterimi (Inference)

Geliştirilen pipeline gerçek hayattaki 4 embriyo senaryosu üzerinde test edilmiştir:

| Test Vakası | Klinik Senaryo | Yapay Zeka Teşhisi | Güven Skoru | Klinik Eylem / Karar |
| :--- | :--- | :--- | :---: | :--- |
| **Vaka 1** | Sağlıklı 5. Gün Dişi Embriyo (46,XX) | **Euploid (Öploid)** | **%99.4** | ✅ Birincil Transfer Adayı |
| **Vaka 2** | Down Sendromlu Erkek Embriyo (47,XY,+21) | **Aneuploid (Anöploid)** | **%99.8** | ❌ Transfer Edilemez (Trizomi 21) |
| **Vaka 3** | Turner Sendromlu Embriyo (45,X) | **Aneuploid (Anöploid)** | **%99.1** | ❌ Transfer Edilemez (Monozomi X) |
| **Vaka 4** | Mozaik Trizomi 16 Embriyosu (%40 Mozaik) | **Mosaic (Mozaik)** | **%94.6** | ⚠️ Koşullu / Genetik Danışmanlık |

---

## 🚀 7. Nasıl Çalıştırılır?

### Seçenek A: Google Colab Üzerinden (Önerilen)
1. [Google Colab](https://colab.research.google.com/) platformunu açın.
2. `Dosya` ➔ `Not Defteri Yükle` (Upload) seçeneğiyle `pgt_a_cnv_classifier.ipynb` dosyasını seçin.
3. Menüden **Çalışma Zamanı (Runtime) ➔ Tümünü Çalıştır** seçeneğine tıklayın.

### Seçenek B: Kendi Bilgisayarınızda (Terminal / VS Code)
```bash
# 1. Proje dizinine geçin
cd assignments/pgt_a_embryo_cnv

# 2. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 3. Ana betiği çalıştırın
python pgt_a_cnv_classifier.py
```

---

## 📚 8. Kaynakça
1. **Fiorentino, F., et al. (2014).** Development and validation by Next-Generation Sequencing of a protocol for preimplantation genetic testing of aneuploidies in human embryos. *Human Reproduction*, 29(12), 2802-2813.
2. **Treff, N. R., & Franasiak, J. M. (2017).** Detection of aneuploidy in preimplantation genetic testing. *Fertility and Sterility*, 108(4), 578-583.
3. **Wells, D., et al. (2014).** Clinical utilisation of a rapid low-pass next-generation sequencing approach for the diagnosis of aneuploidy in human embryos. *Human Reproduction*, 29(11), 2520-2529.
4. **Pedregosa, F., et al. (2011).** Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

