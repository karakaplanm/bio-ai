# ?? Farmakogenomik Apriori Analizi  
Bu proje, gen ekspresyon verilerini kullanarak **ilaç hassasiyeti ve direnç iliþkilerini** Apriori birliktelik kuralý algoritmasýyla analiz eder. Gradio tabanlý etkileþimli arayüz sayesinde, kullanýcý destek (support) ve güven (confidence) eþiklerini belirleyerek hýzlýca kurallar üretebilir.

## ?? Proje Amacý
- Rastgele oluþturulmuþ 100 hücre hattý ve 500 genlik bir veri seti üzerinden,
- IC50 ilaç tepkisi deðerleri hesaplanýr,
- Gen ekspresyonu ikili (yüksek/düþük) olarak kategorize edilir,
- Apriori algoritmasý ile ilaç hassasiyeti/direnç durumlarýný belirleyen gen kombinasyonlarý çýkarýlýr.

## ?? Kullanýlan Teknolojiler
- Python
- mlxtend
- pandas / numpy
- Gradio
- Apriori & Birliktelik Kurallarý

## ? Kurulum
pip install pandas numpy mlxtend gradio

## ? Çalýþtýrma
python apriori_ilac_gen_kesfi.py

## ?? Arayüz Özellikleri
- Destek (support) eþiðini ayarlama  
- Güven (confidence) eþiðini ayarlama  
- Ýlk 10 en güçlü kuralýn görüntülenmesi  
- Markdown + HTML tablo çýktýsý  

## ?? Dosya Yapýsý
?? Proje
 ? ?? apriori_ilac_gen_kesfi.py
 ? ?? README.md

## ?? Not
Bu proje örnek veri üretir; gerçek verilerle kolayca entegre edilebilir.

## ?? Ýletiþim
Geliþtirme veya geniþletme için yardýmcý olabilirim!
