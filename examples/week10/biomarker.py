"""
Identify biomarker genes using Random Forest feature importance.

This script trains a Random Forest classifier on gene expression data
and identifies the top biomarker genes based on feature importance scores.  
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np

# X → 100 örnek (hücre, hasta, doku… olabilir)
X = np.random.rand(100, 50)  # 50 gene expression
# y → binary sınıflar (örneğin, hasta vs. sağlıklı)
y = np.random.randint(0, 2, 100)

clf = RandomForestClassifier()
clf.fit(X, y)

# Random Forest, her ağacın bölünmelerinde hangi genlerin daha çok bilgi kazancı 
# sağladığına bakarak her gene bir önem skoru verir.
# Önem skoru yüksek gen → sınıflandırmada büyük rol oynuyor
# Önem skoru düşük gen → sonuç üzerinde pek etkisi yok
# Bu, genomik biyodizide biomarker discovery için temel yöntemlerden biridir.

importances = clf.feature_importances_
top_genes = np.argsort(importances)[-5:]
print("Top biomarker genes:", top_genes)

"""
np.argsort(importances) → importance değerlerini küçükten büyüğe sıralar
[-5:] → son 5 tanesini alır (en büyük olanlar)
"""

"""
Bu yöntemle:
Kanser alt tiplerini ayıran genler
İlaç yanıtı veren–vermeyen hastaları ayıran genler
Hastalık progresyon göstergeleri
Hastalığa özgü imzalar
bulunabilir.
"""