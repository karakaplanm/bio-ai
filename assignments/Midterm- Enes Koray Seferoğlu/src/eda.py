import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
df = pd.read_csv(BASE/"data"/"breast_cancer.csv")

# 1) Target dağılımı (tek figür)
plt.figure()
df['target'].value_counts().plot(kind='bar')
plt.title('Target Distribution (0=malignant, 1=benign)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(BASE/"reports"/"figures"/"target_distribution.png")
plt.close()

# 2) İlk 6 feature için histogram (teker teker, ayrı figürler)
feature_cols = [c for c in df.columns if c != 'target'][:6]
for col in feature_cols:
    plt.figure()
    df[col].hist(bins=30)
    plt.title(f'Histogram - {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(BASE/"reports"/"figures"/f"hist_{col}.png")
    plt.close()

# 3) Basit korelasyon ısı haritası yerine tek figür kuralını korumak için
#    toplam korelasyon ortalamasını gösteren bir çubuk grafik
#    (renk belirtilmeden, matplotlib varsayılanıyla)
corr = df.drop(columns=['target']).corr().abs().mean().sort_values(ascending=False)
plt.figure()
corr.plot(kind='bar')
plt.title('Mean Absolute Correlation by Feature')
plt.xlabel('Feature')
plt.ylabel('Mean |corr|')
plt.tight_layout()
plt.savefig(BASE/"reports"/"figures"/"mean_abs_corr.png")
plt.close()

print("EDA tamamlandı. Figürler: reports/figures/")
