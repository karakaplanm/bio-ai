# =========================
# Compare predicted ABL1 (P00519) binding of three drugs
# =========================
# Scores come from alpha_deeppurpose.py, alpha_deeppurpose_nilotinib.py
# and alpha_deeppurpose_dasatinib.py (Morgan_CNN_BindingDB, pKd).

import matplotlib.pyplot as plt

drugs = {
    "Imatinib": 7.1242,
    "Nilotinib": 7.4660,
    "Dasatinib": 8.9048,
}

# Weakest at the bottom, strongest at the top
names = list(drugs)
pkd = [drugs[n] for n in names]
kd_nM = [10 ** (-p) * 1e9 for p in pkd]

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.barh(names, pkd, height=0.45, color="#2a78d6")

for i, (p, kd) in enumerate(zip(pkd, kd_nM)):
    ax.text(p + 0.1, i, f"pKd {p:.2f}  ·  Kd {kd:.1f} nM", va="center", fontsize=10, color="#0b0b0b")

ax.set_xlim(0, 11)
ax.set_xlabel("Predicted pKd (higher = stronger binding)", color="#52514e")
ax.set_title("Predicted binding to ABL1 kinase (DeepPurpose)", loc="left", fontsize=12, color="#0b0b0b")
ax.tick_params(colors="#52514e", length=0)
ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7")

plt.tight_layout()
plt.savefig("result/drug_comparison.png", dpi=150)
plt.show()
