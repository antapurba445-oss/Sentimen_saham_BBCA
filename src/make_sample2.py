import pandas as pd
from pathlib import Path

OUT = Path("data/sample_uji.csv")
if OUT.exists():
    raise SystemExit("sample_uji.csv sudah ada. Hapus manual kalau memang mau dibuat ulang.")

df = pd.read_csv("data/scored_v3.csv")
lama = pd.read_csv("data/sample_untuk_label.csv")["title_clean"]
sisa = df[~df["title_clean"].isin(lama)]

s2 = sisa.sample(40, random_state=7)[["title_clean", "hybrid_label"]].copy()
s2["label_manual"] = ""
s2.to_csv(OUT, index=False)
print("Selesai. Isi kolom label_manual di data/sample_uji.csv (pakai Excel)")