import pandas as pd
import numpy as np
from scipy.stats import spearmanr

data = pd.read_csv("data/final_dataset.csv", parse_dates=["date"])
d = data[data["has_news"]].dropna(subset=["sentiment", "ret"]).copy()

# Lead/lag: sentimen hari t vs return hari t+k (k=0 sama hari, k>=1 = prediksi)
for k in range(0, 4):
    d[f"ret_t{k}"] = data["ret"].shift(-k).reindex(d.index)


def uji(df, judul):
    print(f"\n=== {judul} (n hari = {len(df)}) ===")
    for k in range(0, 4):
        x = df[["sentiment", f"ret_t{k}"]].dropna()
        if len(x) < 10:
            print(f"lag {k}: data terlalu sedikit")
            continue
        r, p = spearmanr(x["sentiment"], x[f"ret_t{k}"])
        ket = "sama hari" if k == 0 else f"{k} hari ke depan"
        print(f"lag {k} ({ket:16s}) r={r:+.3f}  p={p:.3f}  n={len(x)}")


uji(d, "Semua hari dengan berita")

# Robustness: buang klaster buyback 19-28 Okt yang mendominasi
uji(d[d["date"] < "2026-10-19"], "Tanpa klaster 19-28 Oktober")

# Apakah volume berita saja sudah berhubungan dengan pergerakan harga?
d["abs_ret"] = d["ret"].abs()
r, p = spearmanr(d["n_news"], d["abs_ret"])
print(f"\nJumlah berita vs |return|: r={r:+.3f}  p={p:.3f}")

print()
for min_n in [1, 3, 5]:
    sub = d[d["n_news"] >= min_n]
    x = sub[["sentiment", "ret_t1"]].dropna()
    if len(x) >= 10:
        r, p = spearmanr(x["sentiment"], x["ret_t1"])
        print(f"n_news >= {min_n}: r={r:+.3f}  p={p:.3f}  n={len(x)}")
    else:
        print(f"n_news >= {min_n}: n={len(x)}, terlalu sedikit untuk diuji")