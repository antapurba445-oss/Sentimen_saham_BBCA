import pandas as pd
import matplotlib
matplotlib.use("Agg")   # simpan ke file saja, tanpa membuka jendela
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/final_dataset.csv", parse_dates=["date"])
d = data[data["date"] >= "2025-12-01"].copy()
s = d[d["has_news"]]

fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                       gridspec_kw={"height_ratios": [3, 2, 1.5]})

ax[0].plot(d["date"], d["close"], color="black", lw=1.2)
ax[0].set_title("Harga Penutupan BBCA (disesuaikan)")

ax[1].bar(s["date"], s["sentiment"],
          color=["#2e7d32" if x > 0 else "#c62828" for x in s["sentiment"]], width=1.0)
ax[1].axhline(0, color="gray", lw=0.8)
ax[1].set_title("Skor Sentimen Harian (hybrid, -1 s/d +1)")

ax[2].bar(s["date"], s["n_news"], color="#546e7a", width=1.0)
ax[2].set_title("Jumlah Berita per Hari")

plt.tight_layout()
plt.savefig("data/sentimen_vs_harga.png", dpi=150)

# Scatter sentimen vs return hari berikutnya
x = s.dropna(subset=["sentiment", "ret_next"])
plt.figure(figsize=(6, 5))
plt.scatter(x["sentiment"], x["ret_next"] * 100, alpha=0.6)
plt.axhline(0, color="gray", lw=0.8); plt.axvline(0, color="gray", lw=0.8)
plt.xlabel("Sentimen hari t"); plt.ylabel("Return hari t+1 (%)")
plt.title("Sentimen vs return hari bursa berikutnya")
plt.tight_layout()
plt.savefig("data/scatter.png", dpi=150)