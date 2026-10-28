import pandas as pd
import yfinance as yf

SCORE_COL = "hybrid_score"
LABEL_COL = "hybrid_label"
NS = "datetime64[ns]"

df = pd.read_csv("data/scored_v3.csv", parse_dates=["published"])
df["published"] = df["published"].astype(NS)

# 0. Buang artikel lama dan halaman statis (bukan berita)
print("Sebelum filter:", len(df))
df = df[df["published"] >= "2025-12-01"]
statis = r"Harga Saham .* Hari Ini|Chart dan Harga|Analisis Teknikal|Cara Beli Saham|^Bank Central Asia Tbk \(BBCA\)$"
df = df[~df["title_clean"].str.contains(statis, case=False, regex=True)]
print("Setelah filter:", len(df))
df = df.sort_values("published").reset_index(drop=True)

# 1. Harga BBCA
start = df["published"].min().strftime("%Y-%m-%d")
px = yf.download("BBCA.JK", start=start, auto_adjust=True, progress=False)
if px.empty:
    raise SystemExit("Data harga kosong. Cek internet, lalu jalankan ulang.")
if isinstance(px.columns, pd.MultiIndex):
    px.columns = px.columns.get_level_values(0)

prices = px[["Close"]].rename(columns={"Close": "close"})
prices.index.name = "date"
prices = prices.reset_index()
prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None).astype(NS)
prices["ret"] = prices["close"].pct_change()
prices["ret_next"] = prices["ret"].shift(-1)

# 2. Berita terbit >= 16:00 dianggap esok hari
df["effective"] = (df["published"] + pd.Timedelta(hours=8)).dt.normalize().astype(NS)

# 3. Petakan ke hari bursa berikutnya
tdates = prices[["date"]].rename(columns={"date": "trading_date"})
df = pd.merge_asof(
    df.sort_values("effective"), tdates.sort_values("trading_date"),
    left_on="effective", right_on="trading_date", direction="forward",
)
df = df.dropna(subset=["trading_date"])

# 4. Agregasi per hari bursa
daily = (df.groupby("trading_date")
           .agg(sentiment=(SCORE_COL, "mean"),
                n_news=("title_clean", "count"),
                pct_pos=(LABEL_COL, lambda s: (s == "positive").mean()),
                pct_neg=(LABEL_COL, lambda s: (s == "negative").mean()))
           .reset_index())

# 5. Gabung (hari tanpa berita dibiarkan NaN)
data = prices.merge(daily, left_on="date", right_on="trading_date", how="left")
data["n_news"] = data["n_news"].fillna(0)
data["has_news"] = data["n_news"] > 0
data.to_csv("data/final_dataset.csv", index=False)

print(f"Hari bursa: {len(data)} | hari dengan berita: {int(data['has_news'].sum())}")
print(f"Periode harga: {data['date'].min().date()} s/d {data['date'].max().date()}")
print(data[data["has_news"]].tail(8)[["date", "close", "sentiment", "n_news"]].to_string())