import pandas as pd
from transformers import pipeline
from tqdm import tqdm

MODEL = "w11wo/indonesian-roberta-base-sentiment-classifier"


def main():
    df = pd.read_csv("data/clean_news.csv")
    texts = df["title_clean"].astype(str).tolist()

    print("Memuat model (pertama kali akan mengunduh ~500 MB)...")
    clf = pipeline(
        "sentiment-analysis",
        model=MODEL,
        top_k=None,        # ambil skor semua label
        truncation=True,
        max_length=128,
    )

    rows = []
    batch = 16
    for i in tqdm(range(0, len(texts), batch)):
        hasil = clf(texts[i:i + batch])
        for res in hasil:
            d = {r["label"].lower(): r["score"] for r in res}
            rows.append(d)

    skor = pd.DataFrame(rows)
    df = pd.concat([df.reset_index(drop=True), skor], axis=1)

    # skor -1 (sangat negatif) s/d +1 (sangat positif)
    df["sentiment_score"] = df["positive"] - df["negative"]
    df["label"] = df[["positive", "neutral", "negative"]].idxmax(axis=1)

    df.to_csv("data/scored_news.csv", index=False)

    print("\nSebaran label:")
    print(df["label"].value_counts())
    print("\nContoh per label:")
    for lab in ["positive", "neutral", "negative"]:
        print(f"\n[{lab.upper()}]")
        print(df[df["label"] == lab]["title_clean"].head(4).to_string())


if __name__ == "__main__":
    main()