import re
import pandas as pd
from difflib import SequenceMatcher


def clean_text(t) -> str:
    t = str(t)
    t = re.sub(r"<.*?>", " ", t)
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def strip_media_suffix(t: str) -> str:
    # ulangi sampai tidak ada lagi akhiran " - Nama Media"
    for _ in range(3):
        baru = re.sub(r"\s+[-–|]\s+[^-–|]{2,40}$", "", t)
        baru = re.sub(r"\s*[-–|]\s*$", "", baru).strip()
        if baru == t:
            break
        t = baru
    return t

def near_duplicate_filter(df, threshold=0.85):
    """Buang berita mirip (satu hari yang sama) berdasarkan kemiripan judul."""
    keep = []
    for _, group in df.groupby(df["published"].dt.date):
        kept_titles = []
        for idx, title in group["title_key"].items():
            if all(SequenceMatcher(None, title, k).ratio() < threshold for k in kept_titles):
                kept_titles.append(title)
                keep.append(idx)
    return df.loc[keep]


def main():
    df = pd.read_csv("data/raw_news.csv")
    print("Awal:", len(df))

    # Buang agregator
    df = df[~df["source"].fillna("").str.contains("feedberry", case=False)].copy()

    df["title_clean"] = df["title"].apply(clean_text).apply(strip_media_suffix)

    pola = r"\bBBCA\b|\bBCA\b|Bank Central Asia"
    df = df[df["title_clean"].str.contains(pola, case=False, regex=True)].copy()
    print("Setelah filter relevansi:", len(df))

    df["published"] = (
        pd.to_datetime(df["published"], utc=True)
        .dt.tz_convert("Asia/Jakarta")
        .dt.tz_localize(None)
    )

    df["title_key"] = (
        df["title_clean"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
    )
    df = df.drop_duplicates(subset="title_key")
    print("Setelah buang duplikat persis:", len(df))

    df = df.sort_values("published")
    df = near_duplicate_filter(df)
    print("Setelah buang duplikat mirip:", len(df))

    df = df.drop(columns="title_key").sort_values("published").reset_index(drop=True)
    df.to_csv("data/clean_news.csv", index=False)
    print(df[["published", "source", "title_clean"]].tail(8).to_string())


if __name__ == "__main__":
    main()