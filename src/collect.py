import feedparser
import pandas as pd
from urllib.parse import quote
from pathlib import Path


def fetch_google_news(query: str) -> pd.DataFrame:
    """Ambil berita dari Google News RSS untuk satu kata kunci."""
    url = (
        "https://news.google.com/rss/search"
        f"?q={quote(query)}&hl=id&gl=ID&ceid=ID:id"
    )
    feed = feedparser.parse(url)

    rows = []
    for e in feed.entries:
        rows.append({
            "title": e.get("title"),
            "link": e.get("link"),
            "published": e.get("published"),
            "source": e.get("source", {}).get("title", ""),
            "query": query,
        })
    return pd.DataFrame(rows)


def main():
    queries = ["BBCA saham", "Bank Central Asia saham", "BCA laba bank"]

    frames = [fetch_google_news(q) for q in queries]
    new_df = pd.concat(frames, ignore_index=True)

    out = Path("data/raw_news.csv")

    # Gabungkan dengan data lama (kalau ada) supaya riwayat menumpuk
    if out.exists():
        old_df = pd.read_csv(out)
        new_df = pd.concat([old_df, new_df], ignore_index=True)

    new_df = new_df.drop_duplicates(subset="link").reset_index(drop=True)
    new_df.to_csv(out, index=False)

    print(f"Total berita tersimpan: {len(new_df)}")
    print(new_df[["published", "title"]].tail(5).to_string())


if __name__ == "__main__":
    main()