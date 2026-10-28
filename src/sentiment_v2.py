import re
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Model multibahasa yang mendukung bahasa Indonesia untuk zero-shot
MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

LABELS = [
    "kabar baik bagi investor saham",
    "kabar buruk bagi investor saham",
    "informasi netral",
]
TEMPLATE = "Berita ini adalah {}."

# Lexicon finansial Indonesia (baseline pembanding)
POS = {"naik", "melonjak", "meroket", "tumbuh", "laba", "cuan", "menguat", "rekor",
       "tembus", "dividen", "buyback", "borong", "net buy", "positif", "optimistis",
       "layak", "target naik", "rebound", "angin segar", "melesat", "untung"}
NEG = {"turun", "anjlok", "merosot", "rugi", "melemah", "koreksi", "tertekan",
       "tekanan", "jual", "net sell", "negatif", "dihantui", "risiko", "ambles",
       "longsor", "hengkang", "pesimistis", "bantah", "gagal"}


def lexicon_score(text: str) -> int:
    t = text.lower()
    pos = sum(1 for w in POS if re.search(rf"\b{re.escape(w)}\b", t))
    neg = sum(1 for w in NEG if re.search(rf"\b{re.escape(w)}\b", t))
    return pos - neg


def main():
    df = pd.read_csv("data/clean_news.csv")
    texts = df["title_clean"].astype(str).tolist()

    print("Memuat model zero-shot (unduhan ~1 GB pertama kali)...")
    clf = pipeline("zero-shot-classification", model=MODEL)

    rows = []
    for i in tqdm(range(0, len(texts), 8)):
        hasil = clf(texts[i:i + 8], candidate_labels=LABELS,
                    hypothesis_template=TEMPLATE, multi_label=False)
        if isinstance(hasil, dict):
            hasil = [hasil]
        for res in hasil:
            d = dict(zip(res["labels"], res["scores"]))
            rows.append({
                "p_pos": d[LABELS[0]],
                "p_neg": d[LABELS[1]],
                "p_neu": d[LABELS[2]],
            })

    skor = pd.DataFrame(rows)
    df = pd.concat([df.reset_index(drop=True), skor], axis=1)
    df["sentiment_score"] = df["p_pos"] - df["p_neg"]
    df["label"] = df[["p_pos", "p_neu", "p_neg"]].idxmax(axis=1).map(
        {"p_pos": "positive", "p_neu": "neutral", "p_neg": "negative"}
    )
    df["lex_score"] = df["title_clean"].apply(lexicon_score)
    df["lex_label"] = df["lex_score"].apply(
        lambda x: "positive" if x > 0 else ("negative" if x < 0 else "neutral")
    )

    df.to_csv("data/scored_v2.csv", index=False)

    print("\nZero-shot:")
    print(df["label"].value_counts())
    print("\nLexicon:")
    print(df["lex_label"].value_counts())
    print("\nKesepakatan zero-shot vs lexicon:",
          f"{(df['label'] == df['lex_label']).mean():.0%}")


if __name__ == "__main__":
    main()