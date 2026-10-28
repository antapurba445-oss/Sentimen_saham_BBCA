import pandas as pd
from lexicon import lexicon_score

df = pd.read_csv("data/scored_v2.csv")
df["lex_score"] = df["title_clean"].apply(lexicon_score)
df["lex_label"] = df["lex_score"].apply(
    lambda x: "positive" if x > 0 else ("negative" if x < 0 else "neutral"))

# Hybrid: pakai lexicon kalau ada kata kunci, kalau tidak serahkan ke zero-shot
pakai_lex = df["lex_label"] != "neutral"
df["hybrid_label"] = df["lex_label"].where(pakai_lex, df["label"])
df["hybrid_score"] = (df["lex_score"] / 2).clip(-1, 1).where(pakai_lex, df["sentiment_score"])

df.to_csv("data/scored_v3.csv", index=False)
print(df["hybrid_label"].value_counts())
