import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score

s = pd.read_csv("data/sample_untuk_label.csv")
s["label_manual"] = s["label_manual"].str.strip().str.lower()
s = s.dropna(subset=["label_manual"])

# Tambahkan hasil model awal (roberta) untuk perbandingan
old = pd.read_csv("data/scored_news.csv")[["title_clean", "label"]]
old = old.drop_duplicates("title_clean").rename(columns={"label": "roberta_label"})
s = s.merge(old, on="title_clean", how="left")

metode = {"roberta (awal)": "roberta_label",
          "zero-shot": "label",
          "lexicon": "lex_label"}

print(f"Jumlah sampel berlabel: {len(s)}\n")
print("Label manual:\n", s["label_manual"].value_counts(), "\n")

for nama, kol in metode.items():
    acc = accuracy_score(s["label_manual"], s[kol])
    f1 = f1_score(s["label_manual"], s[kol], average="macro", zero_division=0)
    print(f"{nama:16s} akurasi={acc:.0%}  macro-F1={f1:.2f}")

print("\nDetail metode terbaik (ganti kolom jika perlu):")
print(classification_report(s["label_manual"], s["label"], zero_division=0))