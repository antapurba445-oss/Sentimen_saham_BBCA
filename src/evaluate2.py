import pandas as pd
import sys
s = pd.read_csv(sys.argv[1])
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report)

s = pd.read_csv("data/sample_untuk_label.csv")
s["hybrid"] = s["lex_label"].where(s["lex_label"] != "neutral", s["label"])

urut = ["negative", "neutral", "positive"]
for nama, kol in [("zero-shot", "label"), ("lexicon", "lex_label"), ("hybrid", "hybrid")]:
    acc = accuracy_score(s["label_manual"], s[kol])
    f1 = f1_score(s["label_manual"], s[kol], average="macro", zero_division=0)
    print(f"\n=== {nama}: akurasi={acc:.0%}, macro-F1={f1:.2f} ===")
    cm = pd.DataFrame(confusion_matrix(s["label_manual"], s[kol], labels=urut),
                      index=[f"asli_{u}" for u in urut],
                      columns=[f"pred_{u}" for u in urut])
    print(cm)

print("\nDetail LEXICON:")
print(classification_report(s["label_manual"], s["lex_label"], zero_division=0))

# Daftar judul yang salah di lexicon, untuk perbaikan daftar kata
salah = s[s["label_manual"] != s["lex_label"]]
print("\nJudul yang salah di lexicon:")
print(salah[["title_clean", "lex_label", "label_manual"]].to_string())