import re

# Cocok di awal kata saja, sehingga "diborong", "menguatnya" tetap terdeteksi
POS = [r"naik", r"melonjak", r"meroket", r"tumbuh", r"laba", r"cuan", r"menguat",
       r"rekor", r"tembus", r"dividen", r"buyback", r"(?:di|mem)?borong",
       r"(?:di)?serok", r"rebutan", r"net buy", r"asing masuk", r"dana (?:besar )?masuk",
       r"positif", r"optimistis", r"layak", r"rebound", r"angin segar", r"melesat",
       r"untung", r"pimpin", r"penopang", r"bangkit", r"jagokan"]

NEG = [r"turun", r"anjlok", r"merosot", r"rugi", r"melemah", r"koreksi", r"tertekan",
       r"tekanan", r"jual", r"net sell", r"negatif", r"dihantui", r"risiko", r"ambles",
       r"longsor", r"hengkang", r"pesimistis", r"bantah", r"gagal", r"terjun"]


def lexicon_score(text: str) -> int:
    t = str(text).lower()
    pos = sum(bool(re.search(rf"\b{p}", t)) for p in POS)
    neg = sum(bool(re.search(rf"\b{p}", t)) for p in NEG)
    return pos - neg