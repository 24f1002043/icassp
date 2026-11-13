from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FEAT = ROOT / "data" / "features"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
for d in (PROC, FEAT, TABLES, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

EMOTIONS = ["A", "D", "F", "H", "N", "S"]
EMO_NAME = {"A": "Anger", "D": "Disgust", "F": "Fear",
            "H": "Happy", "N": "Neutral", "S": "Sad"}
FILE_EMO = {"ANG": "A", "DIS": "D", "FEA": "F",
            "HAP": "H", "NEU": "N", "SAD": "S"}
QUERY_TYPE = {1: "voice", 2: "face", 3: "multimodal"}


def parse_filename(name):
    actor, sentence, emo, level = name.split("_")
    return int(actor), sentence, FILE_EMO[emo], level
