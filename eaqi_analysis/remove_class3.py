import pandas as pd
from pathlib import Path
import os

# Eingabedatei (anpassen falls nötig)
INPUT_FILE = Path("ml_ready_eaqi.csv")
OUTPUT_FILE = Path("ml_ready_eaqi_no3.csv")

# CSV einlesen
path = os.path.abspath(os.path.join(os.path.dirname(__file__),))
df = pd.read_csv(path + "/ml_ready_eaqi.csv", parse_dates=["ts"])

print("Vorher:", df["eaqi_partial_class"].value_counts().to_dict())

# Zeilen mit Klasse 3 löschen
df = df[df["eaqi_partial_class"] != 3]

if "eco2_k" in df.columns:
    df = df.drop(columns=["eco2_k"])
    print("Spalte 'eco2_k' entfernt.")

print("Nachher:", df["eaqi_partial_class"].value_counts().to_dict())

# Neue CSV speichern
df.to_csv(OUTPUT_FILE, index=False)

print(f"[OK] Gespeichert unter: {OUTPUT_FILE.resolve()}")