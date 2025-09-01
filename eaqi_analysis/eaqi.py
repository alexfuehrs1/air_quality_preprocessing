import pandas as pd
import os, numpy as np

# Kombination (schlechtester Wert aus pm2_5 & pm10)
def worst_class(row):
    pm25 = row["pm2_5"] if "pm2_5" in row and pd.notna(row["pm2_5"]) else None
    pm10 = row["pm10"]  if "pm10" in row and pd.notna(row["pm10"]) else None
    res = calc_partial_eaqi(pm10, pm25)
    return res[2] if res else pd.NA

def add_EAQI_label(df: pd.DataFrame, folder: str, sep: str = ",") -> None:
    """
    Berechnet die EAQI-Gesamtklasse (worst case aus PM2.5 und PM10)
    und speichert eine neue CSV mit nur dieser Spalte zusätzlich.
    """

    # sicherstellen, dass die PM-Spalten numerisch sind
    if "pm2_5" in df.columns:
        df["pm2_5"] = pd.to_numeric(df["pm2_5"], errors="coerce")
    if "pm10" in df.columns:
        df["pm10"] = pd.to_numeric(df["pm10"], errors="coerce")

    df["eaqi_partial_class"] = df.apply(worst_class, axis=1)

    # Ausgabeordner vorbereiten
    base_out = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "air_quality_preprocessing", "analysis", "eaqi_analysis")
    )
    output_dir = os.path.join(base_out, folder)
    os.makedirs(output_dir, exist_ok=True)

    # Ziel-Dateipfad
    output_csv = os.path.join(output_dir, "labeled_clean_measurements.csv")

    # CSV speichern (nur mit der neuen Spalte zusätzlich)
    df.to_csv(output_csv, sep=sep, index=False)
    print(f"Fertig! Neue Datei gespeichert: {output_csv}")

def get_eaqi_pm10(value):
    if 0 <= value <= 20:
        return 1, "Good"
    elif 20 < value <= 40:
        return 2, "Fair"
    elif 40 < value <= 50:
        return 3, "Moderate"
    elif 50 < value <= 100:
        return 4, "Poor"
    elif 100 < value <= 150:
        return 5, "Very poor"
    else:  # >150
        return 6, "Extremely poor"


def get_eaqi_pm25(value):
    if 0 <= value <= 10:
        return 1, "Good"
    elif 10 < value <= 20:
        return 2, "Fair"
    elif 20 < value <= 25:
        return 3, "Moderate"
    elif 25 < value <= 50:
        return 4, "Poor"
    elif 50 < value <= 75:
        return 5, "Very poor"
    else:  
        return 6, "Extremely poor"

def calc_partial_eaqi(pm10_24h=None, pm25_24h=None):
    results = []
    if pm10_24h is not None:
        idx, cat = get_eaqi_pm10(pm10_24h)
        results.append(("PM10", pm10_24h, idx, cat))
    if pm25_24h is not None:
        idx, cat = get_eaqi_pm25(pm25_24h)
        results.append(("PM2.5", pm25_24h, idx, cat))
    if not results:
        return None
    # wähle die schlechteste Klasse (höchster Index = schlechtere Luftqualität)
    worst = max(results, key=lambda x: x[2])
    return worst


output_csv = "labeled_measurments.csv"
sep = ","  

base_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "air_quality_preprocessing", "analysis", "out")
)

# alle Unterordner in "out"
subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

for folder in subfolders:
    csv_path = os.path.join(base_path, folder, "results_clean.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df.dropna(axis=0, how="any")
        add_EAQI_label(df, folder)


