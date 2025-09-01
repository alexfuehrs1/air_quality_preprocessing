import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

# =========================
# KONFIGURATION
# =========================
INPUT_PATH      = Path("merged_labeled.csv")       # Eingabedatei
TARGET          = "eaqi_partial_class"             # Klassenziel
OUTPUT_BASE     = Path("ml_ready_eaqi")            # Basisname für Exporte
UNDERSAMPLE_MAX = None
USE_SMOTE       = False
TOP_K_FEATURES  = 25

# =========================
# FEATURES
# =========================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)

    # Rohsensoren (achte auf Spaltennamen in deinen CSVs)
    colmap = {
        "bmp_temperature": "bmp_temp",
        "dht_temperature": "dht_temp",
    }
    base = df.rename(columns=colmap)
    for c in ["eco2","tvoc","humidity","pressure","bmp_temp","dht_temp"]:
        if c in base.columns:
            X[c] = base[c]

    # Ableitungen
    if "tvoc" in X:
        X["log_tvoc"] = np.log1p(X["tvoc"].clip(lower=0))
    if "bmp_temp" in X and "dht_temp" in X:
        X["temp_diff"] = X["bmp_temp"] - X["dht_temp"]
    if "dht_temp" in X and "humidity" in X:
        T  = X["dht_temp"]
        RH = X["humidity"]
        es = 6.112 * np.exp((17.62 * T) / (243.12 + T))
        X["abs_humidity"] = (2.1674 * (RH/100.0) * es) / (273.15 + T) * 1000.0

    # Zeitfeatures – robust für MultiIndex
    ts_idx = _extract_ts_index(df)  # -> DatetimeIndex
    X["minute"]   = ts_idx.minute
    X["hour"]     = ts_idx.hour
    X["dow"]      = ts_idx.dayofweek
    X["month"]    = ts_idx.month
    X["hour_sin"] = np.sin(2*np.pi*X["hour"]/24.0)
    X["hour_cos"] = np.cos(2*np.pi*X["hour"]/24.0)
    X["dow_sin"]  = np.sin(2*np.pi*X["dow"]/7.0)
    X["dow_cos"]  = np.cos(2*np.pi*X["dow"]/7.0)

    return X

def _extract_ts_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Liefert einen DatetimeIndex mit den Zeitstempeln – funktioniert für
    - Single DatetimeIndex
    - MultiIndex mit Ebene 'ts' (oder erster datetime-artiger Ebene)
    - normalen Index + 'ts'-Spalte
    """
    # 1) MultiIndex?
    if isinstance(df.index, pd.MultiIndex):
        # a) Ebene 'ts' vorhanden?
        if "ts" in df.index.names:
            ts = df.index.get_level_values("ts")
            return pd.DatetimeIndex(pd.to_datetime(ts, errors="coerce"))
        # b) erste datetime-artige Ebene finden
        for i in range(df.index.nlevels):
            lvl_vals = df.index.get_level_values(i)
            if np.issubdtype(lvl_vals.dtype, np.datetime64):
                return pd.DatetimeIndex(lvl_vals)
        # c) Fallback: 'ts'-Spalte
        if "ts" in df.columns:
            return pd.DatetimeIndex(pd.to_datetime(df["ts"], errors="coerce"))

    # 2) Single Index?
    else:
        if np.issubdtype(df.index.dtype, np.datetime64):
            return pd.DatetimeIndex(df.index)
        if "ts" in df.columns:
            return pd.DatetimeIndex(pd.to_datetime(df["ts"], errors="coerce"))

    raise ValueError("Konnte keinen Zeitstempel finden (weder DatetimeIndex noch 'ts'-Ebene/Spalte).")

def load_and_merge_all() -> pd.DataFrame:
    base_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "air_quality_preprocessing", "analysis", "eaqi_analysis")
    )
    dfs = []
    subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    for folder in subfolders:
        csv_path = os.path.join(base_path, folder, "labeled_clean_measurements.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["source_folder"] = folder             # <--- wichtig
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
            dfs.append(df)

    if not dfs:
        print("[ERROR] Keine gültigen DataFrames gefunden.")
        return pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values("ts")
    # MultiIndex aufsetzen (ts, source_folder)
    merged = merged.set_index(["ts", "source_folder"]).sort_index()
    print(f"\n[MERGED] Gesamt: {len(merged)} Zeilen aus {len(dfs)} Ordnern")
    return merged

def save_ml_ready(df: pd.DataFrame, output_file: Path):
    # 1) Target vorhanden?
    if TARGET not in df.columns:
        raise ValueError(f"Target-Spalte '{TARGET}' fehlt.")

    # 2) Zeilen mit gültigem Target **auf Zeilenebene** filtern
    mask = df[TARGET].notna()
    df = df.loc[mask].copy()

    # 3) Target casten
    y = df[TARGET].astype("Float64").astype(int)

    # 4) Features bauen (ohne pm2_5 & pm10)
    X = engineer_features(df.drop(columns=["pm2_5","pm10"], errors="ignore"))

    # 5) Export – per Index ausrichten (keine nackten .values!)
    out_full = X.copy()
    out_full[TARGET] = y  # richtet automatisch per Index aus
    out_full = out_full[[TARGET] + [c for c in X.columns]]  # Target nach vorne

    out_file = OUTPUT_BASE.with_suffix(".csv")
    out_full.to_csv(out_file, index_label="ts")

    print(f"[OK] Export: {out_file}  | Zeilen: {len(out_full)} | Features: {X.shape[1]}")

# =========================
# MAIN PIPELINE
# =========================
def main():
    df = load_and_merge_all()
    save_ml_ready(df, OUTPUT_BASE)


if __name__ == "__main__":
    main()
