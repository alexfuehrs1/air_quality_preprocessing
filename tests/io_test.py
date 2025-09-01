# test_load_csv.py
import os, sys
import pandas as pd
import pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from context import DataContext
from steps.io import LoadCSV

def _write_csv(path, df):
    df.to_csv(path, index=False)

def test_loadcsv_reads_sets_index_and_asfreq_inserts_ticks(tmp_path):
    # 3 Zeilen mit 1-Min-Takt, aber mit Lücke zwischen t1 und t3
    ts = pd.to_datetime(["2025-01-01 00:00:00",
                         "2025-01-01 00:01:00",
                         "2025-01-01 00:03:00"])
    df = pd.DataFrame({"timestamp": ts, "pm10": [10.0, 11.0, 13.0]})
    csv = tmp_path / "air.csv"
    _write_csv(csv, df)

    ctx = DataContext(csv_file=str(csv), tz=None)
    LoadCSV(parse_dates_col="timestamp", target_freq="1min").apply(ctx)

    # Index ist DatetimeIndex, 1-Minuten-Raster
    assert isinstance(ctx.df.index, pd.DatetimeIndex)
    assert ctx.df.index.freqstr == "T" or ctx.df.index.freq == pd.tseries.frequencies.to_offset("1min")
    # Es gibt eine eingefügte Zeile (00:02) mit NaN
    assert pd.Timestamp("2025-01-01 00:02:00") in ctx.df.index
    assert pd.isna(ctx.df.loc["2025-01-01 00:02:00", "pm10"])

def test_loadcsv_localizes_naive_to_tz_and_handles_nonexistent_dst(tmp_path):
    # Europe/Berlin: 2025-03-30 02:30 existiert NICHT -> wird auf 03:00 vorgezogen
    ts = pd.to_datetime(["2025-03-30 01:59:00",
                         "2025-03-30 02:30:00",   # nonexistent
                         "2025-03-30 03:05:00"])
    df = pd.DataFrame({"timestamp": ts, "pm2.5": [5.0, 6.0, 7.0]})
    csv = tmp_path / "dst.csv"
    df.to_csv(csv, index=False)

    ctx = DataContext(csv_file=str(csv), tz="Europe/Berlin")
    LoadCSV(parse_dates_col="timestamp", target_freq="1min").apply(ctx)

    idx = ctx.df.index
    # tz-aware
    assert idx.tz is not None and str(idx.tz) in ("Europe/Berlin", "CET")

    # Der 02:30-Punkt wurde auf 03:00 verschoben (nächstgültige Zeit)
    t_shifted = pd.Timestamp("2025-03-30 03:00:00", tz="Europe/Berlin")
    assert t_shifted in idx

    # Wert (6.0) des 02:30-Messpunkts steckt nun bei 03:00
    assert ctx.df.loc[t_shifted, "pm2.5"] == 6.0

    # (Optional) Es gibt keine 02:XX-Zeiten nach 02:00 am DST-Tag
    assert not any(ts.hour == 2 for ts in idx)

def test_loadcsv_converts_aware_utc_to_ctx_tz(tmp_path):
    # Schreibe UTC-Zeitstempel; LoadCSV soll nach Europe/Berlin konvertieren
    ts_utc = pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:01:00"], utc=True)
    df = pd.DataFrame({"timestamp": ts_utc, "eco2": [400, 410]})
    csv = tmp_path / "utc.csv"
    _write_csv(csv, df)

    ctx = DataContext(csv_file=str(csv), tz="Europe/Berlin")
    LoadCSV(parse_dates_col="timestamp", target_freq="1min").apply(ctx)

    idx = ctx.df.index
    assert str(idx.tz) in ("Europe/Berlin", "CET")
    # 00:00 UTC == 01:00 Berlin (Winterzeit)
    assert pd.Timestamp("2025-01-01 01:00:00", tz="Europe/Berlin") == idx[0]

def test_loadcsv_drops_nat_and_resolves_duplicates_keep_last(tmp_path):
    # Baue eine NaT-Zeile (leerer Timestamp) und doppelte Timestamps
    rows = [
        {"timestamp": "2025-01-01 00:00:00", "pm10": 10.0},
        {"timestamp": "",                     "pm10": 11.0},  # NaT
        {"timestamp": "2025-01-01 00:01:00", "pm10": 12.0},
        {"timestamp": "2025-01-01 00:01:00", "pm10": 99.0},  # Duplikat; letzte gewinnt
    ]
    df = pd.DataFrame(rows)
    csv = tmp_path / "dups.csv"
    _write_csv(csv, df)

    ctx = DataContext(csv_file=str(csv), tz=None)
    LoadCSV(parse_dates_col="timestamp", target_freq=None).apply(ctx)

    # NaT-Zeile entfernt
    assert not ctx.df.index.isna().any()
    # Duplikate aufgelöst -> nur ein Eintrag für 00:01, und der Wert ist die „letzte Beobachtung“
    assert ctx.df.index.duplicated().sum() == 0
    assert ctx.df.loc[pd.Timestamp("2025-01-01 00:01:00"), "pm10"] == 99.0

def test_loadcsv_sorts_non_monotonic_index(tmp_path):
    # Timestamps absichtlich unsortiert
    ts = pd.to_datetime(["2025-01-01 00:02:00",
                         "2025-01-01 00:00:00",
                         "2025-01-01 00:01:00"])
    df = pd.DataFrame({"timestamp": ts, "pm10": [12.0, 10.0, 11.0]})
    csv = tmp_path / "unsorted.csv"
    _write_csv(csv, df)

    ctx = DataContext(csv_file=str(csv), tz=None)
    LoadCSV(parse_dates_col="timestamp", target_freq="1min").apply(ctx)

    # Nach apply() ist Index monoton steigend
    assert ctx.df.index.is_monotonic_increasing
    # und alle drei Minuten sind vorhanden (durch asfreq)
    assert len(ctx.df) == 3

def test_loadcsv_raises_when_timestamp_column_missing(tmp_path):
    df = pd.DataFrame({"time": pd.date_range("2025-01-01", periods=3, freq="min"),
                       "pm10": [1.0, 2.0, 3.0]})
    csv = tmp_path / "missing_ts.csv"
    _write_csv(csv, df)

    ctx = DataContext(csv_file=str(csv), tz=None)
    with pytest.raises(ValueError):
        LoadCSV(parse_dates_col="timestamp", target_freq="1min").apply(ctx)

def test_loadcsv_target_freq_none_keeps_original_rows(tmp_path):
    ts = pd.to_datetime(["2025-01-01 00:00:00",
                         "2025-01-01 00:02:00"])
    df = pd.DataFrame({"timestamp": ts, "pm10": [10.0, 12.0]})
    csv = tmp_path / "nofreq.csv"
    _write_csv(csv, df)

    ctx = DataContext(csv_file=str(csv), tz=None)
    LoadCSV(parse_dates_col="timestamp", target_freq=None).apply(ctx)

    # Keine Reindizierung → genau 2 Zeilen
    assert len(ctx.df) == 2
    assert list(ctx.df.index) == list(pd.to_datetime(ts))
