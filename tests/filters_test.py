import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from context import DataContext
from steps.filters import HampelFilter, RateOfChangeCheck, FlatlineCheck
from steps.cleaning import IQRRemoveOutliers


def _minute_index(n: int = 60, start: str = "2025-01-01 00:00:00", freq: str = "1min"):
    """
    Hilfsfunktion für einen DatetimeIndex.
    - n: Anzahl Perioden
    - start: Startzeit als String
    - freq: Frequenz (default "1min", kann z.B. "5min" sein)
    """
    return pd.date_range(start=start, periods=n, freq=freq)


def test_hampel_replace_nan_marks_spike_and_sets_flag():
    idx = _minute_index(60)
    base = np.ones(60) * 10.0
    base[30] = 200.0  # Spike in der Mitte
    df = pd.DataFrame({"pm10": base}, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    HampelFilter(cols=("pm10",), window=11, n_sigmas=3.0,
                 strategy="replace_nan", center=True, add_flag=True).apply(ctx)

    # Spike wird NaN
    assert np.isnan(ctx.df.loc[idx[30], "pm10"]), "Spike sollte als NaN markiert werden"
    # Flag existiert & ist am Spike True
    assert "pm10_hampel_flag" in ctx.df.columns
    assert bool(ctx.df.loc[idx[30], "pm10_hampel_flag"]) is True
    # Nachbarn bleiben numerisch (keine großflächige Löschung)
    assert ctx.df.loc[idx[29], "pm10"] == 10.0
    assert ctx.df.loc[idx[31], "pm10"] == 10.0


def test_hampel_cap_limits_spike_no_nans():
    idx = _minute_index(60)
    base = np.ones(60) * 10.0
    base[30] = 200.0
    df = pd.DataFrame({"pm10": base}, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    # cap mit center=True: med ≈ 10, MAD ≈ 0 -> sigma = eps -> cap in der Nähe von 10
    hf = HampelFilter(cols=("pm10",), window=11, n_sigmas=3.0,
                      strategy="cap", center=True, add_flag=True, eps=1e-9)
    hf.apply(ctx)

    # Keine NaNs entstanden
    assert ctx.df["pm10"].isna().sum() == 0, "cap darf keine NaNs erzeugen"
    # Der gescapte Spike ist ~10 (auf [med ± n_sigmas*sigma] gekappt)
    assert np.isclose(ctx.df.loc[idx[30], "pm10"], 10.0, atol=1e-6)
    # Flag gesetzt
    assert bool(ctx.df.loc[idx[30], "pm10_hampel_flag"]) is True


def test_hampel_eps_guard_on_constant_series_no_flags_no_nans():
    idx = _minute_index(40)
    df = pd.DataFrame({"pm10": np.full(40, 10.0)}, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    HampelFilter(cols=("pm10",), window=11, n_sigmas=3.0,
                 strategy="replace_nan", center=True, add_flag=True, eps=1e-9).apply(ctx)

    # Keine NaNs, da keine Ausreißer in konstanter Serie
    assert ctx.df["pm10"].isna().sum() == 0
    # Flag existiert, aber überall False
    assert "pm10_hampel_flag" in ctx.df.columns
    assert not ctx.df["pm10_hampel_flag"].any()


def test_hampel_excludes_bool_and_existing_flag_columns():
    idx = _minute_index(20)
    df = pd.DataFrame({
        "pm10": np.r_[np.ones(10)*10.0, np.ones(10)*10.0],
        "is_on": [True]*20,                   # bool-Spalte
        "pm10_hampel_flag": [False]*20        # existierende Flag-Spalte
    }, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    HampelFilter(cols=None, window=9, n_sigmas=3.0,
                 strategy="replace_nan", center=True, add_flag=True).apply(ctx)

    # Bool-Spalte bleibt bool, wurde nicht angerührt
    assert ctx.df["is_on"].dtype == bool
    # Flag-Spalte bleibt bool
    assert ctx.df["pm10_hampel_flag"].dtype == bool
    # pm10 bleibt unverändert (keine Spikes)
    assert ctx.df["pm10"].isna().sum() == 0


def test_hampel_center_false_also_detects_spike_trailing():
    """
    Mit center=False (trailing/kausal) soll der Spike ebenfalls erkannt werden.
    """
    idx = _minute_index(60)
    base = np.ones(60) * 10.0
    base[30] = 200.0
    df = pd.DataFrame({"pm10": base}, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    HampelFilter(cols=("pm10",), window=11, n_sigmas=3.0,
                 strategy="replace_nan", center=False, add_flag=True).apply(ctx)

    # Auch kausal wird der Spike entfernt
    assert np.isnan(ctx.df.loc[idx[30], "pm10"]), "Spike sollte auch mit center=False entfernt werden"
    # Flag gesetzt
    assert "pm10_hampel_flag" in ctx.df.columns
    assert bool(ctx.df.loc[idx[30], "pm10_hampel_flag"]) is True


def test_hampel_add_flag_false_creates_no_flag_column():
    idx = _minute_index(60)
    base = np.ones(60) * 10.0
    base[30] = 200.0
    df = pd.DataFrame({"pm10": base}, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    HampelFilter(cols=("pm10",), window=11, n_sigmas=3.0,
                 strategy="replace_nan", center=True, add_flag=False).apply(ctx)

    assert "pm10_hampel_flag" not in ctx.df.columns, "add_flag=False -> keine Flag-Spalte erzeugen"
    assert np.isnan(ctx.df.loc[idx[30], "pm10"])


def test_replace_nan_for_pm10_large_jump_sets_nan_and_flag():
    idx = _minute_index(20)
    s = np.ones(20) * 10.0
    s[10] = 300.0  # harter Sprung
    df = pd.DataFrame({"pm10": s}, index=idx)
    ctx = DataContext(csv_file="", df=df)

    roc = RateOfChangeCheck(
        cols=("pm10",),
        # verwenden die Defaults aus der Klasse ODER explizit setzen:
        max_abs_per_min={"pm10": 80.0},
        max_rel_per_min={"pm10": 1.0},
        rel_level_floor={"pm10": 10.0},
        strategy={"pm10": "replace_nan"}
    )
    roc.apply(ctx)

    assert np.isnan(ctx.df.loc[idx[10], "pm10"]), "Großer RoC-Sprung muss als NaN markiert werden."
    assert bool(ctx.df.loc[idx[10], "pm10_roc_flag"]) is True, "Flag muss gesetzt werden."
    # Erste Zeile niemals flaggen (kein dt)
    assert not bool(ctx.df.loc[idx[0], "pm10_roc_flag"]), "Erste Zeile darf nicht geflaggt werden."


def test_cap_strategy_uses_min_of_abs_and_rel_thresholds_and_dt():
    # 1-Min-Takt, ECO2: prev=600 → nächster Messwert 2000 (Δ=+1400)
    # abs_thr=500 ppm/min, rel_thr=0.5 (=50% von prev=600 -> 300 ppm/min)
    # erlaubte Änderung pro Minute = min(500, 300) = 300 → capped = 600 + 300 = 900
    idx = _minute_index(3)
    df = pd.DataFrame({"ECO2": [600.0, 2000.0, 2000.0]}, index=idx)
    ctx = DataContext(csv_file="", df=df)

    roc = RateOfChangeCheck(
        cols=("ECO2",),
        max_abs_per_min={"ECO2": 500.0},
        max_rel_per_min={"ECO2": 0.5},
        rel_level_floor={"ECO2": 400.0},
        strategy={"ECO2": "cap"}
    )
    roc.apply(ctx)

    assert np.isclose(ctx.df.loc[idx[1], "ECO2"], 900.0, atol=1e-9), "Capping muss auf 900 ppm begrenzen."
    assert bool(ctx.df.loc[idx[1], "ECO2_roc_flag"]) is True, "Flag muss beim Capping gesetzt sein."


def test_cap_respects_dt_minutes_greater_than_one():
    # 5-Minuten-Takt, prev=600 → 2000. dt=5
    # rate = 280 ppm/min < min(500, 300) => kein Verstoß, kein Capping, kein Flag
    idx = _minute_index(3, start="2025-01-01 00:00:00", freq="5min")
    df = pd.DataFrame({"ECO2": [600.0, 2000.0, 2000.0]}, index=idx)
    ctx = DataContext(csv_file="", df=df)

    roc = RateOfChangeCheck(
        cols=("ECO2",),
        max_abs_per_min={"ECO2": 500.0},
        max_rel_per_min={"ECO2": 0.5},
        rel_level_floor={"ECO2": 400.0},
        strategy={"ECO2": "cap"}
    )
    roc.apply(ctx)

    # kein Capping
    assert np.isclose(ctx.df.loc[idx[1], "ECO2"], 2000.0, atol=1e-9), \
        "Bei großem dt darf hier nicht gecappt werden."
    # und KEIN Flag
    assert not bool(ctx.df.loc[idx[1], "ECO2_roc_flag"]), \
        "Ohne Regelverstoß darf kein Flag gesetzt sein."


def test_flag_only_keeps_values_but_sets_flag_for_pressure():
    idx = _minute_index(4)
    # Unrealistischer Sprung im Druck: 1013 → 1019 hPa in 1 Minute (Δ=+6 hPa)
    df = pd.DataFrame({"Pressure": [1013.0, 1019.0, 1019.0, 1019.0]}, index=idx)
    ctx = DataContext(csv_file="", df=df)

    roc = RateOfChangeCheck(
        cols=("Pressure",),
        max_abs_per_min={"Pressure": 1.0},  # streng
        strategy={"Pressure": "flag_only"}
    )
    roc.apply(ctx)

    # Wert bleibt erhalten
    assert ctx.df.loc[idx[1], "Pressure"] == 1019.0, "flag_only darf Werte nicht verändern."
    # Flag gesetzt
    assert bool(ctx.df.loc[idx[1], "Pressure_roc_flag"]) is True, "Flag muss gesetzt werden."


def test_relative_rule_ignores_small_levels_with_floor():
    idx = _minute_index(3)
    # Mini-Level: 1 → 3 (200%) in 1 Minute soll NICHT als Verstoß zählen dank floor=5
    df = pd.DataFrame({"pm2.5": [1.0, 3.0, 3.0]}, index=idx)
    ctx = DataContext(csv_file="", df=df)

    roc = RateOfChangeCheck(
        cols=("pm2.5",),
        max_abs_per_min={"pm2.5": 9999.0},   # abs-Rule aus dem Spiel
        max_rel_per_min={"pm2.5": 1.0},      # 100%/min
        rel_level_floor={"pm2.5": 5.0},      # Floor schützt Miniwerte
        strategy={"pm2.5": "flag_only"}
    )
    roc.apply(ctx)

    assert not bool(ctx.df.loc[idx[1], "pm2.5_roc_flag"]), "Floor sollte false positives der relativen Regel verhindern."
    # Werte bleiben erhalten
    assert ctx.df.loc[idx[1], "pm2.5"] == 3.0


def test_non_monotonic_index_is_sorted_and_processed():
    # Erzeuge absichtlich nicht-monotone Reihenfolge
    idx = _minute_index(4)
    df = pd.DataFrame({"pm10": [10.0, 10.0, 300.0, 300.0]}, index=[idx[0], idx[2], idx[1], idx[3]])
    ctx = DataContext(csv_file="", df=df)

    roc = RateOfChangeCheck(
        cols=("pm10",),
        max_abs_per_min={"pm10": 80.0},
        max_rel_per_min={"pm10": 1.0},
        rel_level_floor={"pm10": 10.0},
        strategy={"pm10": "replace_nan"}
    )
    roc.apply(ctx)

    # Nach Sortierung ist der große Sprung zwischen idx[1] und idx[2]
    assert np.isnan(ctx.df.loc[idx[2], "pm10"]), "Nach Sortierung muss der Sprung korrekt als NaN markiert werden."
    assert bool(ctx.df.loc[idx[2], "pm10_roc_flag"]) is True


def test_duplicate_timestamps_do_not_crash_and_first_row_not_flagged():
    # Duplikate im Index erzeugen dt_min=0 -> wird NaN, darf nicht crashen
    base = pd.Timestamp("2025-01-01 00:00:00")
    idx = pd.DatetimeIndex([base, base, base + pd.Timedelta(minutes=1)])
    df = pd.DataFrame({"pm10": [10.0, 200.0, 200.0]}, index=idx)
    ctx = DataContext(csv_file="", df=df)

    roc = RateOfChangeCheck(
        cols=("pm10",),
        max_abs_per_min={"pm10": 80.0},
        max_rel_per_min={"pm10": 1.0},
        rel_level_floor={"pm10": 10.0},
        strategy={"pm10": "replace_nan"}
    )
    roc.apply(ctx)

    # Erste Zeile: kein Flag
    assert not bool(ctx.df.iloc[0]["pm10_roc_flag"]), "Erste Zeile darf nicht geflaggt werden."
    # Zweite Zeile: dt=0 -> mask sollte nicht crashen; je nach Implementierung keine Änderung erzwingen
    assert "pm10" in ctx.df.columns, "Spalte muss existieren."
    assert "pm10_roc_flag" in ctx.df.columns, "Flag-Spalte muss existieren."


def test_no_thresholds_defined_means_column_is_ignored():
    idx = _minute_index(3)
    df = pd.DataFrame({"UnknownSensor": [1.0, 1000.0, 1000.0]}, index=idx)
    ctx = DataContext(csv_file="", df=df)

    roc = RateOfChangeCheck(
        cols=("UnknownSensor",),
        # keine thresholds gesetzt
        strategy={"UnknownSensor": "replace_nan"}
    )
    roc.apply(ctx)

    # Keine Flag-Spalte, keine Änderungen erwartet
    assert "UnknownSensor_roc_flag" not in ctx.df.columns, "Ohne Thresholds sollte der Kanal ignoriert werden."
    pd.testing.assert_series_equal(ctx.df["UnknownSensor"], df["UnknownSensor"])


def test_flatline_replace_nan_marks_constant_run():
    """
    Ein längeres konstantes Segment (>= min_duration) soll als Flatline erkannt
    und bei strategy='replace_nan' zu NaN werden (zumindest im Innenbereich).
    """
    idx = _minute_index(60, freq="1min")
    # 0..29 konstant 10.0, danach linear steigend (kein Flat)
    y = np.arange(60, dtype=float)
    y[:30] = 10.0
    df = pd.DataFrame({"pm10": y}, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    # min_duration=10min -> bei 1min-Frequenz ~10 Punkte Fensterbreite
    FlatlineCheck(cols=("pm10",), min_duration_minutes=10, tolerance=1e-6,
                  strategy="replace_nan", add_flag=True).apply(ctx)

    # Innenbereich des konstanten Abschnitts sicher maskiert
    assert ctx.df.loc[idx[12:25], "pm10"].isna().all(), "Flatline-Innenbereich sollte NaN sein"
    # Flag vorhanden und im konstanten Bereich gesetzt
    assert "pm10_flatline_flag" in ctx.df.columns
    assert ctx.df.loc[idx[12:25], "pm10_flatline_flag"].all()


def test_flatline_freeze_first_keeps_values_no_nans():
    """
    Bei 'freeze_first' werden Flatline-Segmente auf den (linken) Fensterwert gesetzt.
    Es dürfen dabei KEINE NaNs entstehen; Flag soll gesetzt sein.
    """
    idx = _minute_index(60, freq="1min")
    y = np.arange(60, dtype=float)
    y[:30] = 10.0  # langes Flatline-Segment
    df = pd.DataFrame({"pm10": y}, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    FlatlineCheck(cols=("pm10",), min_duration_minutes=10, tolerance=1e-6,
                  strategy="freeze_first", add_flag=True).apply(ctx)

    # Keine NaNs durch freeze_first
    assert ctx.df["pm10"].isna().sum() == 0, "freeze_first darf keine NaNs erzeugen"
    # Im Innenbereich bleibt Wert 10.0
    assert (ctx.df.loc[idx[12:25], "pm10"] == 10.0).all()
    # Flag vorhanden
    assert "pm10_flatline_flag" in ctx.df.columns
    assert ctx.df.loc[idx[12:25], "pm10_flatline_flag"].all()


def test_flatline_respects_minutes_to_points_with_5min_freq():
    """
    Prüft die Minuten→Punkte-Umrechnung:
      - 5min-Frequenz, min_duration=30min => Fensterbreite ~6 Punkte.
      - Ein kurzes konstantes Segment (4 Punkte = 20min) darf NICHT geflaggt werden.
      - Ein langes Segment (8 Punkte = 40min) MUSS geflaggt werden.
    """
    idx = _minute_index(20, freq="5min")
    y = np.full(20, 12.0, dtype=float)

    # Segment A: 0..3 (4 Punkte = 20min) konstant -> zu kurz
    # Lücke: 4..5 variabel
    y[4] = 13.0; y[5] = 14.0
    # Segment B: 6..13 (8 Punkte = 40min) konstant -> lang genug
    y[6:14] = 10.0
    # Rest variabel
    y[14:] = np.linspace(11.0, 16.0, 6)

    df = pd.DataFrame({"pm10": y}, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    FlatlineCheck(cols=("pm10",), min_duration_minutes=30, tolerance=1e-6,
                  strategy="replace_nan", add_flag=True).apply(ctx)

    # Kurzes Segment A (Innenbereich z.B. Index 1..2) darf nicht NaN sein
    assert ctx.df.loc[idx[1:2], "pm10"].notna().all(), "kurzes 20min-Segment darf nicht geflaggt werden"

    # Langes Segment B (Innenbereich z.B. 8..11) muss NaN sein
    assert ctx.df.loc[idx[8:11], "pm10"].isna().all(), "langes 40min-Segment muss als Flatline markiert werden"
    # Flag in Segment B gesetzt
    assert ctx.df.loc[idx[8:11], "pm10_flatline_flag"].all()


def test_flatline_ignores_bool_and_flag_columns():
    """
    Bool- und *_flag-Spalten dürfen nicht durch den FlatlineCheck gelaufen werden.
    """
    idx = _minute_index(12, freq="1min")
    df = pd.DataFrame({
        "pm10": [10.0]*6 + [11.0]*6,          # kein Flatline-Segment >= min_duration
        "pm10_roc_flag": [False]*12,          # bereits vorhandenes Flag
        "is_on": [True]*12                    # bool-Spalte
    }, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    FlatlineCheck(cols=None, min_duration_minutes=10, tolerance=1e-6,
                  strategy="replace_nan", add_flag=True).apply(ctx)

    # Bool/Flag-Spalten unverändert & bool
    assert "pm10_roc_flag" in ctx.df.columns and ctx.df["pm10_roc_flag"].dtype == bool
    assert "is_on" in ctx.df.columns and ctx.df["is_on"].dtype == bool


def test_flatline_add_flag_false_creates_no_flag_column():
    """
    Wenn add_flag=False gesetzt ist, soll KEINE *_flatline_flag-Spalte entstehen.
    """
    idx = _minute_index(40, freq="1min")
    y = np.full(40, 10.0, dtype=float)  # langes Flatline-Segment
    df = pd.DataFrame({"pm10": y}, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    FlatlineCheck(cols=("pm10",), min_duration_minutes=10, tolerance=1e-6,
                  strategy="replace_nan", add_flag=False).apply(ctx)

    assert "pm10_flatline_flag" not in ctx.df.columns, "add_flag=False darf keine Flag-Spalte erzeugen"
    # Trotzdem sollte Flatline ersetzt worden sein (Innenbereich)
    assert ctx.df.loc[idx[12:25], "pm10"].isna().all()

def test_iqr_remove_outliers_marks_value():
    idx = _minute_index(50)
    s = np.ones(50) * 10.0
    s[5] = 1000.0  # krasser Ausreißer
    df = pd.DataFrame({"pm10": s}, index=idx)
    ctx = DataContext(csv_file="", df=df)

    IQRRemoveOutliers(cols=("pm10",), iqr_factor=3.0).apply(ctx)

    assert np.isnan(ctx.df.loc[idx[5], "pm10"]), "IQR-Schritt sollte extremen Ausreißer entfernen"
