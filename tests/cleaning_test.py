import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from context import DataContext
from steps.cleaning import RemoveInvalidValues, IQRRemoveOutliers

def _minute_index(n=50, start="2025-01-01 00:00:00", freq="1min"):
    return pd.date_range(start=start, periods=n, freq=freq)

def test_replace_nan_out_of_range_to_nan_and_bounds_inclusive():
    idx = _minute_index(6)
    df = pd.DataFrame({
        # limits (0, 2000)
        "pm10": [0.0, 10.0, 2000.0, -1.0, 2500.0, 100.0]
    }, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    RemoveInvalidValues(
        limits={"pm10": (0.0, 2000.0)}, strategy="replace_nan"
    ).apply(ctx)

    # In-Range inklusiv: 0 und 2000 bleiben erhalten
    assert ctx.df.loc[idx[0], "pm10"] == 0.0
    assert ctx.df.loc[idx[2], "pm10"] == 2000.0
    # Out-of-range -> NaN
    assert pd.isna(ctx.df.loc[idx[3], "pm10"])
    assert pd.isna(ctx.df.loc[idx[4], "pm10"])


def test_clip_strategy_caps_values_without_creating_nans():
    idx = _minute_index(5)
    df = pd.DataFrame({
        "pressure": [750.0, 800.0, 1050.0, 1200.0, 1100.0]  # gültig: 800..1100
    }, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    RemoveInvalidValues(
        limits={"pressure": (800.0, 1100.0)}, strategy="clip"
    ).apply(ctx)

    # Gekappt an die Grenzen, keine NaNs
    exp = [800.0, 800.0, 1050.0, 1100.0, 1100.0]
    assert ctx.df["pressure"].tolist() == exp
    assert ctx.df["pressure"].isna().sum() == 0


def test_flag_only_sets_flag_but_keeps_values():
    idx = _minute_index(5)
    df = pd.DataFrame({
        "humidity": [0.0, 50.0, 101.0, -2.0, 100.0]  # gültig: 0..100
    }, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    RemoveInvalidValues(
        limits={"humidity": (0.0, 100.0)},
        strategy="flag_only",
        add_flag=False  # flag_only erzwingt Flag trotzdem
    ).apply(ctx)

    # Werte unverändert
    pd.testing.assert_series_equal(ctx.df["humidity"], df["humidity"])
    # Flag existiert und ist an invaliden Stellen True
    assert "humidity_invalid_flag" in ctx.df.columns
    assert ctx.df["humidity_invalid_flag"].tolist() == [False, False, True, True, False]


def test_add_flag_true_with_replace_nan_marks_and_nans():
    idx = _minute_index(5)
    df = pd.DataFrame({
        "tvoc": [0.0, 10.0, 70000.0, 5.0, -1.0]  # gültig: 0..60000
    }, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    RemoveInvalidValues(
        limits={"tvoc": (0.0, 60000.0)},
        strategy="replace_nan",
        add_flag=True
    ).apply(ctx)

    # Out-of-range -> NaN
    assert pd.isna(ctx.df.loc[idx[2], "tvoc"])
    assert pd.isna(ctx.df.loc[idx[4], "tvoc"])
    # Flag vorhanden und korrekt gesetzt
    assert "tvoc_invalid_flag" in ctx.df.columns
    assert ctx.df["tvoc_invalid_flag"].tolist() == [False, False, True, False, True]


def test_coerce_numeric_true_parses_strings_and_applies_limits():
    idx = _minute_index(6)
    df = pd.DataFrame({
        "eco2": ["300", "500", "abc", "60000", "250", "50001"]  # gültig: 250..50000
    }, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    RemoveInvalidValues(
        limits={"eco2": (250, 50000)},
        strategy="replace_nan",
        coerce_numeric=True
    ).apply(ctx)

    # "300","500" ok; "abc" -> NaN durch Koerzierung; "60000" & "50001" -> NaN durch Grenzen; "250" ok
    out = ctx.df["eco2"].tolist()
    assert out[0] == 300.0
    assert out[1] == 500.0
    assert pd.isna(out[2])          # "abc" -> NaN via to_numeric
    assert pd.isna(out[3])          # >50000 -> NaN
    assert out[4] == 250.0          # Grenze inklusiv
    assert pd.isna(out[5])          # >50000 -> NaN


def test_coerce_numeric_false_skips_non_numeric_column():
    idx = _minute_index(4)
    df = pd.DataFrame({
        "eco2": ["300", "400", "500", "bad"]  # string dtype
    }, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    RemoveInvalidValues(
        limits={"eco2": (250, 50000)},
        strategy="replace_nan",
        coerce_numeric=False  # => nicht-numerische Spalte wird übersprungen
    ).apply(ctx)

    # Spalte unverändert (kein Coerce, kein Replace)
    pd.testing.assert_series_equal(ctx.df["eco2"], df["eco2"])


def test_case_insensitive_column_match_uppercase_column():
    idx = _minute_index(5)
    df = pd.DataFrame({
        "ECO2": [200.0, 300.0, 400.0, 51000.0, 500.0]  # gültig: 250..50000
    }, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    RemoveInvalidValues(
        limits={"eco2": (250.0, 50000.0)},
        strategy="replace_nan",
        case_insensitive=True
    ).apply(ctx)

    # Die Spalte "ECO2" wurde gefunden und invalid (200, 51000) zu NaN gesetzt.
    assert pd.isna(ctx.df.loc[idx[0], "ECO2"])
    assert pd.isna(ctx.df.loc[idx[3], "ECO2"])
    # In-range Werte bleiben
    assert ctx.df.loc[idx[1], "ECO2"] == 300.0
    assert ctx.df.loc[idx[2], "ECO2"] == 400.0
    assert ctx.df.loc[idx[4], "ECO2"] == 500.0


def test_missing_limit_keys_do_not_crash_and_other_cols_processed():
    idx = _minute_index(5)
    df = pd.DataFrame({
        "pm2.5": [5.0, 10.0, -1.0, 12.0, 8.0]  # gültig: 0..1000
    }, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    # 'pm10' in limits fehlt in df -> sollte ignoriert werden, 'pm2.5' verarbeitet werden
    RemoveInvalidValues(
        limits={"pm2.5": (0.0, 1000.0), "pm10": (0.0, 2000.0)},
        strategy="replace_nan"
    ).apply(ctx)

    # pm2.5[2] ist out-of-range -> NaN; kein Crash trotz fehlendem pm10
    assert pd.isna(ctx.df.loc[idx[2], "pm2.5"])

def test_replace_nan_marks_extreme_outlier_iqr_zero_case():
    """49x 10.0 + 1x 1000.0 -> IQR==0, Ausreißer wird NaN mit replace_nan."""
    idx = _minute_index(50)
    y = np.ones(50) * 10.0
    y[5] = 1000.0
    df = pd.DataFrame({"pm10": y}, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    IQRRemoveOutliers(cols=("pm10",), iqr_factor=3.0, strategy="replace_nan").apply(ctx)

    assert np.isnan(ctx.df.loc[idx[5], "pm10"]), "IQR-Schritt sollte extremen Ausreißer entfernen"
    # Nachbarn bleiben erhalten
    assert ctx.df.loc[idx[4], "pm10"] == 10.0
    assert ctx.df.loc[idx[6], "pm10"] == 10.0


def test_cap_strategy_winsorizes_outlier_no_nans():
    """Gleicher Datensatz wie oben, aber 'cap' -> Ausreißer wird auf ~10 zurückgeschnitten, keine NaNs."""
    idx = _minute_index(50)
    y = np.ones(50) * 10.0
    y[10] = 1000.0
    df = pd.DataFrame({"pm10": y}, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    IQRRemoveOutliers(cols=("pm10",), iqr_factor=3.0, strategy="cap").apply(ctx)

    assert ctx.df["pm10"].isna().sum() == 0, "cap darf keine NaNs erzeugen"
    assert np.isclose(ctx.df.loc[idx[10], "pm10"], 10.0, atol=1e-6), "Ausreißer sollte auf den Korridor gekappt werden"


def test_min_samples_prevents_action_when_too_few_points():
    """Wenn weniger als min_samples valide Werte vorliegen, passiert nichts."""
    idx = _minute_index(5)
    df = pd.DataFrame({"pm10": [10.0, 10.0, 1000.0, 10.0, 10.0]}, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    IQRRemoveOutliers(cols=("pm10",), iqr_factor=3.0, strategy="replace_nan", min_samples=10).apply(ctx)
    # Keine Änderung
    pd.testing.assert_series_equal(ctx.df["pm10"], df["pm10"])


def test_add_flag_true_creates_flag_and_marks_outlier():
    """add_flag=True: Flagspalte <col>_iqr_flag wird angelegt und setzt True am Ausreißer."""
    idx = _minute_index(30)
    y = np.ones(30) * 5.0
    y[15] = 50.0
    df = pd.DataFrame({"pm2.5": y}, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    IQRRemoveOutliers(cols=("pm2.5",), strategy="replace_nan", add_flag=True).apply(ctx)

    assert "pm2.5_iqr_flag" in ctx.df.columns
    assert bool(ctx.df.loc[idx[15], "pm2.5_iqr_flag"]) is True
    # Bei replace_nan wird der Ausreißer entfernt
    assert np.isnan(ctx.df.loc[idx[15], "pm2.5"])


def test_excludes_flag_bool_and_label_columns():
    """*_flag, bool und label-Spalten werden ignoriert; numerische wird interpoliert/gesäubert."""
    idx = _minute_index(10)
    df = pd.DataFrame({
        "pm10": [10.0, 10.0, 200.0, 10.0, 10.0, 11.0, 10.0, 12.0, 10.0, 10.0],
        "pm10_iqr_flag": [False]*10,      # existierende Flag-Spalte
        "is_on": [True]*10,               # bool
        "label": ["a"]*10                 # non-numeric
    }, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    IQRRemoveOutliers(cols=None, strategy="replace_nan").apply(ctx)

    # Nicht-numerische/Flag/Bool-Spalten bleiben unverändert
    assert ctx.df["pm10_iqr_flag"].dtype == bool
    assert ctx.df["is_on"].dtype == bool
    assert (ctx.df["label"] == "a").all()
    # Der Ausreißer in pm10 wurde entfernt
    assert np.isnan(ctx.df.loc[idx[2], "pm10"])


def test_handles_nullable_int64_series():
    idx = _minute_index(8)
    eco2 = pd.Series([400, 410, 405, None, 420, 415, 20000, 410], dtype="Int64", index=idx)
    df = pd.DataFrame({"eco2": eco2}, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    IQRRemoveOutliers(cols=("eco2",), strategy="replace_nan").apply(ctx)

    # Ausreißer (20000) wurde entfernt
    assert pd.isna(ctx.df.loc[idx[6], "eco2"])

    # Alle Werte außer dem ursprünglich fehlenden (idx[3]) und dem Ausreißer (idx[6]) sind nicht NaN
    remaining = ctx.df.drop(index=[idx[3], idx[6]])["eco2"]
    assert remaining.notna().all()


def test_strategy_none_keeps_values_but_can_flag():
    """strategy='none' lässt Werte unangetastet; mit add_flag=True wird nur markiert."""
    idx = _minute_index(20)
    y = np.ones(20) * 7.0
    y[5] = 77.0
    df = pd.DataFrame({"pm10": y}, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    IQRRemoveOutliers(cols=("pm10",), strategy="none", add_flag=True).apply(ctx)

    # Werte unverändert
    pd.testing.assert_series_equal(ctx.df["pm10"], df["pm10"])
    # Flag gesetzt
    assert "pm10_iqr_flag" in ctx.df.columns and bool(ctx.df.loc[idx[5], "pm10_iqr_flag"]) is True


def test_normal_noise_not_removed_with_reasonable_iqr_factor():
    """Leichtes Rauschen um 100 darf nicht fälschlich entfernt werden (iqr_factor=3.0)."""
    rng = np.random.default_rng(42)
    idx = _minute_index(200)
    noise = rng.normal(0, 1.0, size=len(idx))  # σ=1
    y = 100.0 + noise
    df = pd.DataFrame({"pm2.5": y}, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    IQRRemoveOutliers(cols=("pm2.5",), iqr_factor=3.0, strategy="replace_nan").apply(ctx)

    # Es sollten nur sehr wenige (idealerweise keine) Punkte entfernt werden.
    frac_nans = ctx.df["pm2.5"].isna().mean()
    assert frac_nans < 0.05, f"Zu viele Punkte entfernt: {frac_nans:.3f}"
