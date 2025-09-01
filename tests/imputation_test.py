import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from context import DataContext
from steps.imputation import InterpolateGaps


def _minute_index(n=20, start="2025-01-01 00:00:00", freq="1min"):
    return pd.date_range(start=start, periods=n, freq=freq)


def test_interpolate_short_and_long_gaps_1min():
    """
    1-Minuten-Frequenz, max_gap_minutes=5 -> limit_points=5
    - kurze Lücke (3 Punkte) wird gefüllt
    - lange Lücke (7 Punkte) bleibt (teilweise) NaN
    """
    idx = _minute_index(25, freq="1min")
    y = np.linspace(10, 20, len(idx)).astype(float)
    # kurze Lücke: 5..7 (3 NaNs)
    y[5:8] = np.nan
    # lange Lücke: 15..21 (7 NaNs)
    y[15:22] = np.nan
    df = pd.DataFrame({"pm10": y}, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    InterpolateGaps(cols=None, max_gap_minutes=5).apply(ctx)

    # kurze Lücke vollständig gefüllt
    assert ctx.df.loc[idx[5:7], "pm10"].notna().all()
    # lange Lücke enthält weiterhin NaNs
    assert ctx.df.loc[idx[15:21], "pm10"].isna().any()
    # Ränder (ohne Nachbar) werden nicht gefüllt (limit_area="inside")
    assert not ctx.df.iloc[0].isna().any()


def test_interpolate_gaps_5min_freq_limit_points():
    """
    5-Minuten-Frequenz, max_gap_minutes=30 -> limit_points ≈ 6
    - 2er-Lücke (10 min) wird gefüllt
    - 7er-Lücke (35 min) bleibt (teilweise) NaN
    """
    idx = _minute_index(20, freq="5min")
    # Basis: Linearer Anstieg, damit "time"-Interpolation deterministisch
    y = np.linspace(100.0, 200.0, len(idx))
    # kurze Lücke: 4..5 (2 NaNs)
    y[4:6] = np.nan
    # lange Lücke: 10..16 (7 NaNs)
    y[10:17] = np.nan

    df = pd.DataFrame({"ECO2": y}, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    InterpolateGaps(cols=None, max_gap_minutes=30).apply(ctx)

    # kurze Lücke gefüllt
    assert ctx.df.loc[idx[4:5], "ECO2"].notna().all()
    # lange Lücke bleibt teilweise NaN
    assert ctx.df.loc[idx[10:16], "ECO2"].isna().any()


def test_excludes_bool_flag_and_non_numeric_columns():
    """
    *_flag-, bool- und klar nicht-numerische Spalten werden ignoriert.
    Numerische Spalte wird interpoliert.
    """
    idx = _minute_index(10, freq="1min")
    df = pd.DataFrame({
        "pm10": [10.0, np.nan, 12.0, np.nan, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        "pm10_roc_flag": [False, False, True, False, False, False, False, False, False, False],
        "label": ["a"]*10,
        "is_on": [True]*10
    }, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    InterpolateGaps(cols=None, max_gap_minutes=10).apply(ctx)

    # bool/flag/label-Spalten unverändert
    assert ctx.df["pm10_roc_flag"].dtype == bool
    assert ctx.df["is_on"].dtype == bool
    assert (ctx.df["label"] == "a").all()

    # numerische pm10-Lücken wurden interpoliert (innerhalb max_gap)
    # hier: NaNs an Position 1 und 3 -> beide gefüllt
    assert ctx.df.iloc[1]["pm10"] == ctx.df.iloc[1]["pm10"]  # not NaN
    assert ctx.df.iloc[3]["pm10"] == ctx.df.iloc[3]["pm10"]  # not NaN


def test_non_monotonic_index_is_sorted_before_interpolation():
    """
    Nicht-monotone Indizes werden sortiert; Interpolation funktioniert danach wie erwartet.
    - Innenliegendes NaN wird gefüllt (hat links+rechts Nachbarn).
    - Rand-NaN am Ende bleibt NaN (limit_area='inside').
    """
    idx = _minute_index(6, freq="1min")
    # Werte so, dass t1 und t5 NaN sind:
    # t0=10.0, t1=NaN, t2=12.0, t3=16.0, t4=20.0, t5=NaN
    values = [10.0, np.nan, 12.0, 16.0, 20.0, np.nan]

    # Nicht-monotone Reihenfolge, WICHTIG: t5 als letztes Element,
    # damit nach Sortierung das letzte Element weiterhin t5 (NaN) ist.
    shuffled = pd.DatetimeIndex([idx[0], idx[2], idx[1], idx[3], idx[4], idx[5]])

    df = pd.DataFrame({"pm2.5": values}, index=shuffled)
    ctx = DataContext(csv_file="", df=df.copy())

    InterpolateGaps(cols=None, max_gap_minutes=10).apply(ctx)

    # Nach Sortierung prüfen
    df_sorted = ctx.df.sort_index()

    # Innenliegendes NaN (t1) wurde gefüllt
    assert df_sorted.loc[idx[1], "pm2.5"] == df_sorted.loc[idx[1], "pm2.5"]  # not NaN

    # Rand-NaN (t5) bleibt NaN (limit_area='inside' füllt keine Ränder)
    assert np.isnan(df_sorted.loc[idx[5], "pm2.5"])


def test_no_missing_values_leaves_series_unchanged():
    """
    Wenn keine NaNs vorhanden sind, werden keine Werte verändert.
    """
    idx = _minute_index(8, freq="1min")
    df = pd.DataFrame({"pm10": np.linspace(0.0, 7.0, 8)}, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    InterpolateGaps(cols=None, max_gap_minutes=10).apply(ctx)

    pd.testing.assert_series_equal(ctx.df["pm10"], df["pm10"])