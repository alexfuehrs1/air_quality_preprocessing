import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from context import DataContext
from steps.aggregation import ComputeMedians


def _minute_index(n=10, start="2025-01-01 00:00:00", freq="1min"):
    return pd.date_range(start=start, periods=n, freq=freq)


def test_rolling_median_closed_right_vs_left_differs():
    """
    Rolling-Fenster '3min' auf Werten [1, 10, 100] bei t=00:02:
      - closed='right' nutzt [1,10,100] → Median = 10
      - closed='left'  nutzt [1,10]      → Median = 5.5
    """
    idx = _minute_index(3)
    df = pd.DataFrame({"pm10": [1.0, 10.0, 100.0]}, index=idx)

    # closed='right'
    ctx_r = DataContext(csv_file="", df=df.copy())
    ComputeMedians(window="3min", min_periods=1, closed="right").apply(ctx_r)
    m_right = ctx_r.df_roll_median.loc[idx[2], "pm10"]

    # closed='left'
    ctx_l = DataContext(csv_file="", df=df.copy())
    ComputeMedians(window="3min", min_periods=1, closed="left").apply(ctx_l)
    m_left = ctx_l.df_roll_median.loc[idx[2], "pm10"]

    assert np.isclose(m_right, 10.0)
    assert np.isclose(m_left, 5.5)
    assert m_right != m_left


def test_min_periods_controls_early_nans():
    """
    min_periods=3 mit window='3min':
      - bei t=00:00 und t=00:01 nicht genug Punkte → NaN
      - ab t=00:02 vorhanden → Wert vorhanden
    """
    idx = _minute_index(3)
    df = pd.DataFrame({"pm2.5": [5.0, 7.0, 9.0]}, index=idx)
    ctx = DataContext(csv_file="", df=df.copy())

    ComputeMedians(window="3min", min_periods=3, closed="right").apply(ctx)
    roll = ctx.df_roll_median["pm2.5"]

    assert np.isnan(roll.loc[idx[0]])
    assert np.isnan(roll.loc[idx[1]])
    assert not np.isnan(roll.loc[idx[2]])


def test_daily_median_two_days_and_flag_exclusion():
    """
    Zwei Kalendertage, täglicher Median korrekt,
    *_flag und bool/label-Spalten werden ignoriert.
    """
    idx = pd.date_range("2025-01-01 00:00:00", periods=5, freq="1min").append(
        pd.date_range("2025-01-02 00:00:00", periods=4, freq="1min")
    )
    vals = [1.0, 2.0, 3.0, 4.0, 5.0] + [10.0, 20.0, 30.0, 40.0]
    df = pd.DataFrame({
        "pm10": vals,
        "pm10_hampel_flag": [False]*len(idx),
        "label": ["x"]*len(idx),
        "is_on": [True]*len(idx),
    }, index=idx)

    ctx = DataContext(csv_file="", df=df.copy())
    ComputeMedians(window="24h", min_periods=1, closed="right", daily_freq="1D").apply(ctx)

    # Nur die numerische Messspalte aggregiert
    assert list(ctx.df_roll_median.columns) == ["pm10"]
    assert list(ctx.df_daily_median.columns) == ["pm10"]

    # Tagesmediane: Tag1 Median von [1,2,3,4,5] = 3; Tag2 Median von [10,20,30,40] = 25
    d1 = pd.Timestamp("2025-01-01 00:00:00")
    d2 = pd.Timestamp("2025-01-02 00:00:00")
    assert np.isclose(ctx.df_daily_median.loc[d1, "pm10"], 3.0)
    assert np.isclose(ctx.df_daily_median.loc[d2, "pm10"], 25.0)


def test_non_monotonic_index_is_sorted_before_rolling():
    """
    Nicht-monotone Indizes werden sortiert; Rolling-Median wird korrekt berechnet.
    """
    idx = _minute_index(4)
    shuffled = pd.DatetimeIndex([idx[0], idx[2], idx[1], idx[3]])
    df = pd.DataFrame({"pm10": [10.0, 30.0, 20.0, 40.0]}, index=shuffled)

    ctx = DataContext(csv_file="", df=df.copy())
    ComputeMedians(window="3min", min_periods=1, closed="right").apply(ctx)

    # Nach Sortierung: Index = [t0,t1,t2,t3]; Median am t2 über [t0,t1,t2] = Median(10,20,30) = 20
    t2 = pd.Timestamp("2025-01-01 00:02:00")
    assert np.isclose(ctx.df_roll_median.loc[t2, "pm10"], 20.0)


def test_raises_if_index_not_datetime():
    """
    Ohne DatetimeIndex soll der Step klar scheitern.
    """
    df = pd.DataFrame({"pm10": [1.0, 2.0, 3.0]}, index=[0, 1, 2])
    ctx = DataContext(csv_file="", df=df.copy())

    try:
        ComputeMedians(window="24h").apply(ctx)
        assert False, "ComputeMedians muss bei Nicht-DatetimeIndex eine Exception werfen"
    except ValueError as e:
        assert "DatetimeIndex" in str(e)


def test_no_eligible_columns_results_in_empty_outputs():
    """
    Wenn alle Spalten Flags/Bool/Labels sind, sollen leere DataFrames entstehen.
    """
    idx = _minute_index(5)
    df = pd.DataFrame({
        "pm10_hampel_flag": [False]*5,
        "label": ["a"]*5,
        "is_on": [True]*5
    }, index=idx)

    ctx = DataContext(csv_file="", df=df.copy())
    ComputeMedians(window="24h", min_periods=1).apply(ctx)

    assert ctx.df_roll_median is not None and ctx.df_daily_median is not None
    assert ctx.df_roll_median.empty and ctx.df_roll_median.index.equals(idx)
    assert ctx.df_daily_median.empty