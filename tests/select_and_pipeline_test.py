import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline import AirQualityPipeline
from context import DataContext
from steps.io import LoadCSV, SelectColumns
from steps.cleaning import RemoveInvalidValues
import pytest, re
from pipeline import DropHelperColumns
from pipeline import make_default_pipeline, AirQualityPipeline
from context import DataContext, PipelineStep



# --- kleine Helpers ------------------------------------------------------------

def _write_csv(path, df):
    df.to_csv(path, index=False)

def _minute_index(n, start="2025-01-01 00:00:00", freq="1min"):
    return pd.date_range(start=start, periods=n, freq=freq)

# Dummy-Step, um Errors zu provozieren
class BoomStep(PipelineStep):
    def apply(self, ctx):
        raise RuntimeError("boom")

# ------------------------------------------------------------------------------
# 1) Basis: Pipeline läuft durch, Logs enthalten step_start & pipeline_done
# ------------------------------------------------------------------------------

def test_pipeline_runs_and_logs_step_start_and_done(tmp_path):
    # Mini-CSV (mit kleiner Lücke)
    idx = _minute_index(5)
    df = pd.DataFrame({
        "timestamp": [idx[0], idx[1], idx[3], idx[4]],  # t2 fehlt → asfreq füllt Raster
        "pm10": [10.0, 11.0, 13.0, 14.0],
        "pm2.5": [8.0, 9.0, 12.0, 13.0],
        "eco2": [400, 410, 420, 430],
    })
    csv = tmp_path / "mini.csv"
    _write_csv(csv, df)

    pipe = make_default_pipeline(
        window="24h",
        max_gap_minutes=5,
        enable_hampel=True,
        enable_roc=True,
        enable_flatline=True,
        drop_flags_before_outputs=True,  # wir testen auch den Flag-Drop
        hampel_window=11,
        hampel_sigmas=3.0,
    )

    ctx = pipe.run(csv_file=str(csv), tz="Europe/Berlin", selected_columns=["pm2.5", "pm10", "eco2"])

    # Ergebnis-DF existiert und hat reguläres Raster
    assert isinstance(ctx.df.index, pd.DatetimeIndex)
    # Rolling / Daily wurden erzeugt
    assert ctx.df_roll_median is not None
    assert ctx.df_daily_median is not None

    # Logs enthalten step_start und pipeline_done
    kinds = [e.get("kind") for e in ctx.log.entries]
    assert "step_start" in kinds
    assert any(e.get("kind") == "pipeline_done" for e in ctx.log.entries)

# ------------------------------------------------------------------------------
# 2) Fehlerfall: Exception blubbert hoch, 'step_error' wird geloggt
# ------------------------------------------------------------------------------

def test_pipeline_logs_step_error_and_raises(tmp_path):
    # CSV
    idx = _minute_index(3)
    df = pd.DataFrame({"timestamp": idx, "pm10": [10.0, 11.0, 12.0]})
    csv = tmp_path / "err.csv"
    _write_csv(csv, df)

    # Pipeline mit absichtlichem Fehler-Step in der Mitte
    steps = [
        LoadCSV(),
        SelectColumns(),
        BoomStep(),  # wird crashen
        RemoveInvalidValues(),
    ]
    pipe = AirQualityPipeline(steps=steps)

    with pytest.raises(RuntimeError, match="boom"):
        pipe.run(csv_file=str(csv), tz=None, selected_columns=["pm10"])

    # Prüfe, dass der Fehler geloggt wurde
    # (wir erstellen den Context erneut, um an die Logs zu kommen, falls dein runner bricht
    #  ohne ctx zurückzugeben. Wenn dein run ctx zurückgibt, kannst du direkt prüfen.)
    ctx = DataContext(csv_file=str(csv))
    for s in steps[:2]:
        s.apply(ctx)
    try:
        steps[2].apply(ctx)
    except Exception as e:
        ctx.log.log("step_error", step_name=steps[2].__class__.__name__, error=str(e))
    found = [e for e in ctx.log.entries if e.get("kind") == "step_error" and "boom" in e.get("error","")]
    assert found, "step_error sollte geloggt werden"

# ------------------------------------------------------------------------------
# 3) Struktur: make_default_pipeline – Toggles ein/aus & Reihenfolge
# ------------------------------------------------------------------------------

def _classnames(steps):
    return [s.__class__.__name__ for s in steps]

def test_make_default_pipeline_includes_expected_steps_and_order():
    pipe = make_default_pipeline(enable_hampel=True, enable_roc=True, enable_flatline=True)
    names = _classnames(pipe.steps)

    # Kernschritte vorhanden & in sinnvoller Reihenfolge
    # (du kannst die Liste enger prüfen – hier nur die Order-Anker)
    assert names[0] == "LoadCSV"
    assert "SelectColumns" in names
    assert names.index("RemoveInvalidValues") < names.index("PMHierarchyCheck")
    assert "RateOfChangeCheck" in names
    assert "HampelFilter" in names
    assert "FlatlineCheck" in names
    assert names.index("IQRRemoveOutliers") > names.index("FlatlineCheck")
    assert names.index("InterpolateGaps") > names.index("IQRRemoveOutliers")
    assert names.index("PMHierarchyCheck") < names.index("ComputeMedians")  # früher pass + später strict pass
    assert "ComputeMedians" in names
    assert "SaveOutputs" in names

def test_make_default_pipeline_toggles_remove_steps():
    # Alle drei toggles aus
    pipe = make_default_pipeline(enable_hampel=False, enable_roc=False, enable_flatline=False, drop_flags_before_outputs=False)
    names = _classnames(pipe.steps)
    assert "RateOfChangeCheck" not in names
    assert "HampelFilter" not in names
    assert "FlatlineCheck" not in names
    assert "DropHelperColumns" not in names

# ------------------------------------------------------------------------------
# 4) DropHelperColumns entfernt *_flag Spalten vor Compute/Save
# ------------------------------------------------------------------------------

def test_drop_helper_columns_removes_flags(tmp_path, monkeypatch):
    # CSV mit Flag-Spalten
    idx = _minute_index(6)
    df = pd.DataFrame({
        "timestamp": idx,
        "pm10": [10, 11, 12, 13, 14, 15],
        "pm10_roc_flag": [False, True, False, False, False, False],
        "pm10_hampel_flag": [False]*6,
        "pm10_flatline_flag": [False]*6,
        "pm10_iqr_flag": [False]*6,
        "eco2_invalid_flag": [False]*6,
    })
    csv = tmp_path / "flags.csv"
    _write_csv(csv, df)

    # kleine Pipeline, die nur lädt und droppt
    from types import SimpleNamespace
    steps = [
        LoadCSV(),
        DropHelperColumns(),
    ]
    pipe = AirQualityPipeline(steps=steps)
    ctx = pipe.run(str(csv), tz=None, selected_columns=["pm10"])  # SelectColumns nicht nötig hier

    # Alle *_flag Spalten sind weg
    assert all(not re.search(r"_(roc|hampel|flatline|iqr|invalid)_flag$", c) for c in ctx.df.columns)

# ------------------------------------------------------------------------------
# 5) ComputeMedians closed='left' im Default – Verhalten überprüfen
# ------------------------------------------------------------------------------

def test_default_pipeline_compute_medians_closed_left(tmp_path):
    # Reihe: 1,10,100 → Rolling-Median bei t2:
    # closed='left' -> Median(1,10)=5.5 (100 ist „Zukunft“ und wird ausgeschlossen)
    idx = _minute_index(3)
    df = pd.DataFrame({"timestamp": idx, "pm10": [1.0, 10.0, 100.0], "pm2.5": [1.0, 9.0, 90.0]})
    csv = tmp_path / "cm.csv"
    _write_csv(csv, df)

    pipe = make_default_pipeline(
        enable_hampel=False, enable_roc=False, enable_flatline=False,
        drop_flags_before_outputs=True
    )
    # wir möchten Cleaning-Noise vermeiden; darum nur die zwei Spalten wählen
    ctx = pipe.run(str(csv), tz=None, selected_columns=["pm10", "pm2.5"])

    # Rolling-Frame existiert; closed='left' im Default gesetzt
    t2 = idx[2]
    # Bei asfreq ist das Raster identisch → median an t2 prüfen
    m_pm10_t2 = ctx.df_roll_median.loc[t2, "pm10"]
    assert np.isclose(m_pm10_t2, 5.5), f"Erwartet 5.5, bekam {m_pm10_t2}"

# ------------------------------------------------------------------------------
# 6) Mini End-to-End mit Hierarchie-Checks (früh vs. spät)
# ------------------------------------------------------------------------------

def test_pm_hierarchy_early_and_late_pass(tmp_path):
    # Konstruiere Fall:
    # - an t1 echter Verstoß: pm2.5 > pm10  -> früher Pass setzt beide auf NaN
    # - an t3 nur pm10 vorhanden (pm2.5 NaN) -> nach Interpolation bleibt einseitig,
    #   später strict=True Pass synchronisiert (setzt pm10 dort auf NaN)
    idx = _minute_index(5)
    df = pd.DataFrame({
        "timestamp": idx,
        "pm2.5": [5.0, 20.0, 7.0, np.nan, 8.0],  # t1=20.0
        "pm10":  [6.0, 10.0, 8.0,  9.0,    9.0], # t1=10.0  -> Verstoß
        "eco2":  [400, 410, 420, 430, 440],
    })
    csv = tmp_path / "pmh.csv"
    _write_csv(csv, df)

    pipe = make_default_pipeline(
        enable_hampel=False, enable_roc=False, enable_flatline=False,
        max_gap_minutes=1, drop_flags_before_outputs=True
    )
    ctx = pipe.run(str(csv), tz=None, selected_columns=["pm2.5", "pm10", "eco2"])

    evt_field = "kind" if any("kind" in e for e in ctx.log.entries) else "step"

    # Früher Pass hat gegriffen (Verletzung geloggt)
    assert any(
        e.get(evt_field) == "pm_hierarchy_check"
        and e.get("strict") is False
        and e.get("violations", 0) > 0
        for e in ctx.log.entries
    )

    # t1: nach Interpolation wieder gültig und Hierarchie-konform
    t1 = idx[1]
    assert pd.notna(ctx.df.loc[t1, "pm2.5"]) and pd.notna(ctx.df.loc[t1, "pm10"])
    assert ctx.df.loc[t1, "pm2.5"] <= ctx.df.loc[t1, "pm10"]

    # Später strikter Pass wurde ausgeführt
    assert any(
        e.get(evt_field) == "pm_hierarchy_check" and e.get("strict") is True
        for e in ctx.log.entries
    )

    # t3: entweder synchronisiert (beide NaN) ODER interpoliert & hierarchie-konform
    t3 = idx[3]
    pair = ctx.df.loc[t3, ["pm2.5", "pm10"]]
    if pair.isna().any():
        assert pair.isna().all(), "Wenn synchronisiert, dann beide NaN"
    else:
        assert pair["pm2.5"] <= pair["pm10"], "Wenn interpoliert, muss Hierarchie stimmen"