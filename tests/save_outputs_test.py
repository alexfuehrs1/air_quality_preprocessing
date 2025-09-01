import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from context import DataContext
from steps.io import SaveOutputs

def test_save_outputs_creates_files(tmp_path):
    idx = pd.date_range("2025-01-01 00:00:00", periods=10, freq="1min")
    df = pd.DataFrame({"pm2.5": np.arange(10, dtype=float), "eco2": 400.0}, index=idx)

    ctx = DataContext(csv_file="", df=df)
    ctx.df_roll_median = df.rolling("24h", min_periods=1).median()
    ctx.df_daily_median = df.resample("1D").median()
    ctx.log.log("dummy_step", ok=True)

    saver = SaveOutputs(output_dir=str(tmp_path), prefix="aq_test", formats=("csv",), include_log=True)
    saver.apply(ctx)

    expected = [
        tmp_path / "aq_test_clean.csv",
        tmp_path / "aq_test_roll_median.csv",
        tmp_path / "aq_test_daily_median.csv",
        tmp_path / "aq_test_log.csv",
    ]
    for p in expected:
        assert p.exists(), f"Datei fehlt: {p}"

    # Schnellcheck: CSV lesbar
    df_loaded = pd.read_csv(tmp_path / "aq_test_clean.csv")
    assert "pm2.5" in df_loaded.columns
