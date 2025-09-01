from analysis.log_kpis import collect_logger_kpis
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
from analysis.impacts_steps import step_impacts
from analysis.run_snapshots import run_with_snapshots
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline import make_default_pipeline

def compare_variants(csv_file, tz, selected_cols, variants: dict):
    """
    variants: dict[name] -> dict(make_default_pipeline kwargs)
      z.B.: {
        "full": {},
        "no_hampel": {"enable_hampel": False},
        "no_roc": {"enable_roc": False},
        "no_flatline": {"enable_flatline": False}
      }
    Returns: dict[name] -> {"ctx":..., "snaps":..., "impacts": pd.DataFrame, "kpis": pd.DataFrame}
    """
    out = {}
    for name, kwargs in variants.items():
        pipe = make_default_pipeline(**kwargs)
        ctx, snaps = run_with_snapshots(pipe, csv_file, tz=tz, selected_columns=selected_cols)
        impacts = step_impacts(snaps)
        kpis = collect_logger_kpis(ctx)
        out[name] = {"ctx": ctx, "snaps": snaps, "impacts": impacts, "kpis": kpis}
    return out

def plot_ablation_summary(results: dict, metric="n_to_nan"):
    """
    Vergleicht Varianten anhand Summe eines Impact-Metrics (z. B. 'n_to_nan' oder 'n_changed').
    """
    totals = {name: res["impacts"][metric].sum() for name, res in results.items()}
    s = pd.Series(totals).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7,4))
    s.plot(kind="bar", ax=ax)
    ax.set_title(f"Ablation: Summe {metric} über alle Schritte & Spalten")
    ax.set_ylabel("count")
    plt.tight_layout()
    return fig
