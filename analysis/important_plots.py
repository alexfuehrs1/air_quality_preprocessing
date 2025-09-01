import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_impact_bar(impacts: pd.DataFrame, by="step"):
    """
    Stacked-Bar: pro Step (oder pro Spalte) wie viele Punkte geändert/zu-NaN/von-NaN.
    Erwartet: impacts[by], impacts[n_changed|n_to_nan|n_from_nan]
    """
    # Prüfen ob `by` existiert
    if by not in impacts.columns:
        raise ValueError(f"plot_impact_bar: Spalte '{by}' fehlt in impacts (Spalten={list(impacts.columns)})")

    needed = ["n_changed", "n_to_nan", "n_from_nan"]
    for col in needed:
        if col not in impacts.columns:
            raise ValueError(f"plot_impact_bar: Spalte '{col}' fehlt in impacts (Spalten={list(impacts.columns)})")

    g = impacts.groupby(by)[needed].sum().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = None
    for comp in needed:
        ax.bar(g.index, g[comp], bottom=bottoms, label=comp)
        bottoms = (bottoms if bottoms is not None else 0) + g[comp]

    ax.set_ylabel("count")
    ax.set_title(f"Impact per {by}")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

def plot_dist_before_after(snaps, column, steps_to_show=("HampelFilter", "IQRRemoveOutliers", "InterpolateGaps")):
    """
    Histogramm-Vergleich: 'raw' (erste Version mit Spalte) vs. nach bestimmten Schritten.
    
    Args:
        snaps: Liste von (step_name, DataFrame) aus run_with_snapshots.
        column: Name der Spalte, die geplottet werden soll.
        steps_to_show: Welche Schritte zusätzlich zur Raw-Verteilung angezeigt werden sollen.
    """
    # --- 1) Raw suchen ---
    raw = None
    for name, df in snaps:
        if column in df.columns:
            raw = df[column].dropna().astype(float)
            break

    if raw is None or raw.empty:
        raise ValueError(f"plot_dist_before_after: Spalte '{column}' nicht in Snapshots gefunden oder leer.")

    # --- 2) Plot vorbereiten ---
    fig, ax = plt.subplots(figsize=(8, 5))

    # Raw-Verteilung
    raw.plot(kind="hist", bins=40, alpha=0.35, ax=ax, density=True, label="raw")

    # --- 3) Bestimmte Schritte einzeichnen ---
    found_any = False
    for name, df in snaps:
        if name in steps_to_show and column in df.columns:
            series = df[column].dropna().astype(float)
            if not series.empty:
                series.plot(kind="hist", bins=40, alpha=0.35, ax=ax, density=True, label=name)
                found_any = True

    if not found_any:
        ax.text(0.5, 0.5, "Keine weiteren Schritte mit dieser Spalte gefunden", 
                ha="center", va="center", transform=ax.transAxes, fontsize=9, color="red")

    # --- 4) Formatierung ---
    ax.set_title(f"Verteilung vor/nach Cleaning – {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Dichte")
    ax.legend()
    plt.tight_layout()

    return fig

def plot_missingness_heatmap(df: pd.DataFrame, cols=None, max_rows=2000):
    """
    Einfache Missingness-Heatmap (Zeit x Spalte).
    """
    if cols is None:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # Downsample bei sehr vielen Zeilen (nur für Plot)
    if len(df) > max_rows:
        df = df.iloc[::int(np.ceil(len(df)/max_rows))].copy()
    M = df[cols].isna().astype(int).values.T  # shape: (len(cols), len(time))
    fig, ax = plt.subplots(figsize=(10, 0.35*len(cols)+2))
    im = ax.imshow(M, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    ax.set_xticks([0, M.shape[1]-1])
    ax.set_xticklabels([str(df.index[0]), str(df.index[-1])], rotation=15)
    ax.set_title("Missingness-Heatmap (1 = NaN)")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    plt.tight_layout()
    return fig

def plot_interpolation_stats(kpis: pd.DataFrame):
    """
    Balken für 'filled' und 'skipped_long' aus dem Interpolate-Log.
    """
    d = kpis[kpis["event"]=="interpolate_gaps"].groupby("column")[["filled","skipped_long"]].sum()
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(8,5))
    d.plot(kind="bar", ax=ax)
    ax.set_title("Interpolation: gefüllte Punkte vs. übersprungene lange Lücken")
    ax.set_ylabel("count"); plt.tight_layout(); return fig

def plot_rate_of_change_effect(kpis: pd.DataFrame):
    """
    Balken: wie oft RoC getriggert (detected) pro Spalte.
    """
    d = kpis[kpis["event"].isin(["rate_of_change_check","rate_of_change_check_v2"])]
    if d.empty: return None
    g = d.groupby("column")["detected"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8,5))
    g.plot(kind="bar", ax=ax)
    ax.set_title("Rate-of-Change: Anzahl erkannter Sprünge")
    ax.set_ylabel("count"); plt.tight_layout(); return fig

def plot_flatline_effect(kpis: pd.DataFrame):
    d = kpis[kpis["event"]=="flatline_check"]
    if d.empty: return None
    g = d.groupby("column")["flagged"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8,5))
    g.plot(kind="bar", ax=ax)
    ax.set_title("Flatline: Anzahl der als 'stuck' geflaggten Punkte")
    ax.set_ylabel("count"); plt.tight_layout(); return fig


