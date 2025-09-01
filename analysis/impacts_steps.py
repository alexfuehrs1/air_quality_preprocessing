import pandas as pd

def _safe_align(a: pd.Series, b: pd.Series):
    a, b = a.align(b, join="outer")
    return a.astype("float64"), b.astype("float64")

def step_impacts(snaps, eps=1e-12) -> pd.DataFrame:
    """
    Baut eine Tabelle (Long-Format) mit Impact je Schritt & Spalte:
      n_changed       : Wert wurde geändert (|Δ| > eps)
      n_to_nan        : vorher Wert, nachher NaN
      n_from_nan      : vorher NaN, nachher Wert
      n_unchanged     : unverändert
      n_total_overlap : Anzahl Datenpunkte im Vergleich (Schnitt der Indizes)
    """
    rows = []
    for i in range(1, len(snaps)):
        step_name, df_after = snaps[i]
        prev_step, df_before = snaps[i-1]

        common_cols = sorted(set(df_before.columns) & set(df_after.columns))
        for col in common_cols:
            a, b = _safe_align(df_before[col], df_after[col])
            both = a.index.intersection(b.index)
            a, b = a.loc[both], b.loc[both]

            before_nan = a.isna()
            after_nan  = b.isna()
            from_nan   = before_nan & ~after_nan
            to_nan     = ~before_nan & after_nan

            # geänderte Werte: beide nicht NaN und |Δ| > eps
            changed = (~before_nan) & (~after_nan) & ((a - b).abs() > eps)
            unchanged = (~before_nan) & (~after_nan) & ~changed

            rows.append({
                "step": step_name,
                "column": col,
                "n_total_overlap": int(len(both)),
                "n_changed": int(changed.sum()),
                "n_to_nan": int(to_nan.sum()),
                "n_from_nan": int(from_nan.sum()),
                "n_unchanged": int(unchanged.sum()),
            })
    return pd.DataFrame(rows)
