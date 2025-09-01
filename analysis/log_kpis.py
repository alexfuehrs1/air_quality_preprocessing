import pandas as pd
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from context import DataContext

def collect_logger_kpis(ctx: DataContext) -> pd.DataFrame:
    """
    Parst ctx.log.entries und extrahiert Kernmetriken je Step.
    Funktioniert mit beiden Logger-Schemata: 'kind' oder 'step'.
    """
    evts = []
    for e in ctx.log.entries:
        evt = e.get("kind") or e.get("step") or ""
        if evt in {"hampel_filter"}:
            ch = e.get("changes", {})
            for col, meta in ch.items():
                evts.append({"event": evt, "column": col, "detected": meta.get("detected", 0)})
        elif evt in {"rate_of_change_check", "rate_of_change_check_v2"}:
            ch = e.get("changes", {})
            for col, meta in ch.items():
                evts.append({"event": evt, "column": col, "detected": meta.get("detected", 0)})
        elif evt in {"flatline_check"}:
            ch = e.get("changes", {})
            for col, meta in ch.items():
                evts.append({"event": evt, "column": col, "flagged": meta.get("flagged", 0)})
        elif evt in {"iqr_remove_outliers"}:
            ch = e.get("bounds", {})
            # hier haben wir Bounds; 'na_before/after' sind zusätzlich im Log
            cols = e.get("cols", [])
            na_after = e.get("na_after", {})
            na_before = e.get("na_before", {})
            for col in cols:
                evts.append({"event": evt, "column": col,
                             "na_added": int(na_after.get(col, 0)) - int(na_before.get(col, 0))})
        elif evt in {"interpolate_gaps"}:
            ch = e.get("changes", {})
            for col, meta in ch.items():
                evts.append({"event": evt, "column": col,
                             "filled": meta.get("filled", 0),
                             "skipped_long": meta.get("skipped_long", 0)})
        elif evt in {"pm_hierarchy_check"}:
            evts.append({"event": evt, "strict": e.get("strict"), "violations": e.get("violations", 0)})
    return pd.DataFrame(evts)
