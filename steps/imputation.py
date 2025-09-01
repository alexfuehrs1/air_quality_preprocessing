from __future__ import annotations
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
from dataclasses import dataclass
from typing import Iterable, Optional, List
import sys, os, re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from context import DataContext, PipelineStep

@dataclass
class InterpolateGaps(PipelineStep):
    cols: Optional[Iterable[str]] = None
    max_gap_minutes: int = 30
    exclude_regex: str = r"(_flag$|class|label|category|id)"  # typische Nicht-Zeitreihen

    def _eligible_columns(self, df: pd.DataFrame, cols: Optional[List[str]]) -> List[str]:
        cand = list(cols) if cols is not None else df.columns.tolist()
        elig: List[str] = []
        pat = re.compile(self.exclude_regex, flags=re.IGNORECASE)
        for c in cand:
            # explizit: keine Flags/Bools/Klassensp.
            if pat.search(c):
                continue
            if ptypes.is_bool_dtype(df[c]):
                continue
            # nur echte numerik (float/int) – aber z.B. Int64-Klassen rausfiltern über regex oben
            if ptypes.is_numeric_dtype(df[c]):
                elig.append(c)
        return elig

    def _allowed_consecutive_nans(self, idx: pd.DatetimeIndex) -> Optional[int]:
        # Schrittweite in Minuten (Median ist robust)
        if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
            return None
        dt = pd.Series(idx).diff().dt.total_seconds().median()
        if pd.isna(dt) or dt <= 0:
            return None
        step_min = dt / 60.0
        # limit in ANZAHL Punkten, mind. 1
        return max(1, int(round(self.max_gap_minutes / step_min)))

    def apply(self, ctx: DataContext) -> None:
        df = ctx.df
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("InterpolateGaps erwartet einen DatetimeIndex.")
        if not df.index.is_monotonic_increasing:
            df.sort_index(inplace=True)

        # Spaltenauswahl
        base_cols = list(self.cols) if self.cols is not None else df.columns.tolist()
        eligible = self._eligible_columns(df, base_cols)

        # limit aus Minuten ableiten (Anzahl max. aufeinanderfolgende NaNs)
        limit = self._allowed_consecutive_nans(df.index)

        changes = {}
        for col in eligible:
            s = df[col]
            missing = s.isna()
            if not missing.any():
                changes[col] = {"na_before": 0, "na_after": 0, "filled": 0, "skipped_long": 0}
                continue

            # Run-Length in COUNT (für Logging + long-run Erkennung)
            grp_id = (missing != missing.shift()).cumsum()
            run_len = missing.groupby(grp_id).transform("size").where(missing, 0)

            na_before = int(missing.sum())

            # Interpolation: zeitbasiert, nur innen, mit limit in COUNT (wenn bestimmbar)
            s_interp = s.interpolate(
                method="time",
                limit=limit,                      # None → kein Limit, sonst nach max_gap_minutes in Count
                limit_direction="both",
                limit_area="inside",
            )

            # Falls limit None war (unregelmäßig), setzen wir zusätzlich eine Zeit-basierte Schranke:
            # (optional) Für einfache/feste Takte reicht das oben.
            if limit is None:
                # Grobe Schutzschicht: lange Lücken (in Minuten) wieder auf NaN setzen
                # Wir approximieren Lückenlänge über Anzahl Punkte * Median-Schritt (falls verfügbar)
                dt_sec = pd.Series(df.index).diff().dt.total_seconds().median()
                if not pd.isna(dt_sec) and dt_sec > 0:
                    step_min = dt_sec / 60.0
                    allowed_len = max(1, int(round(self.max_gap_minutes / step_min)))
                    long_mask = run_len > allowed_len
                    s_final = s_interp.where(~long_mask, np.nan)
                else:
                    s_final = s_interp
            else:
                # Bei limit != None kann 'both' bis zu 2*limit füllen.
                # Wir setzen NaNs für alle Runs, die länger als 'limit' sind, konsequent zurück.
                long_mask = run_len > limit
                s_final = s_interp.where(~long_mask, np.nan)
                
            df[col] = s_final
            na_after = int(df[col].isna().sum())
            changes[col] = {
                "na_before": na_before,
                "na_after": na_after,
                "filled": max(0, na_before - na_after),
                "skipped_long": int(long_mask.sum()) if 'long_mask' in locals() else 0,
                "limit_points": limit,
            }

        ctx.log.log("interpolate_gaps",
                    cols=eligible,
                    max_gap_minutes=self.max_gap_minutes,
                    changes=changes)