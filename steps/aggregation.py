from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from pandas.api import types as ptypes
from typing import List, Optional
import sys, os,re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from context import DataContext, PipelineStep

@dataclass(frozen=True)
class ComputeMedians(PipelineStep):
    """
    Erzeugt rollende und tägliche Mediane für die Zeitreihendaten.

    - df_roll_median: Rolling-Median über ein zeitbasiertes Fenster (z. B. '24h').
      Standard: trailing & inkl. aktuellem Zeitpunkt (closed='right').
      Tipp für leakage-freies Labeling: closed='left' (nimmt NUR Vergangenheitswerte).
    - df_daily_median: Resample auf Tages-Median (Kalendertage gemäß Index-Zeitzone).

    Flag-/Bool-/Labelspalten werden ausgeschlossen, damit keine "Hilfsspalten" einfließen.
    """

    # Zeitfenster für den Rolling-Median (zeitbasiert, z. B. '24h', '7D')
    window: str = "24h"

    # Mindestanzahl Punkte im Fenster (None => Pandas-Default, hier 1)
    min_periods: Optional[int] = 1

    # Welche Fenstergrenze gilt als enthalten? ('right' = inkl. aktueller Zeit, 'left' = exklusiv)
    # Für ML-Labels oft 'left', für rein deskriptive Statistiken 'right'.
    closed: str = "left"   # 'left' | 'right' | 'both' | 'neither'

    # Tagesfrequenz (idR '1D'); bei Bedarf z. B. Arbeits-/Schichttage konfigurierbar
    daily_freq: str = "1D"

    # Spalten, die nie aggregiert werden sollen (Flags/Labels/IDs etc.)
    exclude_regex: str = r"(_flag$|class|label|category|id)"

    def _eligible_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Wählt kontinuierliche numerische Messspalten aus:
        - keine Bool-Dtypen,
        - keine Flag-/Label-Spalten (Regex),
        - nur numerische Dtypen (float/int).
        """
        pat = re.compile(self.exclude_regex, flags=re.IGNORECASE)
        elig: List[str] = []
        for c in df.columns:
            if pat.search(c):
                continue
            if ptypes.is_bool_dtype(df[c]):  # bools nicht aggregieren
                continue
            if ptypes.is_numeric_dtype(df[c]):
                elig.append(c)
        return elig

    def apply(self, ctx: DataContext) -> None:
        df = ctx.df

        # Sicherstellen, dass ein DatetimeIndex vorliegt und sortiert ist.
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("ComputeMedians erwartet einen DatetimeIndex.")
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
            ctx.df = df  # zurück in den Context schreiben

        # Relevante Spalten bestimmen (keine Flags/Bools/Labels).
        cols = self._eligible_columns(df)
        if not cols:
            # Falls nichts geeignet ist, lege leere Frames an – hält den Pipeline-Vertrag ein.
            ctx.df_roll_median = pd.DataFrame(index=df.index)
            ctx.df_daily_median = pd.DataFrame()
            ctx.log.log("compute_medians", window=self.window, note="no eligible numeric columns")
            return

        # Rolling-Median (zeitbasiertes Fenster):
        # - trailing Fenster, Pandas-rolling mit Offsets respektiert Zeitabstände im Index
        # - closed: 'right' (inkl. aktueller Zeit) oder 'left' (exkl. aktueller Zeit) zur Kontrolle von Leakage
        df_roll = (
            df[cols]
            .rolling(window=self.window, min_periods=self.min_periods, closed=self.closed)
            .median()   # skipna=True default; arbeitet spaltenweise
        )

        # Tages-Median:
        # - Resample nach Kalendertagen basierend auf Index-Zeitzone (falls tz-aware)
        # - label='left' → Tageslabel ist der Tagesbeginn
        # - closed='left' → [Tag 00:00, nächster Tag 00:00)
        df_daily = (
            df[cols]
            .resample(self.daily_freq, label="left", closed="left")
            .median()
        )

        # Ergebnisse im Context ablegen (API-kompatibel zu deinem bisherigen Code)
        ctx.df_roll_median = df_roll
        ctx.df_daily_median = df_daily

        # Kompaktes Logging (hilfreich für die Methodenbeschreibung)
        ctx.log.log(
            "compute_medians",
            window=self.window,
            min_periods=self.min_periods,
            closed=self.closed,
            daily_freq=self.daily_freq,
            n_cols=len(cols),
            cols=cols[:8] + (["..."] if len(cols) > 8 else [])
        )