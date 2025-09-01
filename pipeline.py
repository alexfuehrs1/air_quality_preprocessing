from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import re
from context import DataContext
from steps.io import LoadCSV, SaveOutputs, SelectColumns
from steps.cleaning import RemoveInvalidValues, IQRRemoveOutliers
from steps.filters import HampelFilter, RateOfChangeCheck, FlatlineCheck
from steps.imputation import InterpolateGaps
from steps.validation import PMHierarchyCheck
from steps.aggregation import ComputeMedians
from context import PipelineStep


@dataclass(frozen=True)
class DropHelperColumns(PipelineStep):
    """
    Entfernt Hilfsspalten (z. B. *_roc_flag, *_hampel_flag, *_flatline_flag, *_iqr_flag)
    vor Persistierung/Aggregation. So landen Diagnose-Spalten nicht in den Enddateien.
    """
    pattern: str = r"_(roc|hampel|flatline|iqr|invalid)_flag$"

    def apply(self, ctx: DataContext) -> None:
        if ctx.df is None or ctx.df.empty:
            ctx.log.log("drop_helper_columns", removed=[], note="empty df")
            return
        cols = [c for c in ctx.df.columns if re.search(self.pattern, c, flags=re.IGNORECASE)]
        ctx.df.drop(columns=cols, inplace=True, errors="ignore")
        ctx.log.log("drop_helper_columns", removed=cols)


# --- Pipeline-Kern --------------------------------------------------------------
@dataclass
class AirQualityPipeline:
    steps: List[PipelineStep]

    def run(
        self,
        csv_file: str,
        tz: Optional[str] = None,
        selected_columns: Optional[List[str]] = None
    ) -> DataContext:
        """
        Führt alle Steps seriell aus und gibt den End-Context zurück.
        Verbesserungen:
          - Jeder Step wird mit 'step_start' geloggt; bei Fehlern erhältst du 'step_error'
            mit Klassenname und Message → Debuggen ist viel einfacher.
        """
        ctx = DataContext(csv_file=csv_file, tz=tz, selected_cols=selected_columns)
        for step in self.steps:
            name = step.__class__.__name__
            ctx.log.log("step_start", step_name=name)
            try:
                step.apply(ctx)
            except Exception as e:
                # Fehler sichtbar machen und Step benennen
                ctx.log.log("step_error", step=name, error=str(e))
                raise
        ctx.log.log("pipeline_done", n_steps=len(self.steps))
        return ctx


# --- Empfehlenswerte Default-Pipeline ------------------------------------------
def make_default_pipeline(
    window: str = "24h",
    max_gap_minutes: int = 30,
    enable_hampel: bool = True,
    enable_roc: bool = True,
    enable_flatline: bool = True,
    drop_flags_before_outputs: bool = True,
    # sinnvolle Defaults; passe sie bei Bedarf an dein Projekt an:
    hampel_window: int = 25,
    hampel_sigmas: float = 3.0,
    hampel_center: bool = True,  # Cleaning: center=True ok; für leakagefreie Features später False
    iqr_factor: float = 3.0,
) -> AirQualityPipeline:
    """
    Empfohlene Reihenfolge (kurz warum):
      1) LoadCSV            → sauberes Zeitraster (asfreq), tz, Duplikate raus
      2) SelectColumns      → nur relevante Messkanäle rein
      3) RemoveInvalidValues→ harte physikalische Grenzen (z. B. Humidity 0..100 %)
      4) PMHierarchyCheck   → *früh* rohen PM-Widerspruch (PM2.5 > PM10) entfernen
      5) RateOfChangeCheck  → implausible Sprünge gemäß abs/rel Regeln (Flag + Replace/Cap)
      6) HampelFilter       → robuste Ausreißererkennung gegen punktuelles Rauschen
      7) FlatlineCheck      → Hänger/Steher erkennen (Sensor stuck)
      8) IQRRemoveOutliers  → verbleibende Ausreißer global je Kanal
      9) InterpolateGaps    → nur *kurze* Lücken wieder füllen (zeitbasiert, innenliegend)
     10) PMHierarchyCheck   → *spät* strict=True: NaN-Synchronisation zwischen PM2.5/PM10
     11) DropHelperColumns  → *_flag-Spalten vor Persistierung entfernen (optional)
     12) ComputeMedians     → Rolling- und Tages-Mediane (für Auswertung/Plots)
     13) SaveOutputs        → CSV + Log

    Hinweise:
    - RoC VOR Hampel hilft, grobe Telemetriefehler zu „entkernen“, bevor der robuste Filter
      auf feineres Rauschen schaut. Umgekehrt ginge auch, entscheide je nach Datencharakter.
    - Zwei PMHierarchyCheck-Pässe: früh für echte Verstöße, spät (strict=True) zur Synchronisation
      nach Interpolation, damit keine inkonsistenten PM-Paare überleben.
    """
    steps: List[PipelineStep] = [
        LoadCSV(),                 # Zeitreihe laden, tz, asfreq
        #SelectColumns(),           # Subset + evtl. Umbenennungen/Mapping
        RemoveInvalidValues(),     # harte Limits (physikalisch/sensorisch)
        PMHierarchyCheck(strict=False),  # früh: echte PM-Verstöße (beide vorhanden) entfernen
    ]

    if enable_roc:
        steps.append(
            RateOfChangeCheck(cols=None)       # deine V2 mit abs/rel Regeln + Flags
        )

    if enable_hampel:
        steps.append(
            HampelFilter(
                cols=None,
                window=hampel_window,
                n_sigmas=hampel_sigmas,
                strategy="replace_nan",        # konservativ vor Interpolation
                center=hampel_center,
                add_flag=True
            )
        )

    if enable_flatline:
        steps.append(
            FlatlineCheck(
                cols=None,
                min_duration_minutes=30,
                tolerance=0.01,
                strategy="replace_nan",
                add_flag=True
            )
        )

    steps += [
        IQRRemoveOutliers(cols=None, iqr_factor=iqr_factor, strategy="replace_nan", add_flag=True),
        InterpolateGaps(cols=None, max_gap_minutes=max_gap_minutes),
        PMHierarchyCheck(strict=True),         # spät: NaN-Synchronisation (einseitige PM löschen)
    ]

    if drop_flags_before_outputs:
        steps.append(DropHelperColumns())      # Diagnose-Spalten raus

    steps += [
        ComputeMedians(window=window, closed="left"),  # leakagearm: nur Vergangenheit im Rolling
        SaveOutputs(),
    ]

    return AirQualityPipeline(steps)