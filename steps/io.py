from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from context import DataContext, PipelineStep


@dataclass(frozen=True)
class LoadCSV(PipelineStep):
    parse_dates_col: Optional[str] = "timestamp"
    target_freq: Optional[str] = "1min"

    def apply(self, ctx: DataContext) -> None:
        try:
            df = pd.read_csv(ctx.csv_file)
        except Exception as e:
            raise RuntimeError(f"LoadCSV: CSV konnte nicht gelesen werden: {e}") from e

        # Fallback: wenn parse_dates_col fehlt → erste Spalte nehmen
        if self.parse_dates_col not in df.columns:
            ts_col = df.columns[0]  # erste Spalte
        else:
            ts_col = self.parse_dates_col

        # Zeitspalte parsen
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

        # Zeitzone anwenden
        if ctx.tz:
            ts = df[ts_col]
            if ts.dt.tz is None:
                df[ts_col] = ts.dt.tz_localize(
                    ctx.tz, nonexistent="shift_forward", ambiguous="infer"
                )
            else:
                df[ts_col] = ts.dt.tz_convert(ctx.tz)

        # Index setzen & Frequenz erzwingen
        df.set_index(ts_col, inplace=True)
        df.sort_index(inplace=True)
        if self.target_freq:
            df = df.asfreq(self.target_freq)

        ctx.df = df
        ctx.log.log(
            "load_csv",
            rows=len(df),
            tz=str(ctx.tz),
            target_freq=self.target_freq,
            n_columns=df.shape[1],
        )

class SelectColumns(PipelineStep):
    """
    Behalte nur die vom Nutzer gewünschten Messspalten (falls vorhanden).
    timestamp ist Index und bleibt immer.
    """
    def apply(self, ctx: DataContext):
        if ctx.selected_cols is None or ctx.df is None:
            ctx.log.log("select_columns", selected=None, present=list(ctx.df.columns))
            return
        present = [c for c in ctx.selected_cols if c in ctx.df.columns]
        missing = [c for c in (ctx.selected_cols) if c not in ctx.df.columns]
        # Nur die gewünschten & vorhandenen Spalten behalten
        ctx.df = ctx.df[present]
        ctx.log.log("select_columns", selected=ctx.selected_cols, kept=present, missing=missing)
        
        
@dataclass(frozen=True)
class SaveOutputs(PipelineStep):
    """
    Speichert bereinigte Daten, rollende Mediane, Tagesmediane und optional das Log.
    Nur CSV-Ausgabe, mit Zeitstempel-Ordner im Pfad:
    air_quality_preprocessing/analysis/out/<timestamp>/
    """

    prefix: str = "results"
    include_log: bool = True
    index: bool = True
    float_format: str | None = None
    date_format: str | None = None  # z.B. "%Y-%m-%d %H:%M:%S%z"

    def _timestamped_dir(self, ctx: DataContext) -> Path:
        """Erzeuge Basis-Output-Ordner: out/<filename>"""
        # Dateiname ohne Extension aus CSV
        csv_name = Path(ctx.csv_file).stem if ctx.csv_file else "data"

        # Projektwurzel bestimmen (2 Ebenen über steps/)
        project_root = Path(__file__).resolve().parents[2]

        base = project_root / "air_quality_preprocessing" / "analysis" / "out" / csv_name
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _save_csv(self, df: pd.DataFrame, path: Path) -> None:
        """Speichert ein DataFrame als CSV mit gesetzten Optionen."""
        df.to_csv(
            path,
            index=self.index,
            float_format=self.float_format,
            date_format=self.date_format,
        )

    def apply(self, ctx: DataContext) -> None:
        base = self._timestamped_dir(ctx)
        files: dict[str, str] = {}

        if ctx.df is not None and not ctx.df.empty:
            p = base / f"{self.prefix}_clean.csv"
            self._save_csv(ctx.df, p)
            files["clean_csv"] = str(p)

        if ctx.df_roll_median is not None:
            p = base / f"{self.prefix}_roll_median.csv"
            self._save_csv(ctx.df_roll_median, p)
            files["roll_median_csv"] = str(p)

        if ctx.df_daily_median is not None:
            p = base / f"{self.prefix}_daily_median.csv"
            self._save_csv(ctx.df_daily_median, p)
            files["daily_median_csv"] = str(p)

        if self.include_log:
            log_df = ctx.log.as_dataframe()
            if log_df is not None and not log_df.empty:
                p = base / f"{self.prefix}_log.csv"
                log_df.to_csv(p, index=False)
                files["log_csv"] = str(p)

        # im Kontext ablegen + loggen
        ctx.output_files.update(files)
        ctx.log.log("save_outputs", files=files, dir=str(base), prefix=self.prefix)