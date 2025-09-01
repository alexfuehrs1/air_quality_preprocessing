from __future__ import annotations
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Iterable

class PipelineStep:
    def apply(self, ctx: DataContext) -> None: ...
    
@dataclass
class StepLogger:
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, kind: str, **payload):
        """
        kind: Art des Log-Events, z. B. 'step_start', 'iqr_remove_outliers', 'load_csv' …
        payload: beliebige Zusatzinfos (z. B. step_name, params, counters)
        """
        self.entries.append({"kind": kind, **payload})

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries) if self.entries else pd.DataFrame(columns=["kind"])

@dataclass
class DataContext:
    csv_file: str
    tz: Optional[str] = None
    selected_cols: Optional[List[str]] = None
    df: Optional[pd.DataFrame] = None
    df_roll_median: Optional[pd.DataFrame] = None
    df_daily_median: Optional[pd.DataFrame] = None
    log: StepLogger = field(default_factory=StepLogger)
    output_files: Dict[str, str] = field(default_factory=dict)

    def na_summary(self, cols: Iterable[str]) -> Dict[str, int]:
        if self.df is None: return {}
        return {c: int(self.df[c].isna().sum()) for c in cols if c in self.df.columns}

    def numeric_cols(self) -> List[str]:
        if self.df is None: return []
        return [c for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c])]