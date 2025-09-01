from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from context import DataContext, PipelineStep

@dataclass(frozen=True)
class PMHierarchyCheck(PipelineStep):
    strict: bool = False 

    def apply(self, ctx: DataContext) -> None:
        cols = {"pm2.5", "pm10"}
        if not cols.issubset(set(ctx.df.columns)):
            ctx.log.log("pm_hierarchy_check", violations=0, strict=self.strict, note="pm columns missing")
            return

        pm25 = ctx.df["pm2.5"]
        pm10 = ctx.df["pm10"]

        # 1) Klassischer Regel-Verstoß: beide vorhanden & pm2.5 > pm10
        mask_violation = pm25.notna() & pm10.notna() & (pm25 > pm10)
        n_viol = int(mask_violation.sum())
        if n_viol:
            ctx.df.loc[mask_violation, ["pm2.5", "pm10"]] = np.nan

        n_strict_pairs = 0
        if self.strict:
            # 2) Strikte Synchronisierung: wenn genau einer NaN ist, ziehe den anderen nach
            only_pm25_nan = pm25.isna() & pm10.notna()
            only_pm10_nan = pm10.isna() & pm25.notna()
            n_strict_pairs = int(only_pm25_nan.sum() + only_pm10_nan.sum())
            if n_strict_pairs:
                ctx.df.loc[only_pm25_nan, "pm10"] = np.nan
                ctx.df.loc[only_pm10_nan, "pm2.5"] = np.nan

        ctx.log.log("pm_hierarchy_check",
                    violations=n_viol,
                    strict=self.strict,
                    strict_pairs=n_strict_pairs)