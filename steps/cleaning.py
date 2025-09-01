from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, List
import sys, os, re
import pandas as pd
import pandas.api.types as ptypes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from context import DataContext, PipelineStep


@dataclass(frozen=True)
class RemoveInvalidValues(PipelineStep):
    """
    Gültigkeits-Prüfung per Min/Max-Grenzen (geschlossenes Intervall [lo, hi]).
    Standard: Werte außerhalb des erlaubten Bereichs werden zu NaN gesetzt.

    Features:
    - Case-insensitive Matching: 'eco2' findet auch 'ECO2' in den Spalten, ohne umzubenennen.
    - Optionales to_numeric: CSVs liefern oft Strings; mit coerce_numeric=True wird robust gefloatet.
    - Strategien:
        * "replace_nan" (Default): Ausreißer/ungültige Werte -> NaN (konservativ; Interpolation entscheidet später).
        * "clip": auf [lo, hi] beschneiden (keine NaNs; Daten bleiben erhalten, aber beschnitten).
        * "flag_only": Werte nicht verändern, aber *_invalid_flag setzen (QA/Ablation).
    """

    # Grenzwerte pro Kanal (inklusive lo/hi)
    limits: Dict[str, tuple] = field(default_factory=lambda: {
        "pm2.5":   (0, 1000),
        "pm10":    (0, 1000),
        "eco2":    (400, 8192),
        "tvoc":    (0, 1187),
        "humidity":(20, 90),
        "bmp_temp":(0, 65),
        "dht_temp":(0, 50),
        "pressure":(30000, 110000),
        "co2":     (250, 50000),
    })

    # Wie reagieren wir auf ungültige Werte?
    strategy: str = "replace_nan"  # "replace_nan" | "clip" | "flag_only"

    # Sollen numerische Typen bei Bedarf robust konvertiert werden?
    coerce_numeric: bool = True

    # Case-insensitive Spaltenabgleich (nur die tatsächlich vorhandene Spalte wird verändert)
    case_insensitive: bool = True

    # Optional: zusätzliche Flag-Spalte "<col>_invalid_flag" schreiben (bei flag_only automatisch True)
    add_flag: bool = False

    def _resolve_column(self, df: pd.DataFrame, key: str) -> Optional[str]:
        """
        Findet die tatsächliche Spalte im DataFrame zu einem limits-Key.
        - Bei case_insensitive=True wird key.lower() gegen lower(df.columns) gemappt.
        - Gibt None zurück, wenn die Spalte nicht existiert.
        """
        if key in df.columns:
            return key
        if not self.case_insensitive:
            return None
        lower_map = {c.lower(): c for c in df.columns}
        return lower_map.get(key.lower())

    def apply(self, ctx: DataContext) -> None:
        df = ctx.df

        # Für das Log: NaN-Zusammenfassung vor dem Schritt (nur für definierte Limit-Keys, die es gibt)
        present_cols = [self._resolve_column(df, k) for k in self.limits.keys()]
        present_cols = [c for c in present_cols if c is not None]
        before = ctx.na_summary(present_cols) if present_cols else {}

        changes = {}
        missing = []

        for key, (lo, hi) in self.limits.items():
            col = self._resolve_column(df, key)
            if col is None:
                missing.append(key)
                continue

            s = df[col]

            # Optional robust zu float konvertieren (Strings -> NaN). Sonst nur prüfen, ob numerisch.
            if self.coerce_numeric:
                s_work = pd.to_numeric(s, errors="coerce")
            else:
                if not ptypes.is_numeric_dtype(s):
                    # nicht-numerisch und nicht koerzieren → überspringen, aber loggen
                    changes[col] = {"skipped_non_numeric": True}
                    continue
                s_work = s  # numerisch, weiter geht's

            na_before = int(pd.isna(s_work).sum())

            # Ungültig = innerhalb definierter Werte vorhanden, aber außerhalb [lo, hi]
            # (NaNs bleiben NaN; wir markieren keine fehlenden Werte als ungültig)
            invalid_mask = s_work.notna() & ((s_work < lo) | (s_work > hi))
            n_invalid = int(invalid_mask.sum())

            # Optionales Flag schreiben (bei flag_only immer setzen)
            if self.add_flag or self.strategy == "flag_only":
                flag_col = f"{col}_invalid_flag"
                prev_flags = df.get(flag_col, pd.Series(False, index=df.index))
                df[flag_col] = (prev_flags.fillna(False) | invalid_mask).astype(bool)

            # Strategie anwenden
            if self.strategy == "replace_nan":
                s_out = s_work.where(~invalid_mask, np.nan)

            elif self.strategy == "clip":
                # „Winsorizing“: harte Grenzen anwenden (keine NaNs)
                s_out = s_work.clip(lower=lo, upper=hi)

            elif self.strategy == "flag_only":
                # Werte unverändert lassen (aber Flag evtl. gesetzt)
                s_out = s_work
            else:
                # unbekannte Strategie -> sicherheits-halber unverändert
                s_out = s_work

            # In DataFrame zurückschreiben
            df[col] = s_out

            changes[col] = {
                "limits": (float(lo), float(hi)),
                "n_invalid": n_invalid,
                "na_before": na_before,
                "na_after": int(pd.isna(df[col]).sum()),
                "strategy": self.strategy,
                "coerced_to_numeric": bool(self.coerce_numeric),
            }

        # Log schreiben – inkl. fehlender Keys (in den Daten nicht vorhanden)
        ctx.log.log(
            "remove_invalid_values",
            limits=self.limits,
            present_cols=present_cols,
            missing_keys=missing,
            strategy=self.strategy,
            coerce_numeric=self.coerce_numeric,
            case_insensitive=self.case_insensitive,
            add_flag=self.add_flag or (self.strategy == "flag_only"),
            na_before=before,
            na_after=ctx.na_summary(present_cols) if present_cols else {},
            changes=changes
        )


@dataclass(frozen=True)
class IQRRemoveOutliers(PipelineStep):
    """
    Entfernt oder begrenzt Ausreißer pro Spalte mit der IQR-Methode.
    
    Vorgehen:
      1) Spaltenauswahl: nur echte numerische Messspalten (keine *_flag, bool, Labels).
      2) Für jede Spalte:
         - konvertiere robust zu float (nicht-numerisches → NaN),
         - berechne Q1/Q3 auf den vorhandenen Werten (NaNs ignoriert),
         - IQR = Q3 - Q1 → Intervall [Q1 - k*IQR, Q3 + k*IQR] mit k=iqr_factor.
         - IQR==0: enger Toleranzbereich um Q1 (=Q3), statt zu skippen (verhindert „Hängenbleiber“).
         - wende Strategie an:
             * "replace_nan": markiere Ausreißer als NaN (konservativ; gut vor Interpolation)
             * "cap": schneide auf [lo, hi] zurück (Winsorizing; Daten bleiben erhalten)
             * "none": nur Flag (falls add_flag=True), keine Werteänderung
      3) Logging pro Spalte: erkannte Ausreißer, NaN-Änderungen, verwendete Grenzen.

    Hinweise:
      - IQR ist robust gegen einzelne extreme Werte.
      - Bei (nahezu) konstanter Serie ist IQR≈0; mit einer kleinen eps-Spanne vermeiden wir,
        dass alles fälschlich als Ausreißer markiert wird oder gar nichts passiert.
      - Für strengere/lockerere Erkennung `iqr_factor` anpassen (üblich: 1.5…3.0).
    """

    # explizite Spalten; None => nimm ctx.numeric_cols()
    cols: Optional[Iterable[str]] = None

    # Spannweite der Ausreißer-Schranke (k in [Q1 - k*IQR, Q3 + k*IQR])
    iqr_factor: float = 3.0

    # Mindestanzahl gültiger Werte, um IQR stabil zu schätzen
    min_samples: int = 4

    # Reaktion auf Ausreißer: "replace_nan" | "cap" | "none"
    strategy: str = "replace_nan"

    # Optional Flags schreiben (z. B. für QA/Ablation): <col>_iqr_flag
    add_flag: bool = False

    # Flags/Klassen/Labels ausschließen
    exclude_regex: str = r"(_flag$|class|label|category|id)"

    def apply(self, ctx: DataContext) -> None:
        df = ctx.df

        # 1) Spaltenauswahl filtern
        cand = list(self.cols) if self.cols is not None else ctx.numeric_cols()
        pat = re.compile(self.exclude_regex, flags=re.IGNORECASE)
        cols: List[str] = []
        for c in cand:
            if c not in df.columns:
                continue
            if pat.search(c):                 # *_flag, class, label, ...
                continue
            if ptypes.is_bool_dtype(df[c]):   # bool-Spalten ignorieren
                continue
            if not ptypes.is_numeric_dtype(df[c]):  # nur numerische Dtypes
                continue
            cols.append(c)

        bounds: Dict[str, tuple] = {}
        before = ctx.na_summary(cols) if cols else {}

        for col in cols:
            # robust zu float; nicht-numerisches wird NaN
            ser = pd.to_numeric(df[col], errors="coerce").astype("float64")

            # Nur echte Werte für Quantile/IQR verwenden
            vals = ser.to_numpy(dtype="float64")
            vals = vals[np.isfinite(vals)]
            if vals.size < self.min_samples:
                # Zu wenig Daten → keine sinnvolle IQR-Schätzung
                continue

            # 2) Q1/Q3/IQR
            q1, q3 = np.nanpercentile(vals, [25, 75])
            iqr = q3 - q1

            if not np.isfinite(iqr):
                # Extremfall (z. B. alles NaN) → überspringen
                continue

            if iqr == 0.0:
                # (nahezu) konstante Serie: enger Korridor um Q1 (=Q3)
                eps = 1e-9
                lo, hi = q1 - eps, q3 + eps
            else:
                lo = q1 - self.iqr_factor * iqr
                hi = q3 + self.iqr_factor * iqr

            bounds[col] = (float(lo), float(hi))

            # 3) Ausreißer maske
            mask = (ser < lo) | (ser > hi)

            # Optionales QA-Flag schreiben (bool, index-aligned)
            if self.add_flag:
                flag_col = f"{col}_iqr_flag"
                prev = df.get(flag_col, pd.Series(False, index=df.index))
                df[flag_col] = (prev.fillna(False) | mask.fillna(False)).astype(bool)

            # 4) Strategie anwenden
            if self.strategy == "replace_nan":
                # konservativ: Ausreißer entfernen
                df[col] = ser.where(~mask, np.nan)

            elif self.strategy == "cap":
                # behutsam: auf [lo, hi] winsorizen (keine NaNs)
                df[col] = ser.clip(lower=lo, upper=hi)

            else:
                # "none": nur flaggen/loggen, Werte unverändert lassen
                df[col] = ser

        after = ctx.na_summary(cols) if cols else {}

        # 5) Logging – kompakt & reproduzierbar
        ctx.log.log(
            "iqr_remove_outliers",
            cols=cols,
            iqr_factor=self.iqr_factor,
            strategy=self.strategy,
            min_samples=self.min_samples,
            add_flag=self.add_flag,
            bounds=bounds,            # pro Spalte verwendete [lo, hi]
            na_before=before,
            na_after=after
        )