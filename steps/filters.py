from __future__ import annotations
import pandas as pd
import pandas.api.types as ptypes
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Union
import sys, os, re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from context import DataContext, PipelineStep

@dataclass(frozen=True)
class HampelFilter(PipelineStep):
    """
    Robust-Filter (Hampel) gegen Ausreißer.
    Funktionsprinzip:
      - lokaler Median (med) + MAD (Median Absolute Deviation) im rollenden Fenster
      - Punkt ist Ausreißer, wenn |x - med| > n_sigmas * (c * MAD), c≈1.4826
    Reaktion je nach 'strategy':
      - "replace_nan": Ausreißer → NaN (konservativ; gut vor Interpolation)
      - "cap": auf [med ± n_sigmas * c*MAD] begrenzen (behutsam)
      - sonst: unverändert (nur markieren, falls Flags aktiv sind)
    Wichtiger Hinweis: `center=True` nutzt Zukunftsdaten → für reines Cleaning ok,
    für ML-Features/Labels (leakagefrei) eher `center=False` verwenden.
    """

    # Welche Spalten? None → numerische Spalten aus dem Context
    cols: Optional[Iterable[str]] = None

    # Fensterlänge (in PUNKTEN, nicht Minuten). Wird intern auf ungerade Zahl gezwungen.
    window: int = 25

    # Schwellfaktor (Anzahl "Standardabweichungen" auf Basis von MAD)
    n_sigmas: float = 3.0

    # Reaktionsmodus: "replace_nan" | "cap" | "none"
    strategy: str = "replace_nan"

    # Zentriertes Fenster (center=True) oder rein kausal (center=False)?
    center: bool = True

    # Soll zusätzlich eine Flag-Spalte "<col>_hampel_flag" (bool) geschrieben werden?
    add_flag: bool = True

    # Spalten, die grundsätzlich NICHT gefiltert werden sollen (Flags, Labels, IDs, …)
    exclude_regex: str = r"(_flag$|class|label|category|id)"

    # Numerische Untergrenze für sigma: verhindert, dass bei MAD=0 "alles" als Ausreißer gilt
    eps: float = 1e-9

    def apply(self, ctx: DataContext) -> None:
        df = ctx.df

        # Spalten bestimmen: explizit übergeben oder numerische aus dem Context
        cols = list(self.cols) if self.cols is not None else ctx.numeric_cols()

        # Regex für Spalten-Ausschluss (Flags, Labels, etc.)
        pat = re.compile(self.exclude_regex, flags=re.IGNORECASE)

        # Fenster ungerade erzwingen (klassischer Hampel) & mind. 3
        # `| 1` setzt das niederwertigste Bit → macht jede Zahl ungerade.
        k = max(3, int(self.window) | 1)

        # Faktor zur Approx. Std-Abweichung bei normalverteilten Daten
        c = 1.4826

        changes: Dict[str, Dict] = {}

        for col in cols:
            # Nur echte numerische Messspalten bearbeiten
            if col not in df.columns:
                continue
            if pat.search(col):                  # *_flag, class, label, …
                continue
            if ptypes.is_bool_dtype(df[col]):    # keine Bool-Spalten
                continue
            if not ptypes.is_numeric_dtype(df[col]):  # nur numerische Dtypes
                continue

            # Serie robust floaten (coerce → non-numerics werden NaN)
            s = pd.to_numeric(df[col], errors="coerce")
            na_before = int(s.isna().sum())

            # Rolling Median & MAD.
            # Achtung: center=True nutzt Zukunftsdaten → kann in ML-Pipelines Leakage sein.
            med = s.rolling(k, center=self.center, min_periods=1).median()
            mad = (s - med).abs().rolling(k, center=self.center, min_periods=1).median()

            # Robuste Std-Schätzung; mit eps, um bei MAD=0 nicht alles zu flaggen
            sigma = np.maximum(c * mad, self.eps)

            # Ausreißerbedingung
            mask = (s - med).abs() > (self.n_sigmas * sigma)
            mask = mask.fillna(False)

            # Optional: QA-Flag schreiben
            if self.add_flag:
                flag_col = f"{col}_hampel_flag"
                # bool-cast verhindert spätere Interpolations-/IQR-Probleme
                flags_prev = df.get(flag_col, pd.Series(False, index=df.index))
                df[flag_col] = (flags_prev.fillna(False) | mask).astype(bool)

            # Strategie anwenden
            if self.strategy == "replace_nan":
                # Konservativ: problematische Punkte entfernen
                s_out = s.mask(mask, np.nan)

            elif self.strategy == "cap":
                # Behutsam: auf zulässigen Bereich kappen
                low = med - self.n_sigmas * sigma
                high = med + self.n_sigmas * sigma
                # clip respektiert NaNs in low/high
                s_out = s.clip(lower=low, upper=high)

            else:
                # "none" oder unbekannt → unverändert lassen
                s_out = s

            df[col] = s_out

            changes[col] = {
                "detected": int(mask.sum()),
                "na_before": na_before,
                "na_after": int(df[col].isna().sum()),
                "strategy": self.strategy,
                "window": k,
                "center": self.center,
                "n_sigmas": self.n_sigmas,
            }

        # Ein Log-Eintrag (nicht pro Spalte), kompakt & reproduzierbar
        ctx.log.log(
            "hampel_filter",
            cols=[c for c in cols if c in df.columns],
            window=k,
            n_sigmas=self.n_sigmas,
            center=self.center,
            strategy=self.strategy,
            changes=changes
        )


@dataclass(frozen=True)
class RateOfChangeCheck(PipelineStep):
    """
    Rate-of-Change (RoC) Plausibilitätsprüfung für Zeitreihen.
    - Erkennt unplausible Sprünge (Δ pro Minute) anhand absoluter UND/ODER relativer Grenzwerte.
    - Markiert alle betroffenen Zeitstempel in *_roc_flag (bool).
    - Reagiert je Spalte mit einer Strategie:
        * "replace_nan": unplausible Werte durch NaN ersetzen (konservativ)
        * "cap": Sprünge sanft auf erlaubte Rate begrenzen (behutsam)
        * "flag_only": Werte lassen, nur markieren (transparent)
    """
    # Falls None: es werden alle "numerischen" Spalten aus dem Context geprüft.
    cols: Optional[Iterable[str]] = None

    # Absolute Grenzwerte (Einheiten pro Minute), z.B. 80 µg/m³/min für PM10.
    max_abs_per_min: Dict[str, float] = field(default_factory=lambda: {
        "pm2.5": 50.0,
        "pm10": 80.0,
        "TVOC": 200.0,
        "ECO2": 500.0,
        "Temperature": 3.0,
        "Humidity": 10.0,
        "Pressure": 1.0
    })

    # Relative Grenzwerte (Fraktion pro Minute), z.B. 1.0 = 100%/min.
    # Wird gegen max(|x_{t-1}|, floor) geprüft, um Miniwerte nicht zu überbestrafen.
    max_rel_per_min: Dict[str, float] = field(default_factory=lambda: {
        "pm2.5": 1.0,
        "pm10": 1.0,
        "TVOC": 1.5,
        "ECO2": 0.5
    })

    # Untere Schranke für den relativen Nenner
    rel_level_floor: Dict[str, float] = field(default_factory=lambda: {
        "pm2.5": 5.0,
        "pm10": 10.0,
        "TVOC": 50.0,
        "ECO2": 400.0
    })

    # Strategie je Kanal (oder ein globaler String für alle Kanäle).
    strategy: Union[str, Dict[str, str]] = field(default_factory=lambda: {
        "pm2.5": "replace_nan",
        "pm10": "replace_nan",
        "TVOC": "replace_nan",
        "ECO2": "cap",
        "Temperature": "cap",
        "Humidity": "cap",
        "Pressure": "flag_only"
    })

    def apply(self, ctx: DataContext) -> None:
        df = ctx.df
        # Spaltenliste bestimmen: explizit übergeben oder alle numerischen
        cols = list(self.cols) if self.cols is not None else ctx.numeric_cols()

         # Zeitindex sicherstellen (monoton)
        if not df.index.is_monotonic_increasing:
            df.sort_index(inplace=True)

        # Δt pro Zeile in Minuten (erste Zeile NaN). Nicht-positive Δt → NaN (werden ignoriert)
        dt_min = (df.index.to_series().diff().dt.total_seconds() / 60.0)
        dt_min = dt_min.mask(dt_min <= 0, np.nan)
        changes = {}

        for col in cols:
            # Wenn für diese Spalte weder abs. noch rel. Threshold definiert ist → überspringen
            if (col not in self.max_abs_per_min) and (col not in self.max_rel_per_min):
                continue
            
            # Serie als float (für Rate-Berechnungen)
            s = df[col].astype(float)
            ds = s.diff()

            # Grenzwerte auflösen (falls nicht vorhanden → +∞ = „kein Check“)
            abs_thr = float(self.max_abs_per_min.get(col, np.inf))
            rel_thr = float(self.max_rel_per_min.get(col, np.inf))
            floor   = float(self.rel_level_floor.get(col, 0.0))

            # Änderungsrate pro Minute
            rate = ds / dt_min

            # absolute Regel: |rate| > abs_thr
            abs_viol = rate.abs() > abs_thr

            # relative Regel: |rate| > rel_thr * max(|prev|, floor)
            prev = s.shift(1).abs().clip(lower=floor)
            rel_viol = (rate.abs() > (rel_thr * prev))

            # Verstoß, wenn eine der beiden Regeln greift
            mask = abs_viol | rel_viol

            # Flag-Spalte (bool) setzen/akkumulieren: Zeigt an, dass an diesem Timestamp ein RoC-Verstoß vorlag
            flag_col = f"{col}_roc_flag"
            df[flag_col] = df.get(flag_col, False) | mask.fillna(False)

            # Statistik für Log
            na_before = int(s.isna().sum())
            n_hits = int(mask.fillna(False).sum())

            # Strategie je Spalte bestimmen (Dict oder globaler String)
            strat = self.strategy[col] if isinstance(self.strategy, dict) else self.strategy

            if strat == "replace_nan":
                # Konservativ: unplausible Werte entfernen
                s_out = s.mask(mask, np.nan)

            elif strat == "cap":
                # Behutsam: Veränderung auf erlaubte Rate begrenzen
                # erlaubte Rate/min = min(abs_thr, rel_thr * prev)
                sign = np.sign(rate).fillna(0.0)
                allowed = np.minimum(abs_thr, rel_thr * prev)
                capped = s.shift(1) + sign * allowed * dt_min
                s_out = s.where(~mask, capped)

            elif strat == "flag_only":
                # Nur markieren, Werte unverändert lassen
                s_out = s
            else:
                s_out = s

            df[col] = s_out

            # Logging-Daten je Spalte sammeln
            changes[col] = {
                "abs_thr_per_min": None if np.isinf(abs_thr) else abs_thr,
                "rel_thr_per_min": None if np.isinf(rel_thr) else rel_thr,
                "floor": floor,
                "detected": n_hits,
                "na_before": na_before,
                "na_after": int(df[col].isna().sum()),
                "strategy": strat,  
            }

        # Ein Log-Eintrag für den gesamten Schritt (nach allen Spalten)
        ctx.log.log("rate_of_change_check", cols=cols, changes=changes)


@dataclass(frozen=True)
class FlatlineCheck(PipelineStep):
    """
    Erkennung von 'Flatlines' (Sensor hängt/steht) in Zeitreihen.

    Idee:
      - Ein Sensor ist 'flat', wenn seine rollende Standardabweichung über eine
        Mindestdauer nahe 0 ist (unterhalb einer Toleranz).
      - Wir erkennen Flatlines zunächst in der "Fenstermitte" mittels rstd < tolerance
        und erweitern dann das Signal auf den gesamten Flatline-Abschnitt.

    Wichtige Hinweise:
      - `min_duration_minutes` wird über die Index-Schrittweite auf eine Anzahl
        Punkte abgebildet (bei DatetimeIndex). Fallback: direkt als Punkteanzahl.
      - Standardmäßig wird zentriert gearbeitet (centered Erkennung); das ist OK
        für Cleaning/QA. Für leakage-freies Feature-Engineering eher eine kausale
        Alternative (right-aligned Fenster) verwenden.
    """

    # Welche Spalten prüfen? None → alle numerischen aus dem Context
    cols: Optional[Iterable[str]] = None

    # Mindestdauer einer Flatline (in MINUTEN); wird auf Punkte umgerechnet
    min_duration_minutes: int = 30

    # Schwellwert für die rollende Std-Abweichung (kleiner = strenger)
    tolerance: float = 0.01

    # Reaktion auf erkannte Flatlines:
    #  - "replace_nan": Flatline-Werte werden auf NaN gesetzt (konservativ)
    #  - "freeze_first": Werte eines Flatline-Abschnitts auf den Anfangswert setzen
    #  - alles andere: nur markieren (falls add_flag=True), keine Werteänderung
    strategy: str = "replace_nan"

    # Optionales Flagging der erkannten Flatlines als <col>_flatline_flag (bool)
    add_flag: bool = True

    # Spalten, die grundsätzlich nicht geprüft werden (Flags, Labels, IDs etc.)
    exclude_regex: str = r"(_flag$|class|label|category|id)"

    def _minutes_to_points(self, idx: pd.DatetimeIndex, minutes: int) -> int:
        """
        Konvertiert Minuten in eine Punktanzahl anhand der medianen Index-Schrittweite.
        Fallback: direkt 'minutes' als Punkte.
        """
        if isinstance(idx, pd.DatetimeIndex) and len(idx) >= 2:
            dt_sec = pd.Series(idx).diff().dt.total_seconds().median()
            if pd.notna(dt_sec) and dt_sec > 0:
                step_min = dt_sec / 60.0
                # mindestens 2 Punkte; runde sinnvoll
                return max(2, int(round(minutes / step_min)))
        # ungeeigneter Index → behandle 'minutes' wie Punkte
        return max(2, int(minutes))

    def apply(self, ctx: DataContext) -> None:
        df = ctx.df
        # Spaltenauswahl (explizit oder numerische aus dem Context)
        cols = list(self.cols) if self.cols is not None else ctx.numeric_cols()

        # Fensterbreite in PUNKTEN aus Minuten ableiten
        w = self._minutes_to_points(df.index, self.min_duration_minutes)

        changes: Dict[str, Dict] = {}
        pat = re.compile(self.exclude_regex, flags=re.IGNORECASE)

        for col in cols:
            # Nur echte numerische Messspalten, keine Flags/Bools/etc.
            if col not in df.columns:
                continue
            if pat.search(col):
                continue
            if ptypes.is_bool_dtype(df[col]):
                continue
            if not ptypes.is_numeric_dtype(df[col]):
                continue

            s = pd.to_numeric(df[col], errors="coerce")
            na_before = int(s.isna().sum())

            # 1) Rolling-Std über zentriertes Fenster
            #    center=True bedeutet: der Wert zur Zeit t nutzt auch Zukunftswerte (Leakage in ML-Kontext!).
            rstd = s.rolling(window=w, min_periods=w).std()

            # 2) "Treffer" genau dort, wo das Fenster mittig die Std < Toleranz hat
            center_hits = (rstd < self.tolerance).astype(float)

            # 3) Erweiterung auf gesamten Abschnitt:
            #    rolling-max über dasselbe Fenster "füllt" die Flatline-Bereiche aus
            flat_mask = center_hits.rolling(window=w, min_periods=1).max().fillna(False).astype(bool)

            # Optional: Flatline-Flag schreiben (bool)
            if self.add_flag:
                flag_col = f"{col}_flatline_flag"
                prev_flags = df.get(flag_col, pd.Series(False, index=df.index))
                df[flag_col] = (prev_flags.fillna(False) | flat_mask).astype(bool)

            # 4) Strategie anwenden
            if self.strategy == "replace_nan":
                # Konservativ: Flatlines entfernen → spätere Interpolation entscheidet über Rekonstruktion
                s_out = s.mask(flat_mask, np.nan)

            elif self.strategy == "freeze_first":
                # Zusammenhängende True-Segmente (Runs) in flat_mask identifizieren
                # run_id steigt um 1 bei jedem Wechsel True<->False
                run_id = (flat_mask != flat_mask.shift(fill_value=False)).cumsum()

                # Für True-Positionen: ersten Wert des jeweiligen Runs bestimmen
                # (non-True-Runs bleiben NaN, stört nicht, da wir nur bei flat_mask=True ersetzen)
                first_vals = s.where(flat_mask).groupby(run_id).transform("first")

                # Innerhalb der Flatline alle Werte auf den ersten Run-Wert setzen
                s_out = s.where(~flat_mask, first_vals)
            else:
                # Nur markieren, keine Werteänderung
                s_out = s

            df[col] = s_out

            changes[col] = {
                "window_points": w,
                "tolerance": self.tolerance,
                "flagged": int(flat_mask.sum()),
                "na_before": na_before,
                "na_after": int(df[col].isna().sum()),
                "strategy": self.strategy,
            }

        ctx.log.log("flatline_check", cols=[c for c in cols if c in df.columns],
                    window_points=w, min_duration_minutes=self.min_duration_minutes,
                    tolerance=self.tolerance, changes=changes)