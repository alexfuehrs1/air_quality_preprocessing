import os
from datetime import datetime

def save_figures(figs: dict, outdir: str = "air_quality_preprocessing/analysis/plots", dpi: int = 200):
    """
    Speichert eine Sammlung von matplotlib-Figuren.

    Args:
        figs: dict[str, matplotlib.figure.Figure]
              Keys = gewünschte Dateinamen (ohne Endung),
              Values = Figure-Objekte
        outdir: Zielordner
        dpi: Auflösung (DPI) für PNGs
    """
    os.makedirs(outdir, exist_ok=True)

    for name, fig in figs.items():
        if fig is None:
            continue  # falls Plot nicht erzeugt wurde
        path = os.path.join(outdir, f"{name}.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[✓] Saved {path}")