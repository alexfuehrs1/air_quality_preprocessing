from copy import deepcopy
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from context import DataContext

def run_with_snapshots(pipe, csv_file, tz=None, selected_columns=None):
    """
    Führt die Pipeline step-by-step aus und sammelt nach JEDEM Schritt einen Snapshot.
    Returns:
      ctx  : finaler DataContext
      snaps: Liste[(step_name, df_copy)]
    """
    ctx = DataContext(csv_file=csv_file, tz=tz, selected_cols=selected_columns)
    snaps = []
    for step in pipe.steps:
        name = step.__class__.__name__
        ctx.log.log("step_start", step_name=name)
        step.apply(ctx)
        # Deepcopy, damit spätere Schritte das Snapshot nicht verändern
        snaps.append((name, deepcopy(ctx.df)))
    ctx.log.log("pipeline_done", n_steps=len(pipe.steps))
    return ctx, snaps