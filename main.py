from analysis.important_plots import (plot_impact_bar, plot_missingness_heatmap,
                                plot_dist_before_after, plot_interpolation_stats,
                                plot_rate_of_change_effect, plot_flatline_effect,)
from analysis.log_kpis import collect_logger_kpis
from analysis.impacts_steps import step_impacts
from analysis.ablation import compare_variants, plot_ablation_summary
from analysis.save_output import save_figures
from analysis.run_snapshots import run_with_snapshots
from pipeline import make_default_pipeline


# 1) Läufe
path = "air_quality_preprocessing/analysis/input/"
filename = "" # your filename

os_path = path + filename
print(os_path)
pipe = make_default_pipeline(window="24h", max_gap_minutes=30)
ctx, snaps = run_with_snapshots(pipe, csv_file=os_path, tz="Europe/Berlin",)

# 2) Metriken
imp = step_impacts(snaps)
kpi = collect_logger_kpis(ctx)

# 3) Plots bauen
fig1 = plot_impact_bar(impacts=imp, by="step")                  # Impact je Schritt
fig2 = plot_missingness_heatmap(snaps[-1][1])                   # Missingness final
fig3 = plot_dist_before_after(snaps, column="pm2_5")            # Verteilung before/after
fig4 = plot_interpolation_stats(kpi)                            # Interpolation
fig5 = plot_rate_of_change_effect(kpi)                          # RoC
fig6 = plot_flatline_effect(kpi)                                # Flatline

# 4) Ablation (optional)
variants = {"full": {},
            "no_hampel": {"enable_hampel": False},
            "no_roc": {"enable_roc": False},
            "no_flatline": {"enable_flatline": False}}
ablation = compare_variants(os_path, tz="Europe/Berlin",
                            selected_cols=["pm2_5","pm10","eco2"], variants=variants)
fig_ablate = plot_ablation_summary(ablation, metric="n_to_nan")

figs_to_save = {
    "impact_per_step": fig1,
    "missingness_heatmap": fig2,
    "distribution_pm25": fig3,
    "interpolation_stats": fig4,
    "roc_effects": fig5,
    "flatline_effects": fig6,
    "ablation_summary": fig_ablate,
}
outputdir = "air_quality_preprocessing/analysis/plots/" + filename.replace(".csv", "") + "/"
save_figures(figs_to_save, outdir=outputdir, dpi=300)