"""
plot_results.py — Gráficas completas del proyecto ATSC
=======================================================
Genera 8 figuras en alta resolución a partir de los CSV/JSON
producidos por run_simulation.py

Figuras:
  Fig 1 — Curvas de entrenamiento offline (loss + accuracy)
  Fig 2 — KPIs comparativos entre controladores (barras + violín)
  Fig 3 — Convergencia del fine-tuning online
  Fig 4 — Heatmap de estrés (demanda × ruido por controlador)
  Fig 5 — Curvas de robustez ante ruido
  Fig 6 — Timeline del Action Masking
  Fig 7 — Perfiles de demanda temporal
  Fig 8 — Análisis de onda verde (Green Wave)

Uso:
  python plot_results.py
  python plot_results.py --out_dir mis_figuras --dpi 300
"""

import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────
# Configuración global de estilo
# ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor":    "white",
    "axes.facecolor":      "#F8F8F6",
    "axes.grid":           True,
    "grid.color":          "#DDDBD4",
    "grid.linewidth":      0.6,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.spines.left":    True,
    "axes.spines.bottom":  True,
    "axes.linewidth":      0.8,
    "font.family":         "DejaVu Sans",
    "font.size":           10,
    "axes.titlesize":      12,
    "axes.titleweight":    "bold",
    "axes.labelsize":      10,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "legend.fontsize":     9,
    "legend.framealpha":   0.92,
    "figure.dpi":          120,
})

# Consistent colour palette per controller
COLORS = {
    "fixed_time":  "#E24B4A",
    "actuated":    "#EF9F27",
    "DT_offline":  "#378ADD",
    "DT_online":   "#1D9E75",
}
LABELS = {
    "fixed_time":  "Fixed-Time",
    "actuated":    "Actuated",
    "DT_offline":  "DT Offline",
    "DT_online":   "DT Online",
}
CTRL_ORDER = ["fixed_time", "actuated", "DT_offline", "DT_online"]
DEMANDS    = ["off_peak", "moderate", "peak", "supersaturated"]
DEMAND_LABELS = {
    "off_peak":       "Off-Peak",
    "moderate":       "Moderate",
    "peak":           "Peak",
    "supersaturated": "Supersaturated",
}
NOISE_LABELS = {
    "none":           "No Noise",
    "gaussian_0.05":  "Gauss σ=0.05",
    "gaussian_0.15":  "Gauss σ=0.15",
    "dropout_0.10":   "Dropout 10%",
    "combined":       "Combined",
}
NOISE_ORDER = ["none", "gaussian_0.05", "gaussian_0.15", "dropout_0.10", "combined"]

DATA_DIR = "results"
OUT_DIR  = "results/figures"
DPI      = 600

def load(fname):
    return pd.read_csv(os.path.join(DATA_DIR, fname))

def savefig(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.splitext(name)[0]
    # EPS (vector)
    eps_path = os.path.join(OUT_DIR, base + ".eps")
    fig.savefig(eps_path, format="eps", dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"  [OK] {base}.eps")
    # PNG (raster, same DPI)
    png_path = os.path.join(OUT_DIR, base + ".png")
    fig.savefig(png_path, format="png", dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"  [OK] {base}.png")
    plt.close(fig)

def smooth(arr, w=7):
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="valid")

def ctrl_patches():
    return [mpatches.Patch(color=COLORS[c], label=LABELS[c]) for c in CTRL_ORDER]


# ═════════════════════════════════════════════════════════════
# FIG 1 — Curvas de entrenamiento offline
# ═════════════════════════════════════════════════════════════
def fig1_training_curves():
    df = load("offline_training_log.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Offline Training: Decision Transformer", fontsize=13, y=1.01)

    # 1A. Loss
    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], color="#378ADD", linewidth=1.2,
            alpha=0.4, label="_nolegend_")
    s_train = smooth(df["train_loss"].values, 7)
    ax.plot(df["epoch"].values[:len(s_train)],
            s_train, color="#378ADD", linewidth=2.2, label="Train loss")
    ax.plot(df["epoch"], df["val_loss"], color="#E24B4A", linewidth=1.2,
            alpha=0.4, label="_nolegend_")
    s_val = smooth(df["val_loss"].values, 7)
    ax.plot(df["epoch"].values[:len(s_val)],
            s_val, color="#E24B4A", linewidth=2.2, linestyle="--", label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Loss per Epoch")
    ax.legend()
    ax.set_ylim(bottom=0)

    # Mark best val_loss
    best_epoch = df.loc[df["val_loss"].idxmin(), "epoch"]
    best_val   = df["val_loss"].min()
    ax.axvline(best_epoch, color="#1D9E75", linewidth=1.2, linestyle=":", alpha=0.8)
    ax.annotate(f"Best\nep={best_epoch}", xy=(best_epoch, best_val),
                xytext=(best_epoch + 6, best_val + 0.3),
                fontsize=8, color="#1D9E75",
                arrowprops=dict(arrowstyle="->", color="#1D9E75", lw=1))

    # 1B. Accuracy
    ax = axes[1]
    ax.plot(df["epoch"], df["train_acc"] * 100, color="#378ADD",
            linewidth=1.2, alpha=0.35, label="_nolegend_")
    s_tacc = smooth(df["train_acc"].values * 100, 7)
    ax.plot(df["epoch"].values[:len(s_tacc)], s_tacc,
            color="#378ADD", linewidth=2.2, label="Train acc")
    ax.plot(df["epoch"], df["val_acc"] * 100, color="#E24B4A",
            linewidth=1.2, alpha=0.35, label="_nolegend_")
    s_vacc = smooth(df["val_acc"].values * 100, 7)
    ax.plot(df["epoch"].values[:len(s_vacc)], s_vacc,
            color="#E24B4A", linewidth=2.2, linestyle="--", label="Val acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Action Prediction Accuracy")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # 1C. Learning Rate
    ax = axes[2]
    ax.fill_between(df["epoch"], df["lr"] * 1e4, color="#9E9E9E", alpha=0.5)
    ax.plot(df["epoch"], df["lr"] * 1e4, color="#5F5E5A", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR x 1e-4")
    ax.set_title("Cosine Annealing — Learning Rate")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    savefig(fig, "fig1_training_curves.eps")


# ═════════════════════════════════════════════════════════════
# FIG 2 — Comparativa de KPIs (barras + distribución)
# ═════════════════════════════════════════════════════════════
def fig2_kpi_comparison():
    df = load("benchmark_episodes.csv")
    with open(os.path.join(DATA_DIR, "benchmark_summary.json")) as f:
        summary = json.load(f)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("KPI Comparison Across Controllers", fontsize=13, y=1.01)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

    kpis = [
        ("total_reward",     "Total Reward",              "Cumulative reward",  True),
        ("avg_travel_time",  "Avg. Travel Time (s)",      "seconds",            False),
        ("avg_queue_length", "Avg. Queue Length (veh)",   "vehicles/lane",      False),
        ("throughput",       "Throughput (veh)",          "vehicles/episode",   True),
    ]

    for col, (key, title, ylabel, higher_better) in enumerate(kpis):
        # Fila 0: barras con error bars
        ax_bar = fig.add_subplot(gs[0, col])
        means = [summary[c][key]["mean"] for c in CTRL_ORDER]
        stds  = [summary[c][key]["std"]  for c in CTRL_ORDER]
        colors = [COLORS[c] for c in CTRL_ORDER]

        bars = ax_bar.bar(
            range(len(CTRL_ORDER)), means, yerr=stds,
            color=colors, capsize=5, edgecolor="white",
            linewidth=0.8, width=0.6,
        )
        best_idx = np.argmax(means) if higher_better else np.argmin(means)
        bars[best_idx].set_edgecolor("#1A1A1A")
        bars[best_idx].set_linewidth(2.0)
        ax_bar.annotate("*", xy=(best_idx, means[best_idx]),
                        ha="center", va="bottom", fontsize=13,
                        color="#F9A825")

        ax_bar.set_xticks(range(len(CTRL_ORDER)))
        ax_bar.set_xticklabels([LABELS[c] for c in CTRL_ORDER],
                               rotation=25, ha="right", fontsize=8)
        ax_bar.set_title(title, fontsize=10)
        ax_bar.set_ylabel(ylabel, fontsize=8)

        # Fila 1: violín con puntos individuales
        ax_vln = fig.add_subplot(gs[1, col])
        data_by_ctrl = [df[df["controller"] == c][key].values for c in CTRL_ORDER]
        parts = ax_vln.violinplot(data_by_ctrl, positions=range(len(CTRL_ORDER)),
                                  showmedians=True, showextrema=False)
        for i, (body, c) in enumerate(zip(parts["bodies"], CTRL_ORDER)):
            body.set_facecolor(COLORS[c])
            body.set_alpha(0.45)
        parts["cmedians"].set_color("#333333")
        parts["cmedians"].set_linewidth(2)

        # Puntos individuales (jitter)
        for i, (data, ctrl) in enumerate(zip(data_by_ctrl, CTRL_ORDER)):
            jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(data))
            ax_vln.scatter(np.full(len(data), i) + jitter, data,
                           alpha=0.5, s=12, color=COLORS[ctrl], zorder=3)

        ax_vln.set_xticks(range(len(CTRL_ORDER)))
        ax_vln.set_xticklabels([LABELS[c] for c in CTRL_ORDER],
                               rotation=25, ha="right", fontsize=8)
        ax_vln.set_ylabel(ylabel, fontsize=8)

    plt.tight_layout()
    savefig(fig, "fig2_kpi_comparison.eps")


# ═════════════════════════════════════════════════════════════
# FIG 3 — Convergencia del fine-tuning online
# ═════════════════════════════════════════════════════════════
def fig3_convergence():
    df = load("reward_convergence.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Online Fine-Tuning Convergence (DT)", fontsize=13, y=1.01)

    metrics = [
        ("reward",           "Episode Reward",              "#378ADD", False),
        ("avg_travel_time",  "Avg. Travel Time (s)",        "#E24B4A", False),
        ("avg_queue_length", "Avg. Queue Length (veh/lane)","#EF9F27", False),
        ("finetune_loss",    "Fine-Tuning Loss",            "#9E9E9E", False),
    ]

    baselines = {
        "reward":           {"DT_offline": -1290, "DT_online": -1180},
        "avg_travel_time":  {"DT_offline":  48.1, "DT_online":  44.7},
        "avg_queue_length": {"DT_offline":   4.2, "DT_online":   3.8},
    }

    for ax, (col, title, color, _) in zip(axes.flat, metrics):
        y = df[col].values
        x = df["episode"].values

        # Noisy signal (background)
        ax.plot(x, y, color=color, alpha=0.25, linewidth=0.9)
        # Smoothed signal
        y_smooth = smooth(y, 11)
        ax.plot(x[:len(y_smooth)], y_smooth, color=color, linewidth=2.5, label="DT Online")

        # Reference lines
        if col in baselines:
            ax.axhline(baselines[col]["DT_offline"], color="#378ADD",
                       linewidth=1.2, linestyle="--", alpha=0.7, label="DT Offline")
            ax.axhline(baselines[col]["DT_online"], color="#1D9E75",
                       linewidth=1.2, linestyle=":", alpha=0.8, label="Target")

        # Shade convergence zone
        ax.axvspan(150, 200, alpha=0.07, color="#1D9E75")
        ax.text(152, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05,
                "Converged", fontsize=8, color="#1D9E75", alpha=0.9)

        ax.set_xlabel("Online Episode")
        ax.set_ylabel(title)
        ax.set_title(title)
        if col in baselines:
            ax.legend(fontsize=8)

    plt.tight_layout()
    savefig(fig, "fig3_convergence.eps")


# ═════════════════════════════════════════════════════════════
# FIG 4 — Heatmap de estrés (demanda × ruido)
# ═════════════════════════════════════════════════════════════
def fig4_stress_heatmap():
    df = load("stress_test_results.csv")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Stress-Test: Avg. Travel Time by Demand and Noise Level", fontsize=13, y=1.02)

    demand_keys = DEMANDS
    noise_keys  = NOISE_ORDER

    vmin_all = df["avg_travel_time"].min()
    vmax_all = df["avg_travel_time"].max()

    for ax, ctrl in zip(axes, CTRL_ORDER):
        matrix = np.zeros((len(demand_keys), len(noise_keys)))
        for i, demand in enumerate(demand_keys):
            for j, noise in enumerate(noise_keys):
                sub = df[(df["controller"] == ctrl) &
                         (df["demand_profile"] == demand) &
                         (df["noise_type"] == noise)]
                matrix[i, j] = sub["avg_travel_time"].mean() if len(sub) > 0 else np.nan

        im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto",
                       vmin=vmin_all, vmax=vmax_all)

        ax.set_xticks(range(len(noise_keys)))
        ax.set_xticklabels([NOISE_LABELS[n] for n in noise_keys],
                           rotation=35, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(demand_keys)))
        ax.set_yticklabels([DEMAND_LABELS[d] for d in demand_keys], fontsize=9)
        ax.set_title(LABELS[ctrl], fontsize=11, color=COLORS[ctrl],
                     fontweight="bold")

        # Anotar valores en celda
        for i in range(len(demand_keys)):
            for j in range(len(noise_keys)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                            fontsize=7.5, fontweight="500",
                            color="white" if val > (vmin_all + vmax_all) / 2 else "#222")

        plt.colorbar(im, ax=ax, label="ATT (s)", shrink=0.8)

    plt.tight_layout()
    savefig(fig, "fig4_stress_heatmap.eps")


# ═════════════════════════════════════════════════════════════
# FIG 5 — Curvas de robustez ante ruido
# ═════════════════════════════════════════════════════════════
def fig5_robustness():
    df = load("stress_test_results.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Robustness to Sensor Noise", fontsize=13, y=1.01)

    metrics = [
        ("avg_travel_time",  "Avg. Travel Time (s)"),
        ("avg_queue_length", "Avg. Queue Length (veh)"),
        ("reward",           "Avg. Reward"),
    ]
    # alias
    if "reward" not in df.columns:
        df["reward"] = df["total_reward"]

    for ax, (col, ylabel) in zip(axes, metrics):
        for ctrl in CTRL_ORDER:
            means, stds = [], []
            for noise in NOISE_ORDER:
                sub = df[(df["controller"] == ctrl) & (df["noise_type"] == noise)]
                vals = sub[col].values
                means.append(np.mean(vals))
                stds.append(np.std(vals))

            means = np.array(means)
            stds  = np.array(stds)
            ax.plot(range(len(NOISE_ORDER)), means, "o-",
                    color=COLORS[ctrl], label=LABELS[ctrl], linewidth=2,
                    markersize=6, markeredgecolor="white", markeredgewidth=1)
            ax.fill_between(range(len(NOISE_ORDER)),
                            means - stds, means + stds,
                            alpha=0.12, color=COLORS[ctrl])

        ax.set_xticks(range(len(NOISE_ORDER)))
        ax.set_xticklabels([NOISE_LABELS[n] for n in NOISE_ORDER],
                           rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)

    plt.tight_layout()
    savefig(fig, "fig5_robustness.eps")


# ═════════════════════════════════════════════════════════════
# FIG 6 — Timeline del Action Masking
# ═════════════════════════════════════════════════════════════
def fig6_action_mask():
    df = load("action_mask_timeline.csv")

    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    fig.suptitle("Action Masking Safety Shield Timeline (300 steps)", fontsize=13, y=1.01)

    steps = df["step"].values

    # Panel A: Active phase vs. requested action
    ax = axes[0]
    ax.step(steps, df["current_phase"], where="post",
            color="#378ADD", linewidth=2.2, label="Active phase")
    ax.scatter(steps, df["requested_action"],
               c=["#E24B4A" if not v else "#1D9E75"
                  for v in df["action_valid"]],
               s=18, alpha=0.6, zorder=3, label="Requested action")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylabel("Phase")
    ax.set_title("A — Active phase and requested actions (red=intercepted, green=valid)")
    ax.legend(handles=[
        Line2D([0], [0], color="#378ADD", linewidth=2, label="Active phase"),
        Line2D([0], [0], marker="o", color="#1D9E75", linestyle="none", label="Valid action"),
        Line2D([0], [0], marker="o", color="#E24B4A", linestyle="none", label="Intercepted"),
    ], fontsize=8)

    # Panel B: Action mask heatmap
    ax = axes[1]
    mask_matrix = df[["mask_ph0", "mask_ph1", "mask_ph2", "mask_ph3"]].values.T
    im = ax.imshow(mask_matrix, aspect="auto", cmap="RdYlGn",
                   extent=[-0.5, len(steps) - 0.5, -0.5, 3.5],
                   vmin=0, vmax=1)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["Phase 0", "Phase 1", "Phase 2", "Phase 3"])
    ax.set_ylabel("Mask")
    ax.set_title("B — Action mask (green=valid, red=blocked)")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Valid")

    # Highlight intergreen (yellow) steps
    yellow_steps = df[df["yellow_active"] == 1]["step"].values
    for s in yellow_steps:
        ax.axvline(s, color="#EF9F27", alpha=0.4, linewidth=0.8)

    # Panel C: Cumulative intercepted violations
    ax = axes[2]
    ax.fill_between(steps, df["violations_intercepted"],
                    color="#E24B4A", alpha=0.25)
    ax.plot(steps, df["violations_intercepted"],
            color="#E24B4A", linewidth=2, label="Cumulative intercepted")
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Cumulative count")
    ax.set_title("C — Illegal actions intercepted by the safety shield (cumulative)")
    ax.legend(fontsize=8)

    # Final annotation
    total = int(df["violations_intercepted"].iloc[-1])
    pct   = total / len(steps) * 100
    ax.text(len(steps) * 0.98, total * 0.95,
            f"{total} intercepted\n({pct:.0f}% of steps)",
            ha="right", fontsize=9, color="#E24B4A",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#E24B4A", alpha=0.85))

    plt.tight_layout()
    savefig(fig, "fig6_action_mask.eps")


# ═════════════════════════════════════════════════════════════
# FIG 7 — Perfiles de demanda temporal
# ═════════════════════════════════════════════════════════════
def fig7_demand_profiles():
    df = load("demand_profiles.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Temporal Demand Profiles", fontsize=13, y=1.01)

    palette = {"off_peak": "#378ADD", "moderate": "#1D9E75",
               "peak": "#EF9F27", "supersaturated": "#E24B4A"}

    # Panel A: mean density over time
    ax = axes[0]
    for profile in DEMANDS:
        sub = df[df["profile"] == profile].sort_values("t_step")
        ax.fill_between(sub["t_minutes"], sub["mean_density"] - sub["std_density"],
                        sub["mean_density"] + sub["std_density"],
                        alpha=0.12, color=palette[profile])
        ax.plot(sub["t_minutes"], sub["mean_density"],
                color=palette[profile], linewidth=2.2,
                label=DEMAND_LABELS[profile])

    ax.set_xlabel("Time (episode minutes)")
    ax.set_ylabel("Mean density (normalised)")
    ax.set_title("Temporal density by profile (band = +/-1 std)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    # Peak markers
    ax.axvline(60 * 0.38, color="#666", linewidth=1, linestyle=":", alpha=0.6)
    ax.text(60 * 0.38 + 0.5, 0.96, "AM Peak", fontsize=8, color="#666")
    ax.axvline(60 * 0.80, color="#666", linewidth=1, linestyle=":", alpha=0.6)
    ax.text(60 * 0.80 + 0.5, 0.96, "PM Peak", fontsize=8, color="#666")

    # Panel B: density boxplot by profile
    ax = axes[1]
    data_by_profile = [df[df["profile"] == p]["mean_density"].values for p in DEMANDS]
    bp = ax.boxplot(data_by_profile, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker=".", markersize=3, alpha=0.5))
    for patch, profile in zip(bp["boxes"], DEMANDS):
        patch.set_facecolor(palette[profile])
        patch.set_alpha(0.6)

    ax.set_xticks(range(1, len(DEMANDS) + 1))
    ax.set_xticklabels([DEMAND_LABELS[p] for p in DEMANDS])
    ax.set_ylabel("Mean density (normalised)")
    ax.set_title("Density distribution by profile")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    savefig(fig, "fig7_demand_profiles.eps")


# ═════════════════════════════════════════════════════════════
# FIG 8 — Análisis de onda verde
# ═════════════════════════════════════════════════════════════
def fig8_green_wave():
    df = load("green_wave_offsets.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Green-Wave Corridor Coordination Analysis", fontsize=13, y=1.01)

    speeds = sorted(df["speed_kmh"].unique())
    tl_ids = df["tl_id"].unique()
    palette_spd = plt.cm.viridis(np.linspace(0.15, 0.9, len(speeds)))

    # Panel A: Ideal offset by speed and TL
    ax = axes[0]
    for speed, color in zip(speeds, palette_spd):
        sub = df[df["speed_kmh"] == speed].sort_values("distance_m")
        ax.plot(sub["distance_m"], sub["offset_ideal_s"], "o-",
                color=color, linewidth=2, markersize=6,
                label=f"{speed} km/h")
    ax.set_xlabel("Cumulative distance (m)")
    ax.set_ylabel("Ideal offset (s)")
    ax.set_title("Green-wave offsets by target speed")
    ax.legend(title="Speed", fontsize=8)

    # Panel B: ATT with/without green wave (by speed)
    ax = axes[1]
    speed_att_no  = []
    speed_att_yes = []
    for speed in speeds:
        sub = df[df["speed_kmh"] == speed]
        speed_att_no.append(sub["att_no_wave_s"].mean())
        speed_att_yes.append(sub["att_green_wave_s"].mean())

    x = np.arange(len(speeds))
    w = 0.35
    ax.bar(x - w/2, speed_att_no,  w, label="Without green wave", color="#E24B4A", alpha=0.8)
    ax.bar(x + w/2, speed_att_yes, w, label="With green wave",    color="#1D9E75", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} km/h" for s in speeds], fontsize=8)
    ax.set_ylabel("ATT (s)")
    ax.set_title("Avg. travel time: with vs without green wave")
    ax.legend()

    # Panel C: Relative improvement per TL
    ax = axes[2]
    tl_labels = list(tl_ids)
    for speed, color in zip(speeds, palette_spd):
        sub = df[df["speed_kmh"] == speed].sort_values("distance_m")
        ax.plot(range(len(tl_labels)), sub["improvement_pct"].values,
                "o-", color=color, linewidth=1.8, markersize=5,
                label=f"{speed} km/h")
    ax.set_xticks(range(len(tl_labels)))
    ax.set_xticklabels(tl_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("ATT improvement (%)")
    ax.set_title("ATT Improvement per Intersection")
    ax.axhline(0, color="#999", linewidth=0.8, linestyle="--")
    ax.legend(title="Speed", fontsize=8)

    plt.tight_layout()
    savefig(fig, "fig8_green_wave.eps")


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Gráficas ATSC")
    p.add_argument("--out_dir", default="results/figures")
    p.add_argument("--dpi",     type=int, default=150)
    p.add_argument("--fig",     default="all",
                   help="Figura a generar: 1-8 o 'all'")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    OUT_DIR = args.out_dir
    DPI     = args.dpi

    os.makedirs(OUT_DIR, exist_ok=True)

    figs = {
        "1": fig1_training_curves,
        "2": fig2_kpi_comparison,
        "3": fig3_convergence,
        "4": fig4_stress_heatmap,
        "5": fig5_robustness,
        "6": fig6_action_mask,
        "7": fig7_demand_profiles,
        "8": fig8_green_wave,
    }

    print(f"\n{'='*55}")
    print(f"  Generando graficas -> {OUT_DIR}/")
    print(f"{'='*55}")

    targets = figs.keys() if args.fig == "all" else [args.fig]
    for key in targets:
        if key in figs:
            figs[key]()
        else:
            print(f"  [WARN] Fig {key} no existe. Opciones: 1-8 o 'all'")

    print(f"\n  Todas las figuras guardadas en: {OUT_DIR}/")
    print(f"{'='*55}\n")
