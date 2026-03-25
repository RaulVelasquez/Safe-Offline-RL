"""
plot_pipeline.py
================
Generates a professional pipeline diagram for the ATSC Offline-to-Online
framework. Output: pipeline.png + pipeline.eps (600 DPI).

Usage:
    python plot_pipeline.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

OUT_DIR = "results/figures"
DPI = 600

# ── Colour palette ────────────────────────────────────────────────────────────
C_PHASE1   = "#2C6FAC"   # dark blue
C_PHASE1L  = "#D6E8F7"   # light blue fill
C_PHASE2   = "#C96A12"   # dark orange
C_PHASE2L  = "#FDE8D0"   # light orange fill
C_PHASE3   = "#1A7A4A"   # dark green
C_PHASE3L  = "#D2F0E2"   # light green fill
C_SUMO     = "#555555"
C_SUMO_L   = "#EBEBEB"
C_ARROW    = "#333333"
C_WHITE    = "#FFFFFF"
C_GOLD     = "#E8A000"

# ── Helper functions ──────────────────────────────────────────────────────────

def rbox(ax, x, y, w, h, fc, ec, lw=1.5, radius=0.04, alpha=1.0, zorder=2):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)
    return box

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=2.0, zorder=5):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color, lw=lw,
            mutation_scale=16,
        ),
        zorder=zorder,
    )

def text(ax, x, y, s, fs=9, fw="normal", color="#222222",
         ha="center", va="center", zorder=6, wrap=False):
    ax.text(x, y, s, fontsize=fs, fontweight=fw, color=color,
            ha=ha, va=va, zorder=zorder,
            wrap=wrap, clip_on=False)

def phase_label(ax, x, y, number, title, color):
    """Pill-shaped phase badge."""
    rbox(ax, x - 0.27, y - 0.055, 0.54, 0.115,
         fc=color, ec=color, radius=0.05, zorder=7)
    ax.text(x, y, f"PHASE {number}  |  {title}",
            fontsize=8.5, fontweight="bold", color=C_WHITE,
            ha="center", va="center", zorder=8)

# ── Canvas setup ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor(C_WHITE)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(8, 8.55,
        "Safe-Guarded Offline-to-Online DRL for Adaptive Traffic Signal Control",
        fontsize=13, fontweight="bold", color="#1A1A1A",
        ha="center", va="center")
ax.text(8, 8.22,
        "3-Phase Framework: Offline Pre-training  |  Safety Shield  |  Online Fine-tuning",
        fontsize=9.5, color="#555555", ha="center", va="center")

# =============================================================================
# PHASE I  (left panel)
# =============================================================================
P1X, P1Y, P1W, P1H = 0.35, 1.0, 4.6, 6.8
rbox(ax, P1X, P1Y, P1W, P1H, fc=C_PHASE1L, ec=C_PHASE1, lw=2.2, radius=0.07)
phase_label(ax, P1X + P1W / 2, P1Y + P1H - 0.18, "I", "OFFLINE PRE-TRAINING", C_PHASE1)

# -- SUMO + LemgoRL box
rbox(ax, 0.65, 6.45, 4.0, 0.95, fc=C_SUMO_L, ec=C_SUMO, lw=1.2, radius=0.05)
text(ax, 2.65, 7.05, "SUMO + LemgoRL Network", fs=9, fw="bold", color=C_SUMO)
text(ax, 2.65, 6.78, "Realistic 4-intersection urban corridor", fs=8, color="#555")

# -- Controllers
for i, (label, sub, cx) in enumerate([
    ("Fixed-Time", "90 s cycle", 1.3),
    ("Actuated",   "gap-based",  4.0),
]):
    rbox(ax, cx - 0.55, 5.3, 1.1, 0.75, fc=C_WHITE, ec=C_SUMO, lw=1.0, radius=0.04)
    text(ax, cx, 5.72, label, fs=8.5, fw="bold", color=C_SUMO)
    text(ax, cx, 5.48, sub,   fs=7.5, color="#666")

text(ax, 2.65, 5.05, "500 episodes each  =  1,000 total", fs=7.5, color="#666")

# arrow SUMO -> controllers
arrow(ax, 2.65, 6.45, 2.65, 6.06, lw=1.5)

# -- Dataset box
rbox(ax, 0.75, 3.95, 3.8, 0.75, fc=C_WHITE, ec=C_PHASE1, lw=1.2, radius=0.04)
text(ax, 2.65, 4.42, "Offline Dataset  D = {tau_i} (i=1..1000)", fs=8.5, fw="bold", color=C_PHASE1)
text(ax, 2.65, 4.18, "Trajectories: (RTG, state, action) sequences", fs=7.5, color="#555")

arrow(ax, 2.65, 5.3,  2.65, 4.7,  lw=1.5)

# -- DT Architecture box
rbox(ax, 0.65, 1.35, 4.0, 2.32, fc=C_WHITE, ec=C_PHASE1, lw=1.4, radius=0.05)
text(ax, 2.65, 3.45, "Decision Transformer", fs=10, fw="bold", color=C_PHASE1)
text(ax, 2.65, 3.18, "Sequence model: RTG-conditioned policy", fs=8, color="#444")

# mini arch table
rows = [
    ("obs_dim",  "100  (4 TL x 25 features)"),
    ("act_dim",  "16   (4 TL x 4 phases)"),
    ("d_model",  "128"),
    ("n_layer",  "4  blocks  |  n_head = 4"),
    ("context",  "K = 20 steps"),
    ("params",   "1,271,696"),
]
for k, (key, val) in enumerate(rows):
    yy = 2.95 - k * 0.245
    rbox(ax, 0.85, yy - 0.10, 1.2, 0.21, fc=C_PHASE1L, ec="none", radius=0.03)
    text(ax, 1.45, yy, key,  fs=7.5, fw="bold", color=C_PHASE1, ha="center")
    text(ax, 2.92, yy, val,  fs=7.5, color="#333", ha="center")

text(ax, 2.65, 1.55, "Loss: cross-entropy  |  Optimizer: AdamW  lr=1e-4", fs=7.5, color="#555")

arrow(ax, 2.65, 3.95, 2.65, 3.67, lw=1.5)

# checkpoint label at bottom
rbox(ax, 1.15, 1.08, 3.0, 0.30, fc=C_PHASE1, ec=C_PHASE1, radius=0.04)
text(ax, 2.65, 1.235, "best_model.pt   (checkpoint)", fs=8, fw="bold", color=C_WHITE)

# =============================================================================
# PHASE II  (centre panel)
# =============================================================================
P2X, P2Y, P2W, P2H = 5.65, 1.0, 4.7, 6.8
rbox(ax, P2X, P2Y, P2W, P2H, fc=C_PHASE2L, ec=C_PHASE2, lw=2.2, radius=0.07)
phase_label(ax, P2X + P2W / 2, P2Y + P2H - 0.18, "II", "SAFETY + COORDINATION", C_PHASE2)

# -- Action Masking box
rbox(ax, 5.85, 5.5, 4.3, 1.95, fc=C_WHITE, ec=C_PHASE2, lw=1.4, radius=0.05)
text(ax, 8.0, 7.22, "Action Masking Shield", fs=10, fw="bold", color=C_PHASE2)
text(ax, 8.0, 6.97, "Formal guarantee: violations = 0", fs=8, color="#444")

constraints = [
    ("min_green",      "5 s",   "Minimum green time"),
    ("min_intergreen", "3 s",   "Intergreen / yellow"),
    ("max_red",        "120 s", "Max consecutive red"),
]
for k, (param, val, desc) in enumerate(constraints):
    yy = 6.65 - k * 0.37
    rbox(ax, 6.0,  yy - 0.13, 1.05, 0.27, fc=C_PHASE2L, ec=C_PHASE2, lw=0.8, radius=0.03)
    rbox(ax, 7.15, yy - 0.13, 0.5,  0.27, fc=C_PHASE2,  ec=C_PHASE2, lw=0.8, radius=0.03)
    text(ax, 6.525, yy, param, fs=7.5, fw="bold", color=C_PHASE2)
    text(ax, 7.4,   yy, val,   fs=7.5, fw="bold", color=C_WHITE)
    text(ax, 9.05,  yy, desc,  fs=7.5, color="#555")

# -- Mask formula
rbox(ax, 5.95, 5.58, 4.1, 0.52, fc=C_PHASE2L, ec=C_PHASE2, lw=0.8, radius=0.03)
text(ax, 8.0, 5.875,
     "mask[p] = False  =>  logit[p] = -inf  =>  argmax safe",
     fs=7.5, color=C_PHASE2, fw="bold")
text(ax, 8.0, 5.655,
     "Applied to DT logits before argmax at every step",
     fs=7.2, color="#555")

# -- Green Wave box
rbox(ax, 5.85, 3.0, 4.3, 2.22, fc=C_WHITE, ec=C_PHASE2, lw=1.4, radius=0.05)
text(ax, 8.0, 4.98, "Green-Wave Corridor Coordinator", fs=10, fw="bold", color=C_PHASE2)
text(ax, 8.0, 4.73, "Message-passing between adjacent TLs", fs=8, color="#444")

offsets = [("TL0", 0, 0), ("TL1", 200, 14), ("TL2", 400, 28), ("TL3", 600, 43)]
for k, (tl, dist, off) in enumerate(offsets):
    cx = 6.25 + k * 1.0
    rbox(ax, cx - 0.38, 3.75, 0.76, 0.75,
         fc=C_PHASE2L, ec=C_PHASE2, lw=0.9, radius=0.04)
    text(ax, cx, 4.28, tl,            fs=8,   fw="bold", color=C_PHASE2)
    text(ax, cx, 4.05, f"{dist} m",   fs=7.2, color="#555")
    text(ax, cx, 3.82, f"+{off} s",   fs=8,   fw="bold", color=C_PHASE2)

text(ax, 8.0, 3.55, "Offset = floor(i * d / v)   at v = 50 km/h", fs=8, color="#555")
text(ax, 8.0, 3.28, "Reduces inter-intersection stops by ~30%",    fs=8, color=C_PHASE2, fw="bold")

# arrows inside phase II
arrow(ax, 8.0, 5.5,  8.0, 5.22, lw=1.5)

# checkpoint label
rbox(ax, 6.15, 1.08, 3.7, 0.30, fc=C_PHASE2, ec=C_PHASE2, radius=0.04)
text(ax, 8.0, 1.235, "model + shield  (ready to deploy)", fs=8, fw="bold", color=C_WHITE)

# =============================================================================
# PHASE III  (right panel)
# =============================================================================
P3X, P3Y, P3W, P3H = 11.05, 1.0, 4.6, 6.8
rbox(ax, P3X, P3Y, P3W, P3H, fc=C_PHASE3L, ec=C_PHASE3, lw=2.2, radius=0.07)
phase_label(ax, P3X + P3W / 2, P3Y + P3H - 0.18, "III", "ONLINE FINE-TUNING", C_PHASE3)

# -- Fine-tuner box
rbox(ax, 11.25, 5.15, 4.2, 2.45, fc=C_WHITE, ec=C_PHASE3, lw=1.4, radius=0.05)
text(ax, 13.35, 7.38, "Online Fine-Tuner", fs=10, fw="bold", color=C_PHASE3)
text(ax, 13.35, 7.13, "Adapts DT to live traffic conditions", fs=8, color="#444")

ft_rows = [
    ("Episodes",  "200  (online rollout)"),
    ("Frozen",    "Embeddings + bottom 2 blocks"),
    ("Trainable", "Top 2 blocks + action heads"),
    ("Replay",    "FIFO buffer  C = 10,000"),
    ("Update",    "Every 10 steps  |  batch = 32"),
    ("lr",        "5e-5  |  grad clip = 0.5"),
]
for k, (key, val) in enumerate(ft_rows):
    yy = 6.82 - k * 0.295
    rbox(ax, 11.4,  yy - 0.11, 1.05, 0.23, fc=C_PHASE3L, ec="none", radius=0.03)
    text(ax, 11.925, yy, key, fs=7.5, fw="bold", color=C_PHASE3, ha="center")
    text(ax, 13.5,   yy, val, fs=7.5, color="#333", ha="center")

# -- Benchmark box
rbox(ax, 11.25, 2.6, 4.2, 2.25, fc=C_WHITE, ec=C_PHASE3, lw=1.4, radius=0.05)
text(ax, 13.35, 4.60, "Benchmarking", fs=10, fw="bold", color=C_PHASE3)
text(ax, 13.35, 4.35, "30 evaluation episodes per controller", fs=8, color="#444")

bm = [
    ("Fixed-Time", "+0.0%",   "#E24B4A"),
    ("Actuated",   "+24.1%",  "#EF9F27"),
    ("DT Offline", "+31.2%",  "#378ADD"),
    ("DT Online",  "+37.3%",  "#1D9E75"),
]
for k, (ctrl, gain, col) in enumerate(bm):
    yy = 4.05 - k * 0.355
    bar_w = 0.03 + (k * 0.62)
    rbox(ax, 11.4, yy - 0.11, bar_w + 0.5, 0.24, fc=col, ec="none",
         radius=0.03, alpha=0.85)
    text(ax, 11.4 + bar_w * 0.5 + 0.25, yy,
         f"{ctrl}  {gain} reward", fs=7.8, fw="bold", color=C_WHITE, ha="center")

text(ax, 13.35, 2.74,
     "ATT: -34.8%  |  Queue: -43.4%  |  Violations: 0",
     fs=8, fw="bold", color=C_PHASE3)

arrow(ax, 13.35, 5.15, 13.35, 4.85, lw=1.5)

# checkpoint label
rbox(ax, 11.55, 1.08, 3.6, 0.30, fc=C_PHASE3, ec=C_PHASE3, radius=0.04)
text(ax, 13.35, 1.235, "ep_XXXX.pt   (resume anytime)", fs=8, fw="bold", color=C_WHITE)

# =============================================================================
# INTER-PHASE ARROWS
# =============================================================================
# Phase I -> Phase II
arrow(ax, 4.95, 1.235, 6.15, 1.235, color=C_PHASE1, lw=2.5)
text(ax, 5.55, 1.48, "best_model.pt", fs=7.5, color=C_PHASE1, fw="bold")

# Phase II -> Phase III
arrow(ax, 10.35, 1.235, 11.05, 1.235, color=C_PHASE2, lw=2.5)
text(ax, 10.7, 1.48, "model\n+ shield", fs=7, color=C_PHASE2, fw="bold")

# =============================================================================
# RESULT BADGE  (bottom centre)
# =============================================================================
rbox(ax, 3.8, 0.05, 8.4, 0.72, fc="#1A1A1A", ec="#1A1A1A", radius=0.05)
text(ax, 8.0, 0.53,
     "DT Online:  +37.3% reward   |   -34.8% avg travel time   |   +29.5% throughput",
     fs=9.5, fw="bold", color=C_GOLD)
text(ax, 8.0, 0.24,
     "Safety violations = 0  across all episodes  (formal guarantee via Action Masking)",
     fs=8.5, color="#CCCCCC")

# =============================================================================
# Save
# =============================================================================
plt.tight_layout(pad=0)
os.makedirs(OUT_DIR, exist_ok=True)

png_path = os.path.join(OUT_DIR, "pipeline.png")
fig.savefig(png_path, dpi=DPI, bbox_inches="tight", facecolor=C_WHITE)
print(f"[OK] {png_path}")

eps_path = os.path.join(OUT_DIR, "pipeline.eps")
fig.savefig(eps_path, format="eps", dpi=DPI, bbox_inches="tight", facecolor=C_WHITE)
print(f"[OK] {eps_path}")

plt.show()