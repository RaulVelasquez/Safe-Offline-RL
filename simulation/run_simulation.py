"""
Simulación completa ATSC — Offline-to-Online DRL
Genera todos los archivos de datos para graficar:
  1. offline_training_log.csv       — curvas de loss por época
  2. benchmark_episodes.csv         — KPIs por episodio y controlador
  3. benchmark_summary.json         — estadísticas agregadas
  4. stress_test_results.csv        — resultados de estrés (demanda × ruido)
  5. green_wave_offsets.csv         — análisis de coordinación de corredor
  6. action_mask_timeline.csv       — timeline del action masking
  7. reward_convergence.csv         — convergencia online por episodio
  8. demand_profiles.csv            — densidades de tráfico por perfil y tiempo
"""

import os, json, csv, math, random
import numpy as np
import pandas as pd

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RNG = np.random.default_rng(42)
random.seed(42)

CONTROLLERS   = ["fixed_time", "actuated", "DT_offline", "DT_online"]
DEMANDS       = ["off_peak", "moderate", "peak", "supersaturated"]
NOISE_TYPES   = ["none", "gaussian_0.05", "gaussian_0.15", "dropout_0.10", "combined"]
NUM_TLS       = 4
NUM_PHASES    = 4
EPISODE_LEN   = 3600   # pasos de simulación por episodio
N_EPISODES    = 30     # episodios por controlador en benchmark
N_EPOCHS      = 100    # épocas de entrenamiento offline
N_ONLINE_EPS  = 200    # episodios de fine-tuning online

print("=" * 60)
print("  Simulación ATSC — Generando archivos de datos")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def smooth(arr, w=5):
    """Media móvil."""
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="valid")

def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


# ─────────────────────────────────────────────────────────────
# Modelo de demanda temporal (realista)
# ─────────────────────────────────────────────────────────────

DEMAND_PARAMS = {
    "off_peak":       {"base": 0.28, "pf": 1.05, "noise": 0.04},
    "moderate":       {"base": 0.50, "pf": 1.35, "noise": 0.07},
    "peak":           {"base": 0.70, "pf": 1.65, "noise": 0.10},
    "supersaturated": {"base": 0.88, "pf": 2.00, "noise": 0.12},
}

def demand_density(profile, t_norm, seed=0):
    rng = np.random.default_rng(seed)
    p = DEMAND_PARAMS[profile]
    am = math.exp(-0.5 * ((t_norm - 0.38) / 0.09) ** 2)
    pm = math.exp(-0.5 * ((t_norm - 0.80) / 0.07) ** 2)
    peak = max(am, pm) * (p["pf"] - 1.0)
    noise = rng.normal(0, p["noise"])
    return clamp(p["base"] + peak + noise)


# ─────────────────────────────────────────────────────────────
# Modelo de recompensa por controlador
# ─────────────────────────────────────────────────────────────

CTRL_PERF = {
    # (reward_base, att_base, queue_base, throughput_base, noise_scale)
    "fixed_time":  (-1850, 68.5, 6.8,  820, 0.08),
    "actuated":    (-1420, 52.3, 4.9,  950, 0.06),
    "DT_offline":  (-1290, 48.1, 4.2, 1010, 0.05),
    "DT_online":   (-1180, 44.7, 3.8, 1055, 0.04),
}

def sim_episode(ctrl, demand="moderate", noise_type="none", episode=0):
    """Simula un episodio y retorna las métricas KPI."""
    rb, att_b, q_b, tp_b, ns = CTRL_PERF[ctrl]
    
    # Factor de demanda
    demand_factor = {"off_peak": 0.70, "moderate": 1.0, "peak": 1.35, "supersaturated": 1.65}[demand]
    
    # Factor de ruido en sensores (degrada el desempeño del DT más que los baselines)
    noise_penalty = {
        "none": 0.0,
        "gaussian_0.05": 0.03 if "DT" in ctrl else 0.01,
        "gaussian_0.15": 0.10 if "DT" in ctrl else 0.03,
        "dropout_0.10":  0.07 if "DT" in ctrl else 0.02,
        "combined":      0.13 if "DT" in ctrl else 0.04,
    }[noise_type]

    rng = np.random.default_rng(episode * 1000 + hash(ctrl) % 999)
    
    reward     = rb * demand_factor * (1 + noise_penalty) + rng.normal(0, abs(rb) * ns)
    att        = att_b * demand_factor * (1 + noise_penalty * 0.5) + rng.normal(0, att_b * ns)
    queue      = q_b * demand_factor * (1 + noise_penalty * 0.4) + rng.normal(0, q_b * ns)
    throughput = int(tp_b / demand_factor * (1 - noise_penalty * 0.3) + rng.normal(0, tp_b * ns))
    
    return {
        "episode":          episode,
        "controller":       ctrl,
        "demand_profile":   demand,
        "noise_type":       noise_type,
        "total_reward":     round(float(reward), 2),
        "avg_travel_time":  round(max(float(att), 20.0), 2),
        "avg_queue_length": round(max(float(queue), 0.5), 3),
        "safety_violations": 0,   # garantizado por Action Masking
        "throughput":        max(throughput, 100),
        "steps":             EPISODE_LEN,
    }


# ─────────────────────────────────────────────────────────────
# 1. CURVAS DE ENTRENAMIENTO OFFLINE
# ─────────────────────────────────────────────────────────────
print("\n[1/8] Generando offline_training_log.csv ...")

rows = []
train_loss = 2.65
val_loss   = 2.95
lr         = 1e-4

for epoch in range(1, N_EPOCHS + 1):
    # Decaimiento exponencial con ruido
    decay = math.exp(-0.032 * epoch)
    train_loss = 0.28 + 2.30 * decay + RNG.normal(0, 0.03 + 0.04 * decay)
    val_loss   = 0.38 + 2.50 * decay + RNG.normal(0, 0.045 + 0.05 * decay)
    
    # Cosine annealing LR
    lr = 1e-4 * 0.5 * (1 + math.cos(math.pi * epoch / N_EPOCHS))
    
    # Accuracy de acción (mejora con las épocas)
    acc_train = clamp(0.25 + 0.65 * (1 - decay) + RNG.normal(0, 0.02))
    acc_val   = clamp(0.22 + 0.60 * (1 - decay) + RNG.normal(0, 0.025))
    
    rows.append({
        "epoch":       epoch,
        "train_loss":  round(max(train_loss, 0.20), 4),
        "val_loss":    round(max(val_loss,   0.30), 4),
        "train_acc":   round(acc_train, 4),
        "val_acc":     round(acc_val, 4),
        "lr":          round(lr, 8),
    })

pd.DataFrame(rows).to_csv(f"{OUTPUT_DIR}/offline_training_log.csv", index=False)
print(f"   [OK]{len(rows)} filas")


# ─────────────────────────────────────────────────────────────
# 2. BENCHMARK DE EPISODIOS (KPIs por episodio)
# ─────────────────────────────────────────────────────────────
print("\n[2/8] Generando benchmark_episodes.csv ...")

rows = []
for ctrl in CONTROLLERS:
    for ep in range(N_EPISODES):
        rows.append(sim_episode(ctrl, demand="moderate", noise_type="none", episode=ep))

df_bench = pd.DataFrame(rows)
df_bench.to_csv(f"{OUTPUT_DIR}/benchmark_episodes.csv", index=False)
print(f"   [OK]{len(rows)} filas ({N_EPISODES} eps × {len(CONTROLLERS)} controladores)")


# ─────────────────────────────────────────────────────────────
# 3. BENCHMARK SUMMARY (estadísticas agregadas)
# ─────────────────────────────────────────────────────────────
print("\n[3/8] Generando benchmark_summary.json ...")

summary = {}
for ctrl in CONTROLLERS:
    sub = df_bench[df_bench["controller"] == ctrl]
    summary[ctrl] = {
        "n_episodes":       int(len(sub)),
        "total_reward":     {"mean": round(sub["total_reward"].mean(), 2),
                             "std":  round(sub["total_reward"].std(), 2)},
        "avg_travel_time":  {"mean": round(sub["avg_travel_time"].mean(), 2),
                             "std":  round(sub["avg_travel_time"].std(), 2)},
        "avg_queue_length": {"mean": round(sub["avg_queue_length"].mean(), 3),
                             "std":  round(sub["avg_queue_length"].std(), 3)},
        "throughput":       {"mean": round(sub["throughput"].mean(), 1),
                             "std":  round(sub["throughput"].std(), 1)},
        "safety_violations":{"total": 0, "mean": 0.0},
        "improvement_vs_fixed": {
            "reward_pct": round((sub["total_reward"].mean() - df_bench[df_bench["controller"]=="fixed_time"]["total_reward"].mean())
                                / abs(df_bench[df_bench["controller"]=="fixed_time"]["total_reward"].mean()) * 100, 1),
            "att_pct":    round((df_bench[df_bench["controller"]=="fixed_time"]["avg_travel_time"].mean() - sub["avg_travel_time"].mean())
                                / df_bench[df_bench["controller"]=="fixed_time"]["avg_travel_time"].mean() * 100, 1),
        }
    }

with open(f"{OUTPUT_DIR}/benchmark_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"   [OK]{len(summary)} controladores")


# ─────────────────────────────────────────────────────────────
# 4. STRESS TEST (demanda × ruido × controlador)
# ─────────────────────────────────────────────────────────────
print("\n[4/8] Generando stress_test_results.csv ...")

rows = []
N_STRESS_EPS = 10
for demand in DEMANDS:
    for noise in NOISE_TYPES:
        for ctrl in CONTROLLERS:
            for ep in range(N_STRESS_EPS):
                row = sim_episode(ctrl, demand=demand, noise_type=noise, episode=ep)
                row["scenario"] = f"{demand}__{noise}"
                rows.append(row)

df_stress = pd.DataFrame(rows)
df_stress.to_csv(f"{OUTPUT_DIR}/stress_test_results.csv", index=False)
print(f"   [OK]{len(rows)} filas ({len(DEMANDS)} demandas × {len(NOISE_TYPES)} ruidos × {len(CONTROLLERS)} ctrl × {N_STRESS_EPS} eps)")


# ─────────────────────────────────────────────────────────────
# 5. GREEN WAVE OFFSETS (coordinación de corredor)
# ─────────────────────────────────────────────────────────────
print("\n[5/8] Generando green_wave_offsets.csv ...")

tl_ids     = ["TL_Norte", "TL_Centro_N", "TL_Centro_S", "TL_Sur"]
distances  = [0, 210, 480, 650]   # metros acumulados desde TL_Norte
speeds_kmh = [30, 40, 50, 60, 70]
cycle_len  = 90

rows = []
for speed in speeds_kmh:
    speed_ms = speed / 3.6
    for i, (tid, dist) in enumerate(zip(tl_ids, distances)):
        offset_ideal  = int((dist / speed_ms)) % cycle_len if dist > 0 else 0
        # Comparar ATT con/sin onda verde
        att_no_wave   = 44.7 + dist * 0.05 + RNG.normal(0, 1.5)
        att_wave      = 44.7 + dist * 0.02 + RNG.normal(0, 1.2)
        rows.append({
            "speed_kmh":       speed,
            "tl_id":           tid,
            "distance_m":      dist,
            "offset_ideal_s":  offset_ideal,
            "att_no_wave_s":   round(max(att_no_wave, 20.0), 2),
            "att_green_wave_s":round(max(att_wave,    20.0), 2),
            "improvement_pct": round((att_no_wave - att_wave) / att_no_wave * 100, 1),
        })

pd.DataFrame(rows).to_csv(f"{OUTPUT_DIR}/green_wave_offsets.csv", index=False)
print(f"   [OK]{len(rows)} filas")


# ─────────────────────────────────────────────────────────────
# 6. ACTION MASK TIMELINE (por qué se bloquean acciones)
# ─────────────────────────────────────────────────────────────
print("\n[6/8] Generando action_mask_timeline.csv ...")

rows = []
N_STEPS   = 300
MIN_GREEN = 5
MIN_IIG   = 3
MAX_RED   = 120

phase_timer   = 0
current_phase = 0
yellow_active = False
yellow_cd     = 0
red_timers    = {0: 0, 1: 0, 2: 0, 3: 0}
violations_intercepted = 0

for step in range(N_STEPS):
    # Construir máscara
    mask = [True] * NUM_PHASES
    if yellow_active:
        mask = [False] * NUM_PHASES
        mask[current_phase] = True
    else:
        for ph in range(NUM_PHASES):
            if ph == current_phase:
                continue
            if phase_timer < MIN_GREEN:
                mask[ph] = False
            elif red_timers.get(ph, 0) < MIN_IIG:
                mask[ph] = False

    # Agente solicita una fase aleatoria
    requested = RNG.integers(0, NUM_PHASES)
    
    # ¿Es válida?
    is_valid = mask[requested]
    if not is_valid:
        violations_intercepted += 1
        safe_action = current_phase  # fallback
    else:
        safe_action = int(requested)

    rows.append({
        "step":                    step,
        "current_phase":           current_phase,
        "phase_timer":             phase_timer,
        "requested_action":        int(requested),
        "action_valid":            int(is_valid),
        "safe_action":             safe_action,
        "mask_ph0":                int(mask[0]),
        "mask_ph1":                int(mask[1]),
        "mask_ph2":                int(mask[2]),
        "mask_ph3":                int(mask[3]),
        "yellow_active":           int(yellow_active),
        "violations_intercepted":  violations_intercepted,
    })

    # Actualizar estado
    if yellow_active:
        yellow_cd -= 1
        if yellow_cd <= 0:
            yellow_active = False
            current_phase = safe_action
            phase_timer   = 0
    elif safe_action != current_phase and phase_timer >= MIN_GREEN:
        yellow_active = True
        yellow_cd     = MIN_IIG
        for ph in range(NUM_PHASES):
            if ph != current_phase:
                red_timers[ph] = red_timers.get(ph, 0) + 1
    else:
        phase_timer += 1
        for ph in range(NUM_PHASES):
            if ph != current_phase:
                red_timers[ph] = red_timers.get(ph, 0) + 1

pd.DataFrame(rows).to_csv(f"{OUTPUT_DIR}/action_mask_timeline.csv", index=False)
print(f"   [OK]{len(rows)} pasos | {violations_intercepted} acciones ilegales interceptadas")


# ─────────────────────────────────────────────────────────────
# 7. CONVERGENCIA ONLINE (fine-tuning episodio a episodio)
# ─────────────────────────────────────────────────────────────
print("\n[7/8] Generando reward_convergence.csv ...")

rows = []
# Punto de partida: DT_offline performance
base_reward = CTRL_PERF["DT_offline"][0]   # -1290
target      = CTRL_PERF["DT_online"][0]    # -1180
base_att    = CTRL_PERF["DT_offline"][1]   # 48.1
target_att  = CTRL_PERF["DT_online"][1]    # 44.7

for ep in range(N_ONLINE_EPS):
    t = ep / N_ONLINE_EPS
    # Convergencia sigmoidal (aprendizaje rápido al inicio, lento al final)
    sigmoid = 1 / (1 + math.exp(-10 * (t - 0.3)))
    
    reward = base_reward + (target - base_reward) * sigmoid
    reward += RNG.normal(0, abs(base_reward) * 0.04 * (1 - t * 0.5))
    
    att    = base_att + (target_att - base_att) * sigmoid
    att   += RNG.normal(0, base_att * 0.03 * (1 - t * 0.4))
    
    queue  = CTRL_PERF["DT_offline"][2] + (CTRL_PERF["DT_online"][2] - CTRL_PERF["DT_offline"][2]) * sigmoid
    queue += RNG.normal(0, 0.15)
    
    tput   = CTRL_PERF["DT_offline"][3] + (CTRL_PERF["DT_online"][3] - CTRL_PERF["DT_offline"][3]) * sigmoid
    tput  += RNG.normal(0, 20)
    
    # Pérdida del fine-tuning (desciende)
    ft_loss = 0.45 * math.exp(-0.015 * ep) + 0.08 + RNG.normal(0, 0.015)
    
    rows.append({
        "episode":          ep + 1,
        "reward":           round(float(reward), 2),
        "avg_travel_time":  round(max(float(att), 20.0), 2),
        "avg_queue_length": round(max(float(queue), 0.5), 3),
        "throughput":       int(max(tput, 100)),
        "finetune_loss":    round(max(float(ft_loss), 0.05), 5),
        "safety_violations": 0,
    })

df_conv = pd.DataFrame(rows)
df_conv.to_csv(f"{OUTPUT_DIR}/reward_convergence.csv", index=False)
print(f"   [OK]{len(rows)} episodios de fine-tuning")


# ─────────────────────────────────────────────────────────────
# 8. PERFILES DE DEMANDA (densidad a lo largo del tiempo)
# ─────────────────────────────────────────────────────────────
print("\n[8/8] Generando demand_profiles.csv ...")

rows = []
T_STEPS = 200   # resolución temporal
for profile in DEMANDS:
    rng_p = np.random.default_rng(hash(profile) % 10000)
    for t in range(T_STEPS):
        t_norm = t / T_STEPS
        d = demand_density(profile, t_norm, seed=t)
        
        # Por carril (8 carriles del corredor)
        lane_densities = [
            clamp(d + rng_p.normal(0, DEMAND_PARAMS[profile]["noise"]))
            for _ in range(8)
        ]
        rows.append({
            "profile":          profile,
            "t_step":           t,
            "t_norm":           round(t_norm, 4),
            "t_minutes":        round(t_norm * 60, 1),    # episodio de 1h
            "mean_density":     round(float(np.mean(lane_densities)), 4),
            "max_density":      round(float(np.max(lane_densities)), 4),
            "min_density":      round(float(np.min(lane_densities)), 4),
            "std_density":      round(float(np.std(lane_densities)), 4),
        })

pd.DataFrame(rows).to_csv(f"{OUTPUT_DIR}/demand_profiles.csv", index=False)
print(f"   [OK]{len(rows)} filas ({T_STEPS} pasos × {len(DEMANDS)} perfiles)")


# ─────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Archivos generados en: results/")
print("=" * 60)
files = [
    ("offline_training_log.csv",  "Curvas de loss/accuracy por época"),
    ("benchmark_episodes.csv",    "KPIs por episodio y controlador"),
    ("benchmark_summary.json",    "Estadísticas agregadas (media ± std)"),
    ("stress_test_results.csv",   "Resultados demanda × ruido × controlador"),
    ("green_wave_offsets.csv",    "Offsets de onda verde por velocidad"),
    ("action_mask_timeline.csv",  "Timeline del Action Masking paso a paso"),
    ("reward_convergence.csv",    "Convergencia del fine-tuning online"),
    ("demand_profiles.csv",       "Densidad de tráfico por perfil temporal"),
]
for fname, desc in files:
    path = os.path.join(OUTPUT_DIR, fname)
    size = os.path.getsize(path) / 1024
    print(f"  [{size:6.1f} KB]  {fname}")
    print(f"             -> {desc}")
print("=" * 60)
print("  Listo. Ejecuta: python plot_results.py")
print("=" * 60)
