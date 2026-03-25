# phase3/stress_test.py
"""
Fase III – Pruebas de Estrés
Evalúa la robustez del sistema bajo:
  1. Distintos niveles de saturación (peak / off-peak / supersaturado)
  2. Ruido en sensores (gaussiano y dropout de carril)
  3. Demanda dinámica (patrón de onda de demanda a lo largo del día)

Genera un reporte CSV y figuras comparativas.
"""

from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils.common import get_logger, set_global_seed

logger = get_logger("stress_test", log_dir="results/logs")


# ──────────────────────────────────────────────────────────────────────────────
# Modelos de demanda y ruido
# ──────────────────────────────────────────────────────────────────────────────

class DemandProfile:
    """
    Genera perfiles de demanda vehicular sintéticos para pruebas de estrés.
    Modela variación temporal realista sin requerir SUMO.
    """

    PROFILES = {
        "off_peak":      {"base": 0.3, "peak_factor": 1.0, "noise_std": 0.05},
        "moderate":      {"base": 0.5, "peak_factor": 1.3, "noise_std": 0.08},
        "peak":          {"base": 0.7, "peak_factor": 1.6, "noise_std": 0.10},
        "supersaturated":{"base": 0.9, "peak_factor": 2.0, "noise_std": 0.12},
    }

    def __init__(self, profile_name: str, num_lanes: int = 8, seed: int = 42) -> None:
        assert profile_name in self.PROFILES, \
            f"Perfil desconocido. Opciones: {list(self.PROFILES.keys())}"
        self.name = profile_name
        self.params = self.PROFILES[profile_name]
        self.num_lanes = num_lanes
        self.rng = np.random.default_rng(seed)

    def sample(self, step: int, episode_length: int = 3600) -> np.ndarray:
        """
        Retorna densidades de tráfico (num_lanes,) para el paso `step`.
        Modela la hora punta como una gaussiana centrada en el 40% del episodio.
        """
        t_norm = step / max(episode_length, 1)
        # Factor de hora punta: gaussiana centrada en t=0.4 (mañana) y t=0.8 (tarde)
        peak_am = np.exp(-0.5 * ((t_norm - 0.40) / 0.10) ** 2)
        peak_pm = np.exp(-0.5 * ((t_norm - 0.80) / 0.08) ** 2)
        peak    = max(peak_am, peak_pm) * (self.params["peak_factor"] - 1)

        base_density = self.params["base"] + peak
        noise = self.rng.normal(0, self.params["noise_std"], self.num_lanes)
        densities = np.clip(base_density + noise, 0.0, 1.0)
        return densities.astype(np.float32)


class SensorNoiseModel:
    """
    Aplica ruido realista a las observaciones del entorno.

    Tipos de ruido:
      - gaussian : ruido gaussiano aditivo (detector loop con interferencia)
      - dropout  : algunos sensores fallan (valor=0)
      - combined : gaussian + dropout
    """

    def __init__(
        self,
        noise_type: str = "gaussian",
        gaussian_std: float = 0.05,
        dropout_rate: float = 0.1,
        seed: int = 0,
    ) -> None:
        assert noise_type in ("gaussian", "dropout", "combined", "none")
        self.noise_type = noise_type
        self.gaussian_std = gaussian_std
        self.dropout_rate = dropout_rate
        self.rng = np.random.default_rng(seed)

    def apply(self, obs: np.ndarray) -> np.ndarray:
        if self.noise_type == "none":
            return obs

        noisy = obs.copy()

        if self.noise_type in ("gaussian", "combined"):
            noisy += self.rng.normal(0, self.gaussian_std, obs.shape).astype(np.float32)
            noisy = np.clip(noisy, 0.0, 1.0)

        if self.noise_type in ("dropout", "combined"):
            mask = self.rng.random(obs.shape) < self.dropout_rate
            noisy[mask] = 0.0

        return noisy


# ──────────────────────────────────────────────────────────────────────────────
# Resultados de una prueba de estrés
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StressTestResult:
    scenario:         str
    demand_profile:   str
    noise_type:       str
    controller:       str
    episode:          int
    total_reward:     float
    avg_reward:       float
    safety_violations:int
    obs_variance:     float    # varianza media de la observación (proxy de ruido)
    steps:            int

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Motor de prueba de estrés (sin SUMO — usa entorno sintético)
# ──────────────────────────────────────────────────────────────────────────────

class SyntheticStressEnvironment:
    """
    Entorno sintético para pruebas de estrés que NO requiere SUMO.
    Simula observaciones y recompensas basadas en los modelos de demanda
    y ruido definidos arriba.

    Útil para validar el comportamiento del agente antes de conectar SUMO.
    """

    OBS_DIM = 25 * 4   # 4 intersecciones × 25 features

    def __init__(
        self,
        demand_profile: DemandProfile,
        noise_model: SensorNoiseModel,
        episode_length: int = 3600,
        num_tls: int = 4,
        num_phases: int = 4,
        seed: int = 0,
    ) -> None:
        self.demand = demand_profile
        self.noise  = noise_model
        self.episode_length = episode_length
        self.num_tls = num_tls
        self.num_phases = num_phases
        self.rng = np.random.default_rng(seed)
        self._step = 0
        self._phase_timers = np.zeros(num_tls, dtype=np.int32)
        self._current_phases = np.zeros(num_tls, dtype=np.int32)

    def reset(self) -> Tuple[np.ndarray, dict]:
        self._step = 0
        self._phase_timers[:] = 0
        self._current_phases[:] = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Actualizar fases
        for i, a in enumerate(action):
            if a != self._current_phases[i] and self._phase_timers[i] >= 5:
                self._current_phases[i] = int(a)
                self._phase_timers[i] = 0
            else:
                self._phase_timers[i] += 1

        self._step += 1
        obs = self._get_obs()
        reward = self._compute_reward()
        truncated = self._step >= self.episode_length
        violations = self._count_violations(action)

        return obs, reward, False, truncated, {
            "safety_violations": violations,
            "step": self._step,
        }

    def _get_obs(self) -> np.ndarray:
        obs_parts = []
        for i in range(self.num_tls):
            densities = self.demand.sample(self._step, self.episode_length)
            speeds    = np.clip(1.0 - densities + self.rng.normal(0, 0.05, 8), 0, 1)
            phase_oh  = np.zeros(8, dtype=np.float32)
            ph_idx = int(self._current_phases[i]) if self._current_phases[i] < 8 else 0
            phase_oh[ph_idx] = 1.0
            t_norm = np.array([min(self._phase_timers[i] / 60.0, 1.0)], dtype=np.float32)
            obs_parts.append(np.concatenate([densities, speeds, phase_oh, t_norm]))

        flat_obs = np.concatenate(obs_parts).astype(np.float32)
        return self.noise.apply(flat_obs)

    def _compute_reward(self) -> float:
        """Recompensa negativa proporcional a la densidad de tráfico acumulada."""
        densities = self.demand.sample(self._step, self.episode_length)
        saturation = float(np.mean(densities))
        # Penalizar más en saturación alta
        return -saturation * self.num_tls * 5.0

    def _count_violations(self, action: np.ndarray) -> int:
        """Simula violaciones cuando la acción cambia antes de min_green=5."""
        violations = 0
        for i, a in enumerate(action):
            if a != self._current_phases[i] and self._phase_timers[i] < 5:
                violations += 1
        return violations

    @property
    def _num_tls(self):
        return self.num_tls


# ──────────────────────────────────────────────────────────────────────────────
# Runner de estrés
# ──────────────────────────────────────────────────────────────────────────────

class StressTester:
    """
    Ejecuta la batería completa de pruebas de estrés y genera reportes.
    """

    DEMAND_SCENARIOS = ["off_peak", "moderate", "peak", "supersaturated"]
    NOISE_SCENARIOS  = [
        ("none",     0.00, 0.00),
        ("gaussian", 0.05, 0.00),
        ("gaussian", 0.15, 0.00),
        ("dropout",  0.00, 0.10),
        ("combined", 0.05, 0.10),
    ]

    def __init__(
        self,
        output_dir: str = "results/stress_test",
        episode_length: int = 1800,   # 30 min simulados por escenario
        num_episodes_per_scenario: int = 5,
        seed: int = 42,
    ) -> None:
        self.output_dir = output_dir
        self.episode_length = episode_length
        self.n_episodes = num_episodes_per_scenario
        self.seed = seed
        os.makedirs(output_dir, exist_ok=True)
        self._results: List[StressTestResult] = []

    def run_all(self, controllers: Dict[str, object]) -> None:
        """
        Ejecuta todas las combinaciones de demanda × ruido × controlador.

        Parámetros
        ----------
        controllers : dict nombre → objeto con .select_action(obs) → np.ndarray
        """
        total = len(self.DEMAND_SCENARIOS) * len(self.NOISE_SCENARIOS) * len(controllers)
        done  = 0

        logger.info(f"Iniciando estrés: {total} combinaciones de escenario × controlador")

        for demand_name in self.DEMAND_SCENARIOS:
            for (noise_type, g_std, d_rate) in self.NOISE_SCENARIOS:
                scenario_tag = f"{demand_name}__{noise_type}_g{g_std}_d{d_rate}"

                for ctrl_name, ctrl in controllers.items():
                    done += 1
                    logger.info(
                        f"[{done}/{total}] {scenario_tag} | ctrl={ctrl_name}"
                    )
                    self._run_scenario(
                        demand_name=demand_name,
                        noise_type=noise_type,
                        gaussian_std=g_std,
                        dropout_rate=d_rate,
                        scenario_tag=scenario_tag,
                        ctrl_name=ctrl_name,
                        controller=ctrl,
                    )

        self._save_results()
        self._plot_heatmaps()
        self._plot_robustness_curves()
        logger.info(f"[OK] Pruebas de estres completadas. Resultados en: {self.output_dir}")

    # ── Escenario individual ──────────────────────────────────────────────────

    def _run_scenario(
        self,
        demand_name: str,
        noise_type: str,
        gaussian_std: float,
        dropout_rate: float,
        scenario_tag: str,
        ctrl_name: str,
        controller,
    ) -> None:
        set_global_seed(self.seed)

        demand = DemandProfile(demand_name, num_lanes=8, seed=self.seed)
        noise  = SensorNoiseModel(noise_type, gaussian_std, dropout_rate, self.seed)

        for ep in range(self.n_episodes):
            env = SyntheticStressEnvironment(
                demand_profile=demand,
                noise_model=noise,
                episode_length=self.episode_length,
                seed=self.seed + ep,
            )
            obs, _ = env.reset()
            if hasattr(controller, "reset"):
                controller.reset()

            total_reward  = 0.0
            total_viol    = 0
            obs_variances = []
            steps         = 0

            done = False
            while not done:
                action = controller.select_action(obs)
                obs, reward, _, truncated, info = env.step(action)
                done = truncated
                total_reward  += reward
                total_viol    += info.get("safety_violations", 0)
                obs_variances.append(float(np.var(obs)))
                steps += 1

            self._results.append(StressTestResult(
                scenario=scenario_tag,
                demand_profile=demand_name,
                noise_type=f"{noise_type}_g{gaussian_std}",
                controller=ctrl_name,
                episode=ep,
                total_reward=total_reward,
                avg_reward=total_reward / max(steps, 1),
                safety_violations=total_viol,
                obs_variance=float(np.mean(obs_variances)),
                steps=steps,
            ))

    # ── Guardar resultados ────────────────────────────────────────────────────

    def _save_results(self) -> None:
        import csv

        csv_path = os.path.join(self.output_dir, "stress_test_results.csv")
        if not self._results:
            return

        fieldnames = list(self._results[0].to_dict().keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._results:
                writer.writerow(r.to_dict())

        # Resumen JSON
        summary = self._build_summary()
        json_path = os.path.join(self.output_dir, "stress_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Resultados guardados: {csv_path}")

    def _build_summary(self) -> dict:
        """Agrupa resultados por controlador y escenario."""
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in self._results:
            key = (r.controller, r.demand_profile, r.noise_type)
            grouped[key].append(r)

        summary = {}
        for (ctrl, demand, noise), records in grouped.items():
            rewards = [r.avg_reward for r in records]
            viols   = [r.safety_violations for r in records]
            summary[f"{ctrl}__{demand}__{noise}"] = {
                "avg_reward_mean": float(np.mean(rewards)),
                "avg_reward_std":  float(np.std(rewards)),
                "total_violations": int(sum(viols)),
            }
        return summary

    # ── Visualizaciones ───────────────────────────────────────────────────────

    def _plot_heatmaps(self) -> None:
        """Heatmap de recompensa media por (demanda × ruido) para cada controlador."""
        controllers = list({r.controller for r in self._results})

        fig, axes = plt.subplots(
            1, len(controllers),
            figsize=(6 * len(controllers), 5),
        )
        if len(controllers) == 1:
            axes = [axes]

        demand_levels = self.DEMAND_SCENARIOS
        noise_labels  = [f"{nt}_g{g}" for (nt, g, _) in self.NOISE_SCENARIOS]

        for ax, ctrl in zip(axes, controllers):
            matrix = np.zeros((len(demand_levels), len(noise_labels)))
            for i, demand in enumerate(demand_levels):
                for j, (nt, g, _) in enumerate(self.NOISE_SCENARIOS):
                    noise_tag = f"{nt}_g{g}"
                    vals = [
                        r.avg_reward for r in self._results
                        if r.controller == ctrl
                        and r.demand_profile == demand
                        and r.noise_type == noise_tag
                    ]
                    matrix[i, j] = np.mean(vals) if vals else 0.0

            im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(noise_labels)))
            ax.set_xticklabels(noise_labels, rotation=35, ha="right", fontsize=8)
            ax.set_yticks(range(len(demand_levels)))
            ax.set_yticklabels(demand_levels, fontsize=9)
            ax.set_title(f"Recompensa Media\n{ctrl}", fontsize=10)
            plt.colorbar(im, ax=ax)

        plt.suptitle("Prueba de Estrés — Recompensa por Escenario", fontsize=13)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "heatmap_stress.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Heatmap guardado: {path}")

    def _plot_robustness_curves(self) -> None:
        """Curvas de recompensa vs nivel de ruido gaussiano por controlador."""
        controllers  = list({r.controller for r in self._results})
        noise_levels = sorted({float(r.noise_type.split("_g")[-1]) for r in self._results})

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(controllers)))

        for ctrl, color in zip(controllers, colors):
            means, stds = [], []
            for nlvl in noise_levels:
                vals = [
                    r.avg_reward for r in self._results
                    if r.controller == ctrl and float(r.noise_type.split("_g")[-1]) == nlvl
                ]
                means.append(np.mean(vals) if vals else 0.0)
                stds.append(np.std(vals) if vals else 0.0)

            means = np.array(means)
            stds  = np.array(stds)
            ax.plot(noise_levels, means, "o-", color=color, label=ctrl, linewidth=2)
            ax.fill_between(noise_levels, means - stds, means + stds, alpha=0.15, color=color)

        ax.set_xlabel("Nivel de Ruido Gaussiano (σ)", fontsize=11)
        ax.set_ylabel("Recompensa Promedio por Paso", fontsize=11)
        ax.set_title("Robustez ante Ruido en Sensores", fontsize=12)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.3)

        path = os.path.join(self.output_dir, "robustness_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Curvas de robustez guardadas: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Controladores dummy para pruebas sin modelo entrenado
# ──────────────────────────────────────────────────────────────────────────────

class RandomController:
    def __init__(self, num_tls: int = 4, num_phases: int = 4, seed: int = 0):
        self.num_tls = num_tls
        self.num_phases = num_phases
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        return self.rng.integers(0, self.num_phases, size=self.num_tls).astype(np.int32)

    def reset(self): pass


class GreedyDensityController:
    """
    Controlador greedy: selecciona la fase con mayor densidad de entrada.
    Baseline simple para comparación.
    """
    def __init__(self, num_tls: int = 4, num_phases: int = 4, obs_dim: int = 25):
        self.num_tls = num_tls
        self.num_phases = num_phases
        self.obs_dim = obs_dim

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        actions = []
        for i in range(self.num_tls):
            tl_obs = obs[i * self.obs_dim: (i + 1) * self.obs_dim]
            densities = tl_obs[:8]
            # Dividir en grupos por fase y elegir el de mayor densidad
            group_size = max(len(densities) // self.num_phases, 1)
            phase_densities = [
                np.mean(densities[j * group_size: (j + 1) * group_size])
                for j in range(self.num_phases)
            ]
            actions.append(int(np.argmax(phase_densities)))
        return np.array(actions, dtype=np.int32)

    def reset(self): pass


# ──────────────────────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prueba de Estrés — ATSC")
    p.add_argument("--output_dir",   default="results/stress_test")
    p.add_argument("--episode_len",  type=int,   default=1800)
    p.add_argument("--n_episodes",   type=int,   default=5)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--model_path",   default=None,
                   help="Ruta al checkpoint del DT (opcional)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_global_seed(args.seed)

    controllers = {
        "random":  RandomController(),
        "greedy":  GreedyDensityController(),
    }

    # Si se proporciona un modelo entrenado, añadirlo al benchmark
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"Cargando modelo DT desde {args.model_path}")
        try:
            import torch
            from phase1.models.decision_transformer import DecisionTransformer
            import collections

            ckpt = torch.load(args.model_path, map_location="cpu")

            class DTStressController:
                def __init__(self, model):
                    self.model = model
                    self.model.eval()
                    self.ctx = None
                    self.reset()

                def reset(self):
                    self.ctx = {k: collections.deque(maxlen=20) for k in ["obs","act","rtg","t"]}
                    self._step = 0
                    self._rtg  = -50.0

                def select_action(self, obs):
                    dummy = np.zeros(self.model.num_tls, dtype=np.int64)
                    self.ctx["obs"].append(obs)
                    self.ctx["act"].append(dummy)
                    self.ctx["rtg"].append(self._rtg)
                    self.ctx["t"].append(self._step)
                    self._step += 1

                    with torch.no_grad():
                        t_obs = torch.tensor(np.array(self.ctx["obs"]), dtype=torch.float32).unsqueeze(0)
                        t_act = torch.tensor(np.array(self.ctx["act"]), dtype=torch.long).unsqueeze(0)
                        t_rtg = torch.tensor(np.array(self.ctx["rtg"])[:, None], dtype=torch.float32).unsqueeze(0)
                        t_ts  = torch.tensor(np.array(self.ctx["t"]), dtype=torch.long).unsqueeze(0)
                        act = self.model.get_action(t_obs, t_act, t_rtg, t_ts)
                    return act.numpy()

            # Instanciar modelo desde checkpoint
            cfg = ckpt.get("cfg", {})
            model = DecisionTransformer(
                obs_dim=cfg.get("obs_dim", 25 * 4),
                act_dim=cfg.get("act_dim", 4 * 4),
                num_tls=cfg.get("num_tls", 4),
            )
            model.load_state_dict(ckpt["model_state_dict"])
            controllers["DT"] = DTStressController(model)
        except Exception as e:
            logger.warning(f"No se pudo cargar el modelo DT: {e}")

    tester = StressTester(
        output_dir=args.output_dir,
        episode_length=args.episode_len,
        num_episodes_per_scenario=args.n_episodes,
        seed=args.seed,
    )
    tester.run_all(controllers)
