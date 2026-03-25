# phase3/metrics/metrics_extractor.py
"""
Fase III – Extracción de Métricas
Recopila y compara KPIs del sistema DRL contra baselines tradicionales.
KPIs: ATT (Average Travel Time), Queue Length, Safety Violations, Throughput.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ──────────────────────────────────────────────────────────────────────────────
# Estructura de resultados
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeMetrics:
    """KPIs de un episodio individual."""
    episode:          int
    controller:       str
    total_reward:     float
    avg_travel_time:  float   # segundos
    avg_queue_length: float   # vehículos por carril
    safety_violations:int
    throughput:       int     # vehículos que completaron el viaje
    steps:            int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkResults:
    """Agrega métricas de múltiples episodios por controlador."""
    controller: str
    episodes: List[EpisodeMetrics] = field(default_factory=list)

    def add(self, ep: EpisodeMetrics) -> None:
        self.episodes.append(ep)

    def summary(self) -> Dict[str, float]:
        if not self.episodes:
            return {}
        rewards = [e.total_reward      for e in self.episodes]
        atts    = [e.avg_travel_time   for e in self.episodes]
        queues  = [e.avg_queue_length  for e in self.episodes]
        viols   = [e.safety_violations for e in self.episodes]
        tput    = [e.throughput        for e in self.episodes]

        def stats(arr):
            return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

        return {
            "controller":       self.controller,
            "n_episodes":       len(self.episodes),
            "total_reward":     stats(rewards),
            "avg_travel_time":  stats(atts),
            "avg_queue_length": stats(queues),
            "safety_violations":{"total": int(sum(viols)), "mean": float(np.mean(viols))},
            "throughput":       stats(tput),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Colector de métricas desde TraCI
# ──────────────────────────────────────────────────────────────────────────────

class MetricsCollector:
    """
    Recopila métricas de tráfico en tiempo real desde TraCI durante un episodio.
    Se instancia una vez por episodio.
    """

    def __init__(self) -> None:
        self._travel_times: List[float] = []
        self._queue_lengths: List[float] = []
        self._violations: int = 0
        self._arrived: int = 0
        self._steps: int = 0

    def update(self, info: dict) -> None:
        """Llamar en cada step con el `info` retornado por el entorno."""
        try:
            import traci
            # Tiempo de viaje de vehículos que acaban de llegar
            for veh_id in traci.simulation.getArrivedIDList():
                dep_time = traci.simulation.getDepartedIDList()  # approx
                # SUMO reporta tiempos acumulados
            arrived = traci.simulation.getArrivedNumber()
            self._arrived += arrived

            # Cola media en todos los carriles
            all_lanes = traci.lane.getIDList()
            queues = [traci.lane.getLastStepHaltingNumber(l) for l in all_lanes if not l.startswith(":")]
            if queues:
                self._queue_lengths.append(float(np.mean(queues)))

        except Exception:
            pass

        self._violations += info.get("safety_violations", 0)
        self._steps += 1

    def finalize(
        self,
        controller: str,
        episode: int,
        total_reward: float,
    ) -> EpisodeMetrics:
        att   = float(np.mean(self._travel_times)) if self._travel_times else 0.0
        queue = float(np.mean(self._queue_lengths)) if self._queue_lengths else 0.0
        return EpisodeMetrics(
            episode=episode,
            controller=controller,
            total_reward=total_reward,
            avg_travel_time=att,
            avg_queue_length=queue,
            safety_violations=self._violations,
            throughput=self._arrived,
            steps=self._steps,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Benchmarker
# ──────────────────────────────────────────────────────────────────────────────

class Benchmarker:
    """
    Ejecuta y compara múltiples controladores en el mismo entorno.

    Parámetros
    ----------
    env : SafeEnvironmentWrapper
    output_dir : str
    """

    def __init__(self, env, output_dir: str = "results") -> None:
        self.env = env
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._results: Dict[str, BenchmarkResults] = {}

    def run_controller(
        self,
        controller_name: str,
        controller,            # objeto con .select_action(obs) -> np.ndarray
        num_episodes: int = 30,
        verbose: bool = True,
    ) -> BenchmarkResults:
        """
        Ejecuta `num_episodes` episodios con el controlador dado.
        """
        results = BenchmarkResults(controller=controller_name)

        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            if hasattr(controller, "reset"):
                controller.reset()

            collector = MetricsCollector()
            total_reward = 0.0
            done = False

            while not done:
                action = controller.select_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                collector.update(info)

            ep_metrics = collector.finalize(controller_name, ep, total_reward)
            results.add(ep_metrics)

            if verbose and (ep + 1) % 5 == 0:
                print(
                    f"  [{controller_name}] Ep {ep+1}/{num_episodes} | "
                    f"reward={total_reward:.1f} | "
                    f"violations={ep_metrics.safety_violations}"
                )

        self._results[controller_name] = results
        return results

    def save_results(self) -> None:
        """Guarda resultados en JSON y CSV."""
        summaries = {}
        all_rows = []

        for name, res in self._results.items():
            summaries[name] = res.summary()
            for ep in res.episodes:
                all_rows.append(ep.to_dict())

        # JSON con estadísticas agregadas
        json_path = os.path.join(self.output_dir, "benchmark_summary.json")
        with open(json_path, "w") as f:
            json.dump(summaries, f, indent=2)

        # CSV con todos los episodios
        csv_path = os.path.join(self.output_dir, "benchmark_episodes.csv")
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)

        print(f"[OK] Resultados guardados en {self.output_dir}")
        return summaries

    def plot_comparison(self, save: bool = True) -> None:
        """Genera figura comparativa de los KPIs principales."""
        controllers = list(self._results.keys())
        kpis = {
            "Reward Total":          "total_reward",
            "Tiempo Viaje Promedio (s)": "avg_travel_time",
            "Cola Promedio (vehs)":  "avg_queue_length",
            "Throughput (vehs)":     "throughput",
        }

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Comparativa de Controladores – ATSC", fontsize=14, fontweight="bold")

        colors = plt.cm.Set2(np.linspace(0, 1, len(controllers)))

        for ax, (kpi_label, kpi_key) in zip(axes.flat, kpis.items()):
            means, stds = [], []
            for name in controllers:
                data = [getattr(e, kpi_key) for e in self._results[name].episodes]
                means.append(np.mean(data))
                stds.append(np.std(data))

            bars = ax.bar(controllers, means, yerr=stds, capsize=5, color=colors)
            ax.set_title(kpi_label, fontsize=11)
            ax.set_ylabel(kpi_label)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f"{x:.0f}"
            ))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        if save:
            path = os.path.join(self.output_dir, "kpi_comparison.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[OK] Figura guardada: {path}")
        plt.show()

    def print_summary_table(self) -> None:
        """Imprime tabla de resumen en consola."""
        print("\n" + "=" * 70)
        print(f"{'Controlador':<20} {'Reward':>10} {'ATT(s)':>10} "
              f"{'Cola':>8} {'Viols':>7} {'Throughput':>12}")
        print("-" * 70)

        for name, res in self._results.items():
            s = res.summary()
            print(
                f"{name:<20} "
                f"{s['total_reward']['mean']:>10.1f} "
                f"{s['avg_travel_time']['mean']:>10.1f} "
                f"{s['avg_queue_length']['mean']:>8.2f} "
                f"{s['safety_violations']['total']:>7d} "
                f"{s['throughput']['mean']:>12.0f}"
            )
        print("=" * 70)
