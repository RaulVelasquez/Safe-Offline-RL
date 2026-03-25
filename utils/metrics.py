"""
utils/metrics.py
Recolección y reporte de métricas para benchmarking (Fase III).
Compara el agente DRL contra baselines tradicionales.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EpisodeMetrics:
    """Métricas de un episodio completo."""
    controller_name: str
    episode_id: int
    saturation_level: float = 1.0
    sensor_noise_std: float = 0.0

    # Métricas de rendimiento
    avg_travel_time: float = 0.0        # ATT (segundos)
    avg_waiting_time: float = 0.0       # Tiempo promedio de espera
    avg_queue_length: float = 0.0       # Longitud media de cola
    throughput: float = 0.0             # Vehículos que completaron viaje
    total_reward: float = 0.0

    # Métricas de seguridad
    safety_violations: int = 0          # DEBE ser 0 siempre
    illegal_action_attempts: int = 0    # Intentos bloqueados por masking

    # Metadata
    episode_length: int = 0


@dataclass
class BenchmarkReport:
    """Reporte agregado de múltiples episodios."""
    controller_name: str
    num_episodes: int = 0
    results: List[EpisodeMetrics] = field(default_factory=list)

    @property
    def avg_att(self) -> float:
        return float(np.mean([r.avg_travel_time for r in self.results]))

    @property
    def std_att(self) -> float:
        return float(np.std([r.avg_travel_time for r in self.results]))

    @property
    def avg_waiting(self) -> float:
        return float(np.mean([r.avg_waiting_time for r in self.results]))

    @property
    def avg_queue(self) -> float:
        return float(np.mean([r.avg_queue_length for r in self.results]))

    @property
    def avg_throughput(self) -> float:
        return float(np.mean([r.throughput for r in self.results]))

    @property
    def total_safety_violations(self) -> int:
        return sum(r.safety_violations for r in self.results)

    @property
    def avg_reward(self) -> float:
        return float(np.mean([r.total_reward for r in self.results]))

    def to_dict(self) -> dict:
        return {
            "controller": self.controller_name,
            "episodes": self.num_episodes,
            "avg_travel_time_mean": self.avg_att,
            "avg_travel_time_std": self.std_att,
            "avg_waiting_time": self.avg_waiting,
            "avg_queue_length": self.avg_queue,
            "avg_throughput": self.avg_throughput,
            "avg_reward": self.avg_reward,
            "total_safety_violations": self.total_safety_violations,
        }


class MetricsCollector:
    """
    Recolecta métricas por episodio durante evaluación.
    Soporta múltiples controladores y condiciones de estrés.
    """

    def __init__(self, output_dir: str = "results/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._current: Optional[EpisodeMetrics] = None
        self._step_data: Dict[str, List[float]] = defaultdict(list)

    def start_episode(
        self,
        controller_name: str,
        episode_id: int,
        saturation_level: float = 1.0,
        sensor_noise_std: float = 0.0,
    ) -> None:
        self._current = EpisodeMetrics(
            controller_name=controller_name,
            episode_id=episode_id,
            saturation_level=saturation_level,
            sensor_noise_std=sensor_noise_std,
        )
        self._step_data.clear()

    def record_step(
        self,
        reward: float,
        queue_lengths: Optional[List[float]] = None,
        waiting_times: Optional[List[float]] = None,
        safety_violation: bool = False,
        illegal_attempt: bool = False,
    ) -> None:
        if self._current is None:
            return
        self._step_data["rewards"].append(reward)
        if queue_lengths:
            self._step_data["queues"].extend(queue_lengths)
        if waiting_times:
            self._step_data["waiting"].extend(waiting_times)
        if safety_violation:
            self._current.safety_violations += 1
        if illegal_attempt:
            self._current.illegal_action_attempts += 1

    def end_episode(self, travel_times: Optional[List[float]] = None) -> EpisodeMetrics:
        if self._current is None:
            raise RuntimeError("end_episode llamado sin start_episode previo.")

        self._current.total_reward = float(
            np.sum(self._step_data.get("rewards", [0]))
        )
        self._current.avg_queue_length = float(
            np.mean(self._step_data.get("queues", [0]))
        )
        self._current.avg_waiting_time = float(
            np.mean(self._step_data.get("waiting", [0]))
        )
        self._current.episode_length = len(self._step_data.get("rewards", []))

        if travel_times:
            self._current.avg_travel_time = float(np.mean(travel_times))
            self._current.throughput = float(len(travel_times))

        metric = self._current
        self._current = None
        return metric

    def compile_report(
        self, results: List[EpisodeMetrics], controller_name: str
    ) -> BenchmarkReport:
        report = BenchmarkReport(
            controller_name=controller_name,
            num_episodes=len(results),
            results=results,
        )
        return report

    def save_comparison(
        self, reports: List[BenchmarkReport], filename: str = "benchmark.csv"
    ) -> Path:
        """Guarda tabla comparativa entre controladores."""
        rows = [r.to_dict() for r in reports]
        df = pd.DataFrame(rows)
        out_path = self.output_dir / filename
        df.to_csv(out_path, index=False)

        # También guardar JSON detallado
        json_path = self.output_dir / filename.replace(".csv", ".json")
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2)

        logger.info("Reporte comparativo guardado en '%s'.", out_path)
        self._print_table(df)
        return out_path

    def _print_table(self, df: pd.DataFrame) -> None:
        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARK COMPARATIVO")
        logger.info("=" * 70)
        cols = [
            "controller", "avg_travel_time_mean", "avg_travel_time_std",
            "avg_queue_length", "avg_throughput", "total_safety_violations",
        ]
        available = [c for c in cols if c in df.columns]
        logger.info("\n%s\n", df[available].to_string(index=False))


class StressTestRunner:
    """
    Ejecuta pruebas de estrés bajo diferentes niveles de saturación
    y ruido en sensores (Fase III).
    """

    def __init__(self, env, agent, collector: MetricsCollector, cfg: dict):
        self.env = env
        self.agent = agent
        self.collector = collector
        self.stress_cfg = cfg["stress_testing"]
        self.num_eval_episodes: int = self.stress_cfg["num_eval_episodes"]

    def run(self) -> Dict[str, List[EpisodeMetrics]]:
        results_by_condition: Dict[str, List[EpisodeMetrics]] = {}

        sat_levels = self.stress_cfg["saturation_levels"]
        noise_levels = self.stress_cfg["sensor_noise_std"]

        for sat in sat_levels:
            for noise in noise_levels:
                condition_key = f"sat{sat}_noise{noise}"
                logger.info(
                    "Prueba de estrés: saturación=%.1f, ruido=%.2f", sat, noise
                )
                ep_results = self._run_condition(sat, noise)
                results_by_condition[condition_key] = ep_results

        return results_by_condition

    def _run_condition(
        self, saturation: float, noise_std: float
    ) -> List[EpisodeMetrics]:
        results = []
        for ep in range(self.num_eval_episodes):
            self.collector.start_episode(
                controller_name="drl_agent",
                episode_id=ep,
                saturation_level=saturation,
                sensor_noise_std=noise_std,
            )
            obs, info = self.env.reset()

            # Aplicar ruido de sensor
            if noise_std > 0:
                obs = obs + np.random.normal(0, noise_std, obs.shape).astype(np.float32)

            done = False
            while not done:
                masks = info.get("action_masks", None)
                action = self.agent.predict(obs, masks)
                obs, reward, terminated, truncated, info = self.env.step(action)

                if noise_std > 0:
                    obs = obs + np.random.normal(0, noise_std, obs.shape).astype(np.float32)

                self.collector.record_step(reward=reward)
                done = terminated or truncated

            metric = self.collector.end_episode()
            results.append(metric)

        return results
