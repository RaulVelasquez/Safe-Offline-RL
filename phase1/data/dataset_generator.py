# phase1/data/dataset_generator.py
"""
Fase I – Síntesis del Conjunto de Datos
Genera trayectorias expertas usando controladores tradicionales
(tiempo fijo y actuado) para el corpus de entrenamiento offline.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from phase1.env.sumo_env import SUMOTrafficEnv


# ──────────────────────────────────────────────────────────────────────────────
# Estructura de datos de una trayectoria
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Trajectory:
    """Un episodio completo de interacción agente-entorno."""
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    terminals: List[bool] = field(default_factory=list)

    def append(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(done)

    def to_dict(self) -> dict:
        return {
            "observations": np.array(self.observations, dtype=np.float32),
            "actions": np.array(self.actions, dtype=np.int32),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "terminals": np.array(self.terminals, dtype=bool),
            "returns_to_go": self._compute_rtg(),
        }

    def _compute_rtg(self, gamma: float = 1.0) -> np.ndarray:
        """Return-to-go para cada timestep (sin descuento por defecto)."""
        rewards = np.array(self.rewards, dtype=np.float32)
        rtg = np.zeros_like(rewards)
        cumulative = 0.0
        for t in reversed(range(len(rewards))):
            cumulative = rewards[t] + gamma * cumulative
            rtg[t] = cumulative
        return rtg


# ──────────────────────────────────────────────────────────────────────────────
# Controladores baseline para generación de datos
# ──────────────────────────────────────────────────────────────────────────────

class FixedTimeController:
    """
    Controlador de tiempo fijo: cicla por las fases cada `cycle_length` pasos.
    """

    def __init__(self, num_tls: int, num_phases: int, cycle_length: int = 30):
        self.num_tls = num_tls
        self.num_phases = num_phases
        self.cycle_length = cycle_length
        self._step = 0

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        phase = (self._step // self.cycle_length) % self.num_phases
        self._step += 1
        return np.full(self.num_tls, phase, dtype=np.int32)

    def reset(self) -> None:
        self._step = 0


class ActuatedController:
    """
    Controlador actuado simplificado:
    Extiende la fase verde si hay demanda detectada (densidad > umbral).
    """

    def __init__(
        self,
        num_tls: int,
        num_phases: int,
        obs_dim_per_tl: int = 25,
        density_threshold: float = 0.3,
        min_green: int = 5,
        max_green: int = 60,
    ):
        self.num_tls = num_tls
        self.num_phases = num_phases
        self.obs_dim = obs_dim_per_tl
        self.density_threshold = density_threshold
        self.min_green = min_green
        self.max_green = max_green

        self._phase_timers = np.zeros(num_tls, dtype=np.int32)
        self._current_phases = np.zeros(num_tls, dtype=np.int32)

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        actions = []
        for i in range(self.num_tls):
            tl_obs = obs[i * self.obs_dim: (i + 1) * self.obs_dim]
            # Densidades están en los primeros 8 valores
            densities = tl_obs[:8]
            demand = np.mean(densities)

            timer = self._phase_timers[i]
            phase = self._current_phases[i]

            if timer >= self.max_green or (
                timer >= self.min_green and demand < self.density_threshold
            ):
                # Avanzar a siguiente fase
                phase = (phase + 1) % self.num_phases
                self._current_phases[i] = phase
                self._phase_timers[i] = 0
            else:
                self._phase_timers[i] += 1

            actions.append(phase)
        return np.array(actions, dtype=np.int32)

    def reset(self) -> None:
        self._phase_timers[:] = 0
        self._current_phases[:] = 0


# ──────────────────────────────────────────────────────────────────────────────
# Generador de Dataset
# ──────────────────────────────────────────────────────────────────────────────

class OfflineDatasetGenerator:
    """
    Genera y persiste trayectorias offline usando controladores baseline.

    Parámetros
    ----------
    env : SUMOTrafficEnv
        Entorno ya instanciado.
    output_dir : str
        Carpeta donde se guardarán los archivos .pkl del dataset.
    num_phases : int
        Número de fases de cada semáforo.
    """

    def __init__(
        self,
        env: SUMOTrafficEnv,
        output_dir: str = "data/offline_dataset",
        num_phases: int = 4,
    ) -> None:
        self.env = env
        self.output_dir = output_dir
        self.num_phases = num_phases
        os.makedirs(output_dir, exist_ok=True)

    # ── API pública ───────────────────────────────────────────────────────────

    def generate(
        self,
        num_fixed_episodes: int = 500,
        num_actuated_episodes: int = 500,
        save_every: int = 50,
    ) -> None:
        """
        Genera `num_fixed_episodes` episodios con FixedTimeController
        y `num_actuated_episodes` con ActuatedController.
        """
        print("=" * 60)
        print("Generando dataset offline…")
        print(f"  Fixed-time: {num_fixed_episodes} episodios")
        print(f"  Actuated:   {num_actuated_episodes} episodios")
        print("=" * 60)

        num_tls = self.env._num_tls

        fixed_ctrl = FixedTimeController(
            num_tls=num_tls,
            num_phases=self.num_phases,
        )
        actuated_ctrl = ActuatedController(
            num_tls=num_tls,
            num_phases=self.num_phases,
        )

        self._run_episodes(
            controller=fixed_ctrl,
            tag="fixed",
            n_episodes=num_fixed_episodes,
            save_every=save_every,
        )
        self._run_episodes(
            controller=actuated_ctrl,
            tag="actuated",
            n_episodes=num_actuated_episodes,
            save_every=save_every,
        )
        print("[OK] Dataset generado en:", self.output_dir)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _run_episodes(
        self,
        controller,
        tag: str,
        n_episodes: int,
        save_every: int,
    ) -> None:
        buffer: List[dict] = []

        for ep in tqdm(range(n_episodes), desc=f"[{tag}]"):
            traj = self._collect_episode(controller)
            buffer.append(traj.to_dict())
            controller.reset() if hasattr(controller, "reset") else None

            if (ep + 1) % save_every == 0 or ep == n_episodes - 1:
                chunk_id = (ep + 1) // save_every
                fname = os.path.join(
                    self.output_dir, f"{tag}_chunk_{chunk_id:04d}.pkl"
                )
                with open(fname, "wb") as f:
                    pickle.dump(buffer, f)
                buffer = []

    def _collect_episode(self, controller) -> Trajectory:
        obs, _ = self.env.reset()
        traj = Trajectory()
        done = False

        while not done:
            action = controller.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            traj.append(obs, action, reward, done)
            obs = next_obs

        return traj


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades de carga
# ──────────────────────────────────────────────────────────────────────────────

def load_offline_dataset(dataset_dir: str) -> dict:
    """
    Carga todos los chunks del dataset offline en un único dict NumPy.

    Retorna
    -------
    dict con keys: observations, actions, rewards, terminals, returns_to_go
    """
    import glob

    files = sorted(glob.glob(os.path.join(dataset_dir, "*.pkl")))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos en {dataset_dir}")

    all_obs, all_act, all_rew, all_term, all_rtg = [], [], [], [], []

    for f in files:
        with open(f, "rb") as fh:
            chunk: List[dict] = pickle.load(fh)
        for traj in chunk:
            all_obs.append(traj["observations"])
            all_act.append(traj["actions"])
            all_rew.append(traj["rewards"])
            all_term.append(traj["terminals"])
            all_rtg.append(traj["returns_to_go"])

    return {
        "observations": np.concatenate(all_obs, axis=0),
        "actions": np.concatenate(all_act, axis=0),
        "rewards": np.concatenate(all_rew, axis=0),
        "terminals": np.concatenate(all_term, axis=0),
        "returns_to_go": np.concatenate(all_rtg, axis=0),
    }
