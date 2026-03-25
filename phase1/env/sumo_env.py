# phase1/env/sumo_env.py
"""
Fase I – Calibración del Entorno
Wrapper Gymnasium sobre SUMO/TraCI para control de señales de tráfico.
Compatible con LemgoRL benchmark.
"""

from __future__ import annotations

import os
import sys
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# TraCI se instala junto a SUMO
try:
    import traci
    import sumolib
except ImportError:
    raise ImportError(
        "TraCI/sumolib no encontrado. Asegúrate de tener SUMO instalado "
        "y SUMO_HOME configurado en tu entorno."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _check_sumo_home() -> str:
    """Verifica que SUMO_HOME esté configurado."""
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home is None:
        raise EnvironmentError(
            "La variable de entorno SUMO_HOME no está configurada.\n"
            "Ejemplo: export SUMO_HOME=/usr/share/sumo"
        )
    tools = os.path.join(sumo_home, "tools")
    if tools not in sys.path:
        sys.path.append(tools)
    return sumo_home


# ──────────────────────────────────────────────────────────────────────────────
# Intersection Agent (single-TL wrapper)
# ──────────────────────────────────────────────────────────────────────────────

class IntersectionAgent:
    """
    Encapsula la lógica de observación y acción para UNA intersección.
    """

    def __init__(
        self,
        tl_id: str,
        min_green: int = 5,
        yellow_time: int = 3,
        max_green: int = 60,
        obs_radius: float = 200.0,
    ) -> None:
        self.tl_id = tl_id
        self.min_green = min_green
        self.yellow_time = yellow_time
        self.max_green = max_green
        self.obs_radius = obs_radius

        self._phase_duration: int = 0
        self._current_phase: int = 0
        self._yellow_active: bool = False
        self._yellow_countdown: int = 0
        self._num_phases: int = 0

    # ── Inicialización tras conectar TraCI ────────────────────────────────────

    def init_from_traci(self) -> None:
        logic = traci.trafficlight.getAllProgramLogics(self.tl_id)[0]
        self._num_phases = len(logic.phases)
        self._current_phase = traci.trafficlight.getPhase(self.tl_id)
        self._phase_duration = 0

    @property
    def num_phases(self) -> int:
        return self._num_phases

    # ── Observación ───────────────────────────────────────────────────────────

    def get_observation(self) -> np.ndarray:
        """
        Vector de observación por intersección:
          - Densidad de vehículos por carril (normalizada)
          - Velocidad media por carril
          - Fase actual (one-hot)
          - Tiempo en fase actual (normalizado)
        """
        lane_ids = traci.trafficlight.getControlledLanes(self.tl_id)
        unique_lanes = list(dict.fromkeys(lane_ids))  # preservar orden, sin dups

        densities, speeds = [], []
        for lane in unique_lanes:
            length = max(traci.lane.getLength(lane), 1.0)
            n_vehs = traci.lane.getLastStepVehicleNumber(lane)
            mean_spd = traci.lane.getLastStepMeanSpeed(lane)
            densities.append(min(n_vehs / (length / 7.5), 1.0))   # ~7.5m/veh
            speeds.append(min(mean_spd / 13.9, 1.0))               # ~50km/h ref

        # Rellenar hasta 8 carriles para dimensión fija
        max_lanes = 8
        densities = (densities + [0.0] * max_lanes)[:max_lanes]
        speeds = (speeds + [0.0] * max_lanes)[:max_lanes]

        # One-hot de fase
        phase_oh = [0.0] * max(self._num_phases, 1)
        if 0 <= self._current_phase < len(phase_oh):
            phase_oh[self._current_phase] = 1.0
        phase_oh = (phase_oh + [0.0] * 8)[:8]

        # Tiempo en fase normalizado
        t_norm = min(self._phase_duration / self.max_green, 1.0)

        obs = np.array(densities + speeds + phase_oh + [t_norm], dtype=np.float32)
        return obs  # shape: (25,)

    # ── Acción ────────────────────────────────────────────────────────────────

    def apply_action(self, action: int) -> None:
        """
        Aplica una acción de cambio de fase con lógica de amarillo.
        action: índice de la fase objetivo.
        """
        if self._yellow_active:
            self._yellow_countdown -= 1
            if self._yellow_countdown <= 0:
                self._yellow_active = False
                self._current_phase = self._pending_phase
                traci.trafficlight.setPhase(self.tl_id, self._current_phase)
                self._phase_duration = 0
            return

        if action != self._current_phase:
            if self._phase_duration >= self.min_green:
                # Activar amarillo intermedio
                self._yellow_active = True
                self._yellow_countdown = self.yellow_time
                self._pending_phase = action
                # SUMO usa la fase +1 como amarillo en muchos programas
                yellow_phase = self._current_phase + 1
                if yellow_phase < self._num_phases:
                    try:
                        traci.trafficlight.setPhase(self.tl_id, yellow_phase)
                    except traci.TraCIException:
                        pass
            # Si no cumple min_green, ignorar acción (safety)
        self._phase_duration += 1

    # ── Métricas locales ──────────────────────────────────────────────────────

    def get_reward(self) -> float:
        """
        Recompensa basada en presión (diferencia de densidades entre
        carriles entrantes y salientes). Minimiza congestión.
        """
        lane_ids = traci.trafficlight.getControlledLanes(self.tl_id)
        unique_lanes = list(dict.fromkeys(lane_ids))

        incoming_pressure = sum(
            traci.lane.getLastStepHaltingNumber(l) for l in unique_lanes
        )
        return -float(incoming_pressure)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Intersection SUMO Environment
# ──────────────────────────────────────────────────────────────────────────────

class SUMOTrafficEnv(gym.Env):
    """
    Entorno multi-intersección sobre SUMO para control adaptativo de señales.

    Parámetros
    ----------
    net_file, route_file, additional_file : str
        Archivos de configuración SUMO.
    tl_ids : list[str] | None
        IDs de semáforos a controlar. Si None, se usan todos.
    step_length : float
        Segundos por paso de simulación.
    episode_length : int
        Pasos por episodio.
    sumo_binary : str
        "sumo" (headless) o "sumo-gui" (visual).
    min_green, yellow_time, max_green : int
        Restricciones de tiempo de señalización (seguridad).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    OBS_DIM = 25   # dimensión por intersección

    def __init__(
        self,
        net_file: str,
        route_file: str,
        additional_file: Optional[str] = None,
        tl_ids: Optional[List[str]] = None,
        step_length: float = 1.0,
        episode_length: int = 3600,
        sumo_binary: str = "sumo",
        min_green: int = 5,
        yellow_time: int = 3,
        max_green: int = 60,
        obs_radius: float = 200.0,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        _check_sumo_home()

        self.net_file = net_file
        self.route_file = route_file
        self.additional_file = additional_file
        self._tl_ids_cfg = tl_ids
        self.step_length = step_length
        self.episode_length = episode_length
        self.sumo_binary = sumo_binary
        self.min_green = min_green
        self.yellow_time = yellow_time
        self.max_green = max_green
        self.obs_radius = obs_radius
        self.render_mode = render_mode

        self._conn_label: str = "sim_0"
        self._agents: Dict[str, IntersectionAgent] = {}
        self._step: int = 0
        self._running: bool = False

        # Espacios se definen tras primera conexión a TraCI
        # Los definimos provisionalmente con tl_ids_cfg o conteo de red
        self._num_tls = len(tl_ids) if tl_ids else 4  # fallback
        self._define_spaces()

    # ── Spaces ────────────────────────────────────────────────────────────────

    def _define_spaces(self) -> None:
        n = self._num_tls
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(n * self.OBS_DIM,),
            dtype=np.float32,
        )
        # Acción discreta por intersección: MultiDiscrete [n_phases] * n_tls
        # Usamos 4 fases por defecto; se ajusta en reset()
        self.action_space = spaces.MultiDiscrete([4] * n)

    # ── SUMO lifecycle ────────────────────────────────────────────────────────

    def _build_sumo_cmd(self) -> List[str]:
        cmd = [
            self.sumo_binary,
            "--net-file", self.net_file,
            "--route-files", self.route_file,
            "--step-length", str(self.step_length),
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--quit-on-end", "true",
        ]
        if self.additional_file:
            cmd += ["--additional-files", self.additional_file]
        return cmd

    def _start_simulation(self) -> None:
        if self._running:
            self._close_simulation()
        traci.start(self._build_sumo_cmd(), label=self._conn_label)
        self._running = True

        # Descubrir semáforos
        all_tls = traci.trafficlight.getIDList()
        tl_ids = self._tl_ids_cfg if self._tl_ids_cfg else list(all_tls)
        self._num_tls = len(tl_ids)

        self._agents = {
            tid: IntersectionAgent(
                tl_id=tid,
                min_green=self.min_green,
                yellow_time=self.yellow_time,
                max_green=self.max_green,
                obs_radius=self.obs_radius,
            )
            for tid in tl_ids
        }
        for agent in self._agents.values():
            agent.init_from_traci()

        # Redefinir espacios con info real
        self._define_spaces()
        max_phases = max(a.num_phases for a in self._agents.values())
        self.action_space = spaces.MultiDiscrete(
            [max_phases] * self._num_tls
        )

    def _close_simulation(self) -> None:
        if self._running:
            try:
                traci.close()
            except Exception:
                pass
            self._running = False

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._step = 0
        self._start_simulation()
        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self._running, "Llama reset() antes de step()."

        # Aplicar acción a cada agente
        for idx, (tid, agent) in enumerate(self._agents.items()):
            agent.apply_action(int(action[idx]))

        traci.simulationStep()
        self._step += 1

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = False
        truncated = self._step >= self.episode_length
        info = self._get_info()

        if truncated:
            self._close_simulation()

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        pass  # SUMO-GUI maneja el render

    def close(self) -> None:
        self._close_simulation()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        parts = [agent.get_observation() for agent in self._agents.values()]
        return np.concatenate(parts, axis=0).astype(np.float32)

    def _get_reward(self) -> float:
        return sum(a.get_reward() for a in self._agents.values())

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step": self._step,
            "tl_ids": list(self._agents.keys()),
        }
