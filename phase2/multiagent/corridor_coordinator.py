# phase2/multiagent/corridor_coordinator.py
"""
Fase II – Coordinación de Corredor Multi-Agente
Implementa el espacio de observación compartido y comunicación entre
intersecciones para sincronización espacio-temporal (onda verde).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Grafo de corredor
# ──────────────────────────────────────────────────────────────────────────────

class CorridorGraph:
    """
    Representa la topología de un corredor urbano como un grafo dirigido.
    Cada nodo es una intersección; las aristas indican relaciones upstream/downstream.

    Parámetros
    ----------
    tl_ids : list[str]
        IDs de semáforos en orden a lo largo del corredor.
    distances : list[float]
        Distancias en metros entre intersecciones consecutivas.
    """

    def __init__(self, tl_ids: List[str], distances: Optional[List[float]] = None) -> None:
        self.tl_ids = tl_ids
        self.n = len(tl_ids)
        self.distances = distances or [200.0] * (self.n - 1)
        assert len(self.distances) == self.n - 1

        self.id_to_idx: Dict[str, int] = {tid: i for i, tid in enumerate(tl_ids)}

    def get_neighbors(self, tl_id: str, radius: int = 1) -> List[str]:
        """Retorna los vecinos dentro de `radius` saltos."""
        idx = self.id_to_idx[tl_id]
        lo = max(0, idx - radius)
        hi = min(self.n, idx + radius + 1)
        return [self.tl_ids[j] for j in range(lo, hi) if j != idx]

    def get_upstream(self, tl_id: str) -> Optional[str]:
        idx = self.id_to_idx[tl_id]
        return self.tl_ids[idx - 1] if idx > 0 else None

    def get_downstream(self, tl_id: str) -> Optional[str]:
        idx = self.id_to_idx[tl_id]
        return self.tl_ids[idx + 1] if idx < self.n - 1 else None


# ──────────────────────────────────────────────────────────────────────────────
# Módulo de comunicación entre agentes (mensaje-paso)
# ──────────────────────────────────────────────────────────────────────────────

class MessagePassingLayer(nn.Module):
    """
    Una ronda de intercambio de mensajes entre intersecciones vecinas.
    Cada agente agrega la información de sus vecinos a su propia observación.

    Arquitectura:
        h_i^{t+1} = MLP(concat(h_i^t, Σ_{j∈N(i)} MLP_msg(h_j^t)))
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 64,
        msg_dim: int = 32,
    ) -> None:
        super().__init__()
        self.msg_encoder = nn.Sequential(
            nn.Linear(obs_dim, msg_dim),
            nn.ReLU(),
        )
        self.aggregator = nn.Sequential(
            nn.Linear(obs_dim + msg_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(
        self,
        obs: torch.Tensor,                    # (B, N, obs_dim)
        adj: torch.Tensor,                    # (N, N) — matriz de adyacencia
    ) -> torch.Tensor:                        # (B, N, obs_dim)
        B, N, D = obs.shape
        msgs = self.msg_encoder(obs)          # (B, N, msg_dim)

        # Agregación: suma de mensajes de vecinos
        # adj: (N, N) → (1, N, N)
        adj_exp = adj.unsqueeze(0).expand(B, -1, -1)           # (B, N, N)
        agg = torch.bmm(adj_exp, msgs)                          # (B, N, msg_dim)

        combined = torch.cat([obs, agg], dim=-1)                # (B, N, D+msg_dim)
        return self.aggregator(combined)                         # (B, N, D)


# ──────────────────────────────────────────────────────────────────────────────
# Coordinador de Corredor
# ──────────────────────────────────────────────────────────────────────────────

class CorridorCoordinator:
    """
    Gestiona la observación aumentada (local + contexto vecinal) para
    cada intersección del corredor.

    Parámetros
    ----------
    graph : CorridorGraph
    obs_dim_per_tl : int
        Dimensión de la observación local de un solo TL.
    comm_rounds : int
        Rondas de paso de mensajes por step.
    hidden_dim : int
    msg_dim : int
    """

    def __init__(
        self,
        graph: CorridorGraph,
        obs_dim_per_tl: int = 25,
        comm_rounds: int = 1,
        hidden_dim: int = 64,
        msg_dim: int = 32,
    ) -> None:
        self.graph = graph
        self.obs_dim = obs_dim_per_tl
        self.comm_rounds = comm_rounds

        # Matriz de adyacencia (vecinos inmediatos del corredor)
        N = graph.n
        adj = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            if i > 0:
                adj[i, i - 1] = 1.0
            if i < N - 1:
                adj[i, i + 1] = 1.0
        # Normalizar por grado
        deg = adj.sum(axis=1, keepdims=True).clip(min=1)
        self._adj = torch.tensor(adj / deg, dtype=torch.float32)

        self.mp_layer = MessagePassingLayer(obs_dim_per_tl, hidden_dim, msg_dim)

    def augment_observations(
        self,
        flat_obs: np.ndarray,   # (N * obs_dim,)
        device: Optional[torch.device] = None,
    ) -> np.ndarray:            # (N * obs_dim,)
        """
        Enriquece cada observación local con contexto de vecinos
        mediante paso de mensajes.
        """
        N = self.graph.n
        obs = torch.tensor(
            flat_obs.reshape(N, self.obs_dim), dtype=torch.float32
        ).unsqueeze(0)            # (1, N, obs_dim)

        adj = self._adj
        if device is not None:
            obs = obs.to(device)
            adj = adj.to(device)

        with torch.no_grad():
            for _ in range(self.comm_rounds):
                obs = self.mp_layer(obs, adj)

        return obs.squeeze(0).cpu().numpy().flatten()

    def compute_coordination_reward(
        self,
        flat_obs: np.ndarray,
        individual_rewards: np.ndarray,  # (N,)
        coordination_weight: float = 0.3,
    ) -> np.ndarray:
        """
        Ajusta las recompensas individuales con un componente de
        coordinación global (suma normalizada de todas las recompensas).
        
        r_i' = (1-w)*r_i + w * mean(r_all)
        """
        global_reward = np.mean(individual_rewards)
        return (
            (1 - coordination_weight) * individual_rewards
            + coordination_weight * global_reward
        )

    def suggest_green_wave_offsets(
        self,
        speed_kmh: float = 50.0,
        cycle_length: int = 90,
    ) -> List[int]:
        """
        Calcula offsets de tiempo para una onda verde ideal.
        Offset_i = (distance_cumulative / speed) % cycle_length

        Parámetros
        ----------
        speed_kmh : float   Velocidad objetivo en km/h
        cycle_length : int  Duración del ciclo en segundos
        """
        speed_ms = speed_kmh / 3.6
        offsets = [0]
        cumulative_dist = 0.0

        for dist in self.graph.distances:
            cumulative_dist += dist
            travel_time = cumulative_dist / speed_ms
            offset = int(travel_time) % cycle_length
            offsets.append(offset)

        return offsets
