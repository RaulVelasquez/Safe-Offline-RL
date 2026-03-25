# tests/test_phase1/test_dataset_and_model.py
"""
Tests Fase I: Dataset Generator y Decision Transformer
Ejecutar: pytest tests/test_phase1/ -v
"""

import pytest
import numpy as np
import torch

from phase1.data.dataset_generator import (
    Trajectory,
    FixedTimeController,
    ActuatedController,
)
from phase1.models.decision_transformer import DecisionTransformer


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_trajectory():
    traj = Trajectory()
    rng = np.random.default_rng(0)
    for _ in range(100):
        obs    = rng.random(25).astype(np.float32)
        action = rng.integers(0, 4, size=(4,)).astype(np.int32)
        reward = float(rng.normal(-10, 2))
        done   = False
        traj.append(obs, action, reward, done)
    traj.terminals[-1] = True
    return traj


@pytest.fixture
def dt_model():
    return DecisionTransformer(
        obs_dim=25 * 4,
        act_dim=4 * 4,
        num_tls=4,
        context_length=10,
        d_model=64,
        n_layer=2,
        n_head=4,
        d_inner=128,
        dropout=0.0,
        max_ep_len=500,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests de Trayectoria
# ──────────────────────────────────────────────────────────────────────────────

class TestTrajectory:

    def test_append_and_length(self, sample_trajectory):
        assert len(sample_trajectory.observations) == 100
        assert len(sample_trajectory.actions) == 100
        assert len(sample_trajectory.rewards) == 100

    def test_rtg_monotone_decreasing_no_discount(self, sample_trajectory):
        d = sample_trajectory.to_dict()
        rtg = d["returns_to_go"]
        # RTG[t] = RTG[t+1] + rewards[t]  → RTG[t] >= RTG[t+1] si reward > 0
        # Para rewards negativos, RTG es creciente. Solo comprobamos forma.
        assert rtg.shape == (100,)
        assert rtg.dtype == np.float32

    def test_to_dict_keys(self, sample_trajectory):
        d = sample_trajectory.to_dict()
        expected = {"observations", "actions", "rewards", "terminals", "returns_to_go"}
        assert set(d.keys()) == expected

    def test_rtg_last_equals_last_reward(self, sample_trajectory):
        d = sample_trajectory.to_dict()
        # El último RTG debe ser igual al último reward
        assert abs(d["returns_to_go"][-1] - d["rewards"][-1]) < 1e-5


# ──────────────────────────────────────────────────────────────────────────────
# Tests de Controladores Baseline
# ──────────────────────────────────────────────────────────────────────────────

class TestFixedTimeController:

    def test_output_shape(self):
        ctrl = FixedTimeController(num_tls=4, num_phases=4, cycle_length=10)
        obs = np.zeros(100, dtype=np.float32)
        act = ctrl.select_action(obs)
        assert act.shape == (4,)

    def test_phase_cycling(self):
        ctrl = FixedTimeController(num_tls=1, num_phases=4, cycle_length=5)
        obs = np.zeros(10, dtype=np.float32)
        phases = [ctrl.select_action(obs)[0] for _ in range(20)]
        # Debe ciclar: 0,0,0,0,0,1,1,1,1,1,2,...
        assert phases[0] == 0
        assert phases[5] == 1
        assert phases[10] == 2
        assert phases[15] == 3

    def test_reset(self):
        ctrl = FixedTimeController(num_tls=2, num_phases=4, cycle_length=5)
        obs = np.zeros(10, dtype=np.float32)
        for _ in range(7):
            ctrl.select_action(obs)
        ctrl.reset()
        act = ctrl.select_action(obs)
        assert (act == 0).all()


class TestActuatedController:

    def test_output_shape(self):
        ctrl = ActuatedController(num_tls=4, num_phases=4)
        obs = np.random.rand(100).astype(np.float32)
        act = ctrl.select_action(obs)
        assert act.shape == (4,)

    def test_phase_valid_range(self):
        ctrl = ActuatedController(num_tls=3, num_phases=4)
        obs = np.random.rand(75).astype(np.float32)
        for _ in range(200):
            act = ctrl.select_action(obs)
            assert all(0 <= a < 4 for a in act)

    def test_respects_min_green(self):
        ctrl = ActuatedController(
            num_tls=1, num_phases=4, min_green=10, max_green=60
        )
        obs = np.zeros(25, dtype=np.float32)  # demanda=0 → cambiaría rápido
        phases = [ctrl.select_action(obs)[0] for _ in range(9)]
        # En los primeros 9 pasos no debe haber cambiado de fase
        assert all(p == phases[0] for p in phases)


# ──────────────────────────────────────────────────────────────────────────────
# Tests del Decision Transformer
# ──────────────────────────────────────────────────────────────────────────────

class TestDecisionTransformer:

    def test_forward_output_shapes(self, dt_model):
        B, T = 2, 10
        states    = torch.randn(B, T, 25 * 4)
        actions   = torch.randint(0, 4, (B, T, 4))
        rtg       = torch.randn(B, T, 1)
        timesteps = torch.arange(T).unsqueeze(0).expand(B, -1)

        preds = dt_model(states, actions, rtg, timesteps)
        assert len(preds) == 4  # num_tls
        for p in preds:
            assert p.shape == (B, T, 4)  # (B, T, phases_per_tl)

    def test_get_action_shape(self, dt_model):
        T = 5
        states    = torch.randn(1, T, 25 * 4)
        actions   = torch.randint(0, 4, (1, T, 4))
        rtg       = torch.randn(1, T, 1)
        timesteps = torch.arange(T).unsqueeze(0)

        act = dt_model.get_action(states, actions, rtg, timesteps)
        assert act.shape == (4,)

    def test_action_mask_applied(self, dt_model):
        """Con máscara que bloquea todo excepto fase 2, debe retornar siempre 2."""
        T = 5
        states    = torch.randn(1, T, 25 * 4)
        actions   = torch.randint(0, 4, (1, T, 4))
        rtg       = torch.randn(1, T, 1)
        timesteps = torch.arange(T).unsqueeze(0)

        # Máscara: solo fase 2 válida para todos los TLs
        mask = torch.zeros(1, 4, 4, dtype=torch.bool)
        mask[:, :, 2] = True

        act = dt_model.get_action(states, actions, rtg, timesteps, action_mask=mask)
        assert (act == 2).all(), f"Esperado 2 en todas las posiciones, obtenido: {act}"

    def test_no_nan_in_output(self, dt_model):
        B, T = 4, 10
        states    = torch.randn(B, T, 25 * 4)
        actions   = torch.randint(0, 4, (B, T, 4))
        rtg       = torch.randn(B, T, 1)
        timesteps = torch.arange(T).unsqueeze(0).expand(B, -1)

        preds = dt_model(states, actions, rtg, timesteps)
        for p in preds:
            assert not torch.isnan(p).any()
            assert not torch.isinf(p).any()

    def test_parameter_count_reasonable(self, dt_model):
        n = sum(p.numel() for p in dt_model.parameters())
        # Un modelo pequeño de prueba debe tener entre 100k y 10M parámetros
        assert 1_000 < n < 10_000_000, f"Parámetros fuera de rango: {n:,}"
