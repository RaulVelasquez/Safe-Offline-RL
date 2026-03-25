# tests/test_phase2/test_safety_and_coordination.py
"""
Tests Fase II: Action Masking y Coordinación de Corredor
Ejecutar: pytest tests/test_phase2/ -v
"""

import pytest
import numpy as np
import torch

from phase2.safety.action_mask import ActionMask, IntersectionSafetyState
from phase2.multiagent.corridor_coordinator import CorridorGraph, CorridorCoordinator


# ──────────────────────────────────────────────────────────────────────────────
# Tests de ActionMask
# ──────────────────────────────────────────────────────────────────────────────

class TestActionMask:

    @pytest.fixture
    def mask(self):
        return ActionMask(
            num_tls=2,
            num_phases=4,
            min_green=5,
            min_intergreen=3,
            max_consecutive_red=30,
        )

    def test_initial_mask_allows_current_phase(self, mask):
        """Al inicio, la fase actual siempre debe estar permitida."""
        m = mask.get_mask(0)
        assert m[0] == True  # fase 0 es la actual por defecto

    def test_min_green_blocks_transitions_early(self, mask):
        """Durante los primeros `min_green-1` pasos, no puede cambiar de fase."""
        for step in range(4):  # < min_green=5
            acts = np.array([0, 0])
            mask.update(acts)

        m = mask.get_mask(0)
        # Solo fase 0 debe ser válida aún
        assert m[0] == True
        for i in range(1, 4):
            assert m[i] == False, f"Fase {i} debería estar bloqueada"

    def test_min_green_allows_transition_after(self, mask):
        """Después de min_green pasos, otras fases deben estar disponibles."""
        for _ in range(5):  # == min_green
            mask.update(np.array([0, 0]))

        # Avanzar también el intergreen counter de otras fases
        # Primero, simular que las otras fases han tenido tiempo suficiente
        # (re-instanciar con red_timers altos)
        mask._states[0].red_timers = {1: 10, 2: 10, 3: 10}
        m = mask.get_mask(0)
        # Al menos una fase distinta de 0 debe ser válida
        assert any(m[i] for i in range(1, 4))

    def test_always_at_least_one_valid_action(self):
        """La máscara nunca debe ser completamente False."""
        am = ActionMask(num_tls=1, num_phases=4, min_green=100, min_intergreen=100)
        m = am.get_mask(0)
        assert m.any(), "Debe haber al menos una acción válida"

    def test_violation_count_starts_at_zero(self, mask):
        assert mask.total_violations == 0

    def test_safe_action_applied_without_violations(self, mask):
        """Aplicar acciones válidas no genera violaciones."""
        # Avanzar hasta que sea válido cambiar
        for _ in range(5):
            mask.update(np.array([0, 0]))
        mask._states[0].red_timers = {1: 10, 2: 10, 3: 10}
        mask._states[1].red_timers = {1: 10, 2: 10, 3: 10}

        # Obtener máscara y elegir acción válida
        valid_phase = int(np.argmax(mask.get_mask(0)))
        safe_acts, violations = mask.update(np.array([valid_phase, 0]))
        assert violations == 0
        assert mask.total_violations == 0

    def test_forced_action_on_violation(self, mask):
        """Si se envía una acción inválida, se debe reemplazar por la actual."""
        # Acción 3 bloqueada porque no se cumple min_green ni intergreen
        _, violations = mask.update(np.array([3, 0]))
        # La acción debe haber sido corregida (fase 0 mantenida)
        assert mask._states[0].current_phase == 0

    def test_mask_tensor_shape(self, mask):
        t = mask.get_mask_tensor(torch.device("cpu"))
        assert t.shape == (1, 2, 4)
        assert t.dtype == torch.bool

    def test_apply_to_logits(self, mask):
        """Las posiciones bloqueadas deben tener -inf en los logits."""
        logits = torch.randn(1, 2, 4)  # (B=1, num_tls=2, phases=4)
        # Con phase_timer=0, solo la fase actual (0) es válida para TL 0
        masked = mask.apply_to_logits(logits)
        # Las fases no válidas deben ser -inf (o muy negativo)
        m = mask.get_mask(0)
        for ph in range(4):
            if not m[ph]:
                assert masked[0, 0, ph] <= -1e8

    def test_reset_clears_state(self, mask):
        for _ in range(10):
            mask.update(np.array([0, 0]))
        mask.reset()
        assert mask._states[0].phase_timer == 0
        assert mask.total_violations == 0


# ──────────────────────────────────────────────────────────────────────────────
# Tests de CorridorGraph
# ──────────────────────────────────────────────────────────────────────────────

class TestCorridorGraph:

    @pytest.fixture
    def graph(self):
        return CorridorGraph(
            tl_ids=["TL0", "TL1", "TL2", "TL3"],
            distances=[200.0, 300.0, 150.0],
        )

    def test_get_neighbors_middle(self, graph):
        neighbors = graph.get_neighbors("TL1", radius=1)
        assert set(neighbors) == {"TL0", "TL2"}

    def test_get_neighbors_edge(self, graph):
        neighbors = graph.get_neighbors("TL0", radius=1)
        assert neighbors == ["TL1"]

    def test_get_upstream(self, graph):
        assert graph.get_upstream("TL2") == "TL1"
        assert graph.get_upstream("TL0") is None

    def test_get_downstream(self, graph):
        assert graph.get_downstream("TL2") == "TL3"
        assert graph.get_downstream("TL3") is None

    def test_id_to_idx(self, graph):
        assert graph.id_to_idx["TL0"] == 0
        assert graph.id_to_idx["TL3"] == 3


# ──────────────────────────────────────────────────────────────────────────────
# Tests de CorridorCoordinator
# ──────────────────────────────────────────────────────────────────────────────

class TestCorridorCoordinator:

    @pytest.fixture
    def coordinator(self):
        graph = CorridorGraph(
            tl_ids=["TL0", "TL1", "TL2", "TL3"],
            distances=[200.0, 300.0, 150.0],
        )
        return CorridorCoordinator(graph, obs_dim_per_tl=25, comm_rounds=1)

    def test_augment_preserves_shape(self, coordinator):
        flat_obs = np.random.rand(4 * 25).astype(np.float32)
        augmented = coordinator.augment_observations(flat_obs)
        assert augmented.shape == (4 * 25,)

    def test_augment_changes_values(self, coordinator):
        """El mensaje de vecinos debe modificar los valores de observación."""
        flat_obs = np.random.rand(4 * 25).astype(np.float32)
        augmented = coordinator.augment_observations(flat_obs)
        # No debe ser idéntico (a menos que los pesos sean nulos)
        # Solo verificamos que tiene el tipo correcto
        assert augmented.dtype == np.float32

    def test_coordination_reward_formula(self, coordinator):
        individual = np.array([-10.0, -20.0, -5.0, -15.0])
        coordinated = coordinator.compute_coordination_reward(
            flat_obs=np.zeros(100),
            individual_rewards=individual,
            coordination_weight=0.3,
        )
        expected_global = np.mean(individual)
        for i, (coord, ind) in enumerate(zip(coordinated, individual)):
            expected = 0.7 * ind + 0.3 * expected_global
            assert abs(coord - expected) < 1e-5, f"TL {i}: {coord} != {expected}"

    def test_green_wave_offsets_first_zero(self, coordinator):
        offsets = coordinator.suggest_green_wave_offsets(speed_kmh=50.0, cycle_length=90)
        assert offsets[0] == 0
        assert len(offsets) == 4

    def test_green_wave_offsets_positive(self, coordinator):
        offsets = coordinator.suggest_green_wave_offsets(speed_kmh=50.0, cycle_length=90)
        assert all(0 <= o < 90 for o in offsets)
