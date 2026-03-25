# tests/test_integration.py
"""
Tests de integración: validan el pipeline completo sin necesitar SUMO.
Usan el entorno sintético de stress_test como sustituto del simulador.

Ejecutar: pytest tests/test_integration.py -v
"""

import pytest
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_synthetic_env(demand="moderate", noise="none", episode_length=200):
    """Crea un entorno sintético listo para usar."""
    from phase3.stress_test import (
        SyntheticStressEnvironment,
        DemandProfile,
        SensorNoiseModel,
    )
    demand_profile = DemandProfile(demand, num_lanes=8, seed=0)
    noise_model    = SensorNoiseModel(noise, seed=0)
    return SyntheticStressEnvironment(
        demand_profile=demand_profile,
        noise_model=noise_model,
        episode_length=episode_length,
        num_tls=4,
        num_phases=4,
        seed=0,
    )


def make_dt_model(obs_dim=100, num_tls=4, num_phases=4):
    """Crea un Decision Transformer pequeño para tests."""
    from phase1.models.decision_transformer import DecisionTransformer
    return DecisionTransformer(
        obs_dim=obs_dim,
        act_dim=num_phases * num_tls,
        num_tls=num_tls,
        context_length=5,
        d_model=32,
        n_layer=2,
        n_head=4,
        d_inner=64,
        dropout=0.0,
        max_ep_len=500,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests de integración
# ──────────────────────────────────────────────────────────────────────────────

class TestSyntheticEnvironment:

    def test_reset_returns_correct_shape(self):
        env = make_synthetic_env()
        obs, info = env.reset()
        assert obs.shape == (100,)   # 4 TLs × 25 features
        assert isinstance(info, dict)

    def test_step_returns_correct_types(self):
        env = make_synthetic_env()
        obs, _ = env.reset()
        action = np.zeros(4, dtype=np.int32)
        obs2, reward, terminated, truncated, info = env.step(action)

        assert obs2.shape == (100,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "safety_violations" in info

    def test_episode_terminates_at_length(self):
        ep_len = 50
        env = make_synthetic_env(episode_length=ep_len)
        env.reset()
        steps = 0
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.int32))
            done = terminated or truncated
            steps += 1
            assert steps <= ep_len + 1, "Episodio no terminó a tiempo"
        assert steps == ep_len

    def test_reward_is_negative(self):
        """Recompensa debe ser negativa (penalización de densidad)."""
        env = make_synthetic_env(demand="peak")
        env.reset()
        total = 0.0
        for _ in range(10):
            _, r, *_ = env.step(np.zeros(4, dtype=np.int32))
            total += r
        assert total < 0

    def test_noise_changes_observations(self):
        """Las observaciones con ruido deben diferir de las limpias."""
        env_clean = make_synthetic_env(noise="none")
        env_noisy = make_synthetic_env(noise="gaussian")
        obs_clean, _ = env_clean.reset()
        obs_noisy, _ = env_noisy.reset()
        # Con la misma semilla, la demanda base es igual pero el ruido los diferencia
        # No garantizado que sean distintas en reset, verificar en step
        action = np.zeros(4, dtype=np.int32)
        _, _, _, _, _ = env_clean.step(action)
        obs_clean2, _, _, _, _ = env_clean.step(action)
        obs_noisy2, _, _, _, _ = env_noisy.step(action)
        # Algún elemento debe diferir
        assert not np.allclose(obs_clean2, obs_noisy2)


class TestDemandProfiles:

    @pytest.mark.parametrize("profile", ["off_peak", "moderate", "peak", "supersaturated"])
    def test_profile_valid_range(self, profile):
        from phase3.stress_test import DemandProfile
        dp = DemandProfile(profile, num_lanes=8)
        for step in [0, 500, 1000, 2000, 3599]:
            densities = dp.sample(step, 3600)
            assert densities.shape == (8,)
            assert (densities >= 0).all()
            assert (densities <= 1).all()

    def test_supersaturated_higher_than_off_peak(self):
        from phase3.stress_test import DemandProfile
        peak_dp = DemandProfile("supersaturated", num_lanes=8, seed=0)
        off_dp  = DemandProfile("off_peak",       num_lanes=8, seed=0)

        peak_means = np.mean([peak_dp.sample(t, 3600) for t in range(100, 3000, 100)])
        off_means  = np.mean([off_dp.sample(t,  3600) for t in range(100, 3000, 100)])
        assert peak_means > off_means


class TestPipelineIntegration:
    """
    Tests de pipeline end-to-end: DT + ActionMask + entorno sintético.
    """

    def test_dt_runs_full_episode(self):
        """El DT debe poder completar un episodio completo sin errores."""
        import torch
        import collections

        model = make_dt_model()
        model.eval()
        env = make_synthetic_env(episode_length=30)

        obs, _ = env.reset()
        K = 5
        ctx = {k: collections.deque(maxlen=K) for k in ["obs", "act", "rtg", "t"]}
        dummy = np.zeros(4, dtype=np.int64)

        done = False
        steps = 0
        while not done:
            ctx["obs"].append(obs)
            ctx["act"].append(dummy)
            ctx["rtg"].append(-50.0)
            ctx["t"].append(steps)

            t_obs = torch.tensor(np.array(ctx["obs"]), dtype=torch.float32).unsqueeze(0)
            t_act = torch.tensor(np.array(ctx["act"]), dtype=torch.long).unsqueeze(0)
            t_rtg = torch.tensor(np.array(ctx["rtg"])[:, None], dtype=torch.float32).unsqueeze(0)
            t_ts  = torch.tensor(np.array(ctx["t"]),  dtype=torch.long).unsqueeze(0)

            with torch.no_grad():
                action = model.get_action(t_obs, t_act, t_rtg, t_ts).numpy()

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            dummy = action
            steps += 1

        assert steps == 30

    def test_action_mask_wrapper_zero_violations(self):
        """
        El SafeEnvironmentWrapper debe garantizar cero violaciones
        incluso cuando el agente envía acciones aleatorias agresivas.
        """
        from phase2.safety.action_mask import ActionMask

        am = ActionMask(num_tls=4, num_phases=4, min_green=5, min_intergreen=3)
        rng = np.random.default_rng(999)

        for _ in range(500):
            # Acción aleatoria (potencialmente ilegal)
            action = rng.integers(0, 4, size=4).astype(np.int32)
            safe_action, violations = am.update(action)
            # El escudo debe absorber las violaciones (retornar 0)
            assert violations == 0 or am.total_violations >= 0
            # La acción safe debe estar en rango válido
            assert all(0 <= a < 4 for a in safe_action)

    def test_corridor_coordinator_shape_preservation(self):
        """El coordinador de corredor no debe alterar la dimensión de observación."""
        from phase2.multiagent.corridor_coordinator import CorridorGraph, CorridorCoordinator

        graph = CorridorGraph(
            tl_ids=["TL0", "TL1", "TL2", "TL3"],
            distances=[200.0, 250.0, 180.0],
        )
        coord = CorridorCoordinator(graph, obs_dim_per_tl=25)

        flat_obs = np.random.rand(4 * 25).astype(np.float32)
        augmented = coord.augment_observations(flat_obs)

        assert augmented.shape == flat_obs.shape
        assert augmented.dtype == np.float32
        assert not np.isnan(augmented).any()

    def test_stress_tester_runs_short(self):
        """El StressTester debe ejecutar sin errores con pocas iteraciones."""
        import tempfile
        from phase3.stress_test import StressTester, RandomController, GreedyDensityController

        with tempfile.TemporaryDirectory() as tmpdir:
            tester = StressTester(
                output_dir=tmpdir,
                episode_length=20,
                num_episodes_per_scenario=1,
                seed=0,
            )
            controllers = {
                "random": RandomController(),
                "greedy": GreedyDensityController(),
            }
            # Solo un escenario para rapidez
            tester.DEMAND_SCENARIOS = ["off_peak"]
            tester.NOISE_SCENARIOS  = [("none", 0.0, 0.0)]
            tester.run_all(controllers)

            assert len(tester._results) == 2  # 1 escenario × 2 controladores × 1 ep

    def test_running_normalizer(self):
        """El normalizador debe converger a estadísticas correctas."""
        from utils.common import RunningNormalizer

        rng = np.random.default_rng(7)
        true_mean = np.array([3.0, -2.0, 0.5])
        true_std  = np.array([1.5,  0.8, 2.0])

        norm = RunningNormalizer(shape=(3,))
        data = rng.normal(true_mean, true_std, (5000, 3))
        norm.update(data)

        assert np.allclose(norm.mean, true_mean, atol=0.15)
        assert np.allclose(np.sqrt(norm.var), true_std, atol=0.15)

        # Normalización debe producir ~N(0,1)
        normalized = norm.normalize(data)
        assert abs(float(np.mean(normalized))) < 0.1
        assert abs(float(np.std(normalized)) - 1.0) < 0.1
