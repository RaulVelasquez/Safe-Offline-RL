# tests/test_phase3/test_online_and_metrics.py
"""
Tests Fase III: Online Replay Buffer y Métricas
Ejecutar: pytest tests/test_phase3/ -v
"""

import pytest
import numpy as np
import json
import os
import tempfile

from phase3.online.online_finetuner import OnlineReplayBuffer
from phase3.metrics.metrics_extractor import (
    Trajectory,
    EpisodeMetrics,
    BenchmarkResults,
    MetricsCollector,
)


# ──────────────────────────────────────────────────────────────────────────────
# Tests de OnlineReplayBuffer
# ──────────────────────────────────────────────────────────────────────────────

class TestOnlineReplayBuffer:

    @pytest.fixture
    def buffer(self):
        return OnlineReplayBuffer(capacity=100)

    def _push_n(self, buffer, n: int) -> None:
        for i in range(n):
            obs    = np.random.rand(25).astype(np.float32)
            action = np.random.randint(0, 4, size=(4,))
            buffer.push(obs, action, reward=-float(i), rtg=-float(i * 0.5), timestep=i)

    def test_empty_buffer_length(self, buffer):
        assert len(buffer) == 0

    def test_push_increases_length(self, buffer):
        self._push_n(buffer, 10)
        assert len(buffer) == 10

    def test_capacity_respected(self, buffer):
        self._push_n(buffer, 150)
        assert len(buffer) == 100  # capacity=100

    def test_sample_returns_none_when_insufficient(self, buffer):
        self._push_n(buffer, 5)
        result = buffer.sample(batch_size=4, context_length=10)
        assert result is None  # 5 < 10+1

    def test_sample_shapes(self, buffer):
        self._push_n(buffer, 50)
        batch = buffer.sample(batch_size=8, context_length=10)
        assert batch is not None
        assert batch["observations"].shape == (8, 10, 25)
        assert batch["actions"].shape      == (8, 10, 4)
        assert batch["rtg"].shape          == (8, 10, 1)
        assert batch["timesteps"].shape    == (8, 10)

    def test_sample_dtype(self, buffer):
        self._push_n(buffer, 50)
        batch = buffer.sample(batch_size=4, context_length=5)
        import torch
        assert batch["observations"].dtype == torch.float32
        assert batch["actions"].dtype      == torch.int64
        assert batch["rtg"].dtype          == torch.float32
        assert batch["timesteps"].dtype    == torch.int64

    def test_fifo_order(self):
        """Los items más nuevos desplazan a los más antiguos."""
        buf = OnlineReplayBuffer(capacity=5)
        for i in range(10):
            obs = np.full(10, i, dtype=np.float32)
            buf.push(obs, np.zeros(4, dtype=int), float(-i), float(-i), i)
        # Los últimos 5 (i=5..9) deben estar en el buffer
        obs_arr = np.array(list(buf._obs))
        for row in obs_arr:
            assert row[0] >= 5


# ──────────────────────────────────────────────────────────────────────────────
# Tests de EpisodeMetrics y BenchmarkResults
# ──────────────────────────────────────────────────────────────────────────────

class TestEpisodeMetrics:

    def test_to_dict_has_all_keys(self):
        em = EpisodeMetrics(
            episode=0, controller="test",
            total_reward=-100.0, avg_travel_time=45.2,
            avg_queue_length=3.5, safety_violations=0,
            throughput=120, steps=3600,
        )
        d = em.to_dict()
        expected = {
            "episode", "controller", "total_reward",
            "avg_travel_time", "avg_queue_length",
            "safety_violations", "throughput", "steps",
        }
        assert set(d.keys()) == expected


class TestBenchmarkResults:

    @pytest.fixture
    def results_with_data(self):
        br = BenchmarkResults(controller="DT")
        for i in range(10):
            br.add(EpisodeMetrics(
                episode=i, controller="DT",
                total_reward=float(-100 + i * 5),
                avg_travel_time=float(50 - i * 0.5),
                avg_queue_length=float(4 - i * 0.1),
                safety_violations=0,
                throughput=100 + i * 2,
                steps=3600,
            ))
        return br

    def test_summary_has_correct_keys(self, results_with_data):
        s = results_with_data.summary()
        assert "total_reward" in s
        assert "avg_travel_time" in s
        assert "safety_violations" in s
        assert "throughput" in s

    def test_summary_mean_correct(self, results_with_data):
        s = results_with_data.summary()
        rewards = [float(-100 + i * 5) for i in range(10)]
        expected_mean = np.mean(rewards)
        assert abs(s["total_reward"]["mean"] - expected_mean) < 1e-4

    def test_zero_violations_tracked(self, results_with_data):
        s = results_with_data.summary()
        assert s["safety_violations"]["total"] == 0

    def test_empty_summary_returns_empty(self):
        br = BenchmarkResults(controller="empty")
        assert br.summary() == {}


# ──────────────────────────────────────────────────────────────────────────────
# Tests de MetricsCollector
# ──────────────────────────────────────────────────────────────────────────────

class TestMetricsCollector:

    def test_violations_accumulated(self):
        collector = MetricsCollector()
        for _ in range(5):
            collector.update({"safety_violations": 2})
        collector.update({"safety_violations": 0})
        assert collector._violations == 10

    def test_steps_counted(self):
        collector = MetricsCollector()
        for _ in range(7):
            collector.update({})
        assert collector._steps == 7

    def test_finalize_returns_episode_metrics(self):
        collector = MetricsCollector()
        for _ in range(10):
            collector.update({"safety_violations": 0})
        em = collector.finalize(controller="test", episode=1, total_reward=-50.0)
        assert isinstance(em, EpisodeMetrics)
        assert em.controller == "test"
        assert em.total_reward == -50.0
        assert em.steps == 10
        assert em.safety_violations == 0
