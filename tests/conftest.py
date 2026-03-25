# tests/conftest.py
"""
Fixtures compartidos entre todos los tests del proyecto.
Pytest los descubre automáticamente gracias a conftest.py.
"""

import pytest
import numpy as np


@pytest.fixture(scope="session")
def rng():
    """Generador de números aleatorios reproducible."""
    return np.random.default_rng(42)


@pytest.fixture
def dummy_obs_4tl():
    """Observación plana para 4 intersecciones (4 × 25 = 100 features)."""
    return np.random.rand(100).astype(np.float32)


@pytest.fixture
def dummy_obs_single():
    """Observación de una sola intersección (25 features)."""
    return np.random.rand(25).astype(np.float32)


@pytest.fixture
def dummy_actions_4tl():
    """Acciones para 4 intersecciones (4 fases cada una)."""
    return np.array([0, 1, 2, 3], dtype=np.int32)


@pytest.fixture
def episode_data():
    """
    Dataset de episodio completo sintético para tests de entrenamiento.
    Retorna dict compatible con TrajectoryDataset.
    """
    T = 500   # pasos totales
    rng = np.random.default_rng(0)
    obs     = rng.random((T, 100)).astype(np.float32)
    actions = rng.integers(0, 4, (T, 4)).astype(np.int32)
    rewards = rng.normal(-10, 3, T).astype(np.float32)
    terminals = np.zeros(T, dtype=bool)
    terminals[99]  = True
    terminals[199] = True
    terminals[299] = True
    terminals[399] = True
    terminals[499] = True

    # RTG descendente dentro de cada episodio
    rtg = np.zeros(T, dtype=np.float32)
    for ep_end in [99, 199, 299, 399, 499]:
        ep_start = ep_end - 99
        cumsum = 0.0
        for t in range(ep_end, ep_start - 1, -1):
            cumsum += rewards[t]
            rtg[t] = cumsum

    return {
        "observations":  obs,
        "actions":       actions,
        "rewards":       rewards,
        "terminals":     terminals,
        "returns_to_go": rtg,
    }
