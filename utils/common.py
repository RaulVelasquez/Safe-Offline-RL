# utils/common.py
"""
Utilidades compartidas entre las tres fases del proyecto.
Incluye: logging estructurado, gestión de semillas, checkpoints,
normalización de observaciones y métricas de tráfico.
"""

from __future__ import annotations

import os
import json
import random
import logging
import hashlib
import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────────────

def get_logger(name: str, log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Retorna un logger con formato estructurado.
    Si `log_dir` se especifica, también escribe a archivo.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Ya configurado

    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler de consola
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Handler de archivo
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}_{timestamp}.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibilidad
# ──────────────────────────────────────────────────────────────────────────────

def set_global_seed(seed: int) -> None:
    """
    Fija la semilla en todos los generadores de números aleatorios
    relevantes para reproducibilidad del experimento.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Normalización de Observaciones (Running Mean/Std)
# ──────────────────────────────────────────────────────────────────────────────

class RunningNormalizer:
    """
    Calcula media y desviación estándar de forma incremental (online)
    usando el algoritmo de Welford.

    Uso:
        norm = RunningNormalizer(obs_dim)
        norm.update(obs_batch)       # alimentar con arrays (N, D)
        normalized = norm.normalize(obs)
    """

    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-8) -> None:
        self.shape = shape
        self.epsilon = epsilon
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = 0

    def update(self, batch: np.ndarray) -> None:
        """Actualiza estadísticas con un batch de observaciones (N, *shape)."""
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == len(self.shape):
            batch = batch[np.newaxis]  # añadir dim de batch

        n = batch.shape[0]
        batch_mean = batch.mean(axis=0)
        batch_var  = batch.var(axis=0)

        # Combinación de Welford
        total = self.count + n
        delta = batch_mean - self.mean
        self.mean  = self.mean + delta * n / total
        self.var   = (
            self.var * self.count
            + batch_var * n
            + delta ** 2 * self.count * n / total
        ) / total
        self.count = total

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.mean) / (np.sqrt(self.var) + self.epsilon)

    def denormalize(self, obs_norm: np.ndarray) -> np.ndarray:
        return obs_norm * np.sqrt(self.var) + self.mean

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, var=self.var, count=np.array([self.count]))

    @classmethod
    def load(cls, path: str, shape: Tuple[int, ...]) -> "RunningNormalizer":
        data = np.load(path + ".npz")
        norm = cls(shape)
        norm.mean  = data["mean"]
        norm.var   = data["var"]
        norm.count = int(data["count"][0])
        return norm


# ──────────────────────────────────────────────────────────────────────────────
# Gestión de Checkpoints
# ──────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Administra el guardado y carga de checkpoints con historial.

    Mantiene los últimos `keep_n` checkpoints y siempre preserva
    el checkpoint con mejor métrica.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_n: int = 5,
        metric_name: str = "val_loss",
        mode: str = "min",   # "min" | "max"
    ) -> None:
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_n = keep_n
        self.metric_name = metric_name
        self.mode = mode
        self._history_file = self.dir / "checkpoint_history.json"
        self._history: list = self._load_history()
        self._best_value = float("inf") if mode == "min" else float("-inf")

    def save(
        self,
        state: Dict[str, Any],
        step: int,
        metric_value: Optional[float] = None,
        tag: str = "ckpt",
    ) -> Path:
        """
        Guarda un checkpoint. Retorna la ruta del archivo guardado.
        Si `metric_value` es el mejor hasta ahora, también guarda como 'best'.
        """
        fname = self.dir / f"{tag}_step{step:07d}.pt"

        try:
            import torch
            torch.save(state, fname)
        except ImportError:
            # Fallback a pickle si torch no está disponible
            import pickle
            with open(fname, "wb") as f:
                pickle.dump(state, f)

        record = {
            "path": str(fname),
            "step": step,
            "metric": metric_value,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self._history.append(record)
        self._save_history()
        self._cleanup_old()

        # Guardar mejor
        if metric_value is not None and self._is_better(metric_value):
            self._best_value = metric_value
            best_path = self.dir / "best_model.pt"
            import shutil
            shutil.copy2(fname, best_path)

        return fname

    def load_best(self) -> Optional[Dict]:
        best_path = self.dir / "best_model.pt"
        if not best_path.exists():
            return None
        try:
            import torch
            return torch.load(best_path, map_location="cpu")
        except Exception:
            return None

    def load_latest(self) -> Optional[Dict]:
        if not self._history:
            return None
        latest = self._history[-1]["path"]
        try:
            import torch
            return torch.load(latest, map_location="cpu")
        except Exception:
            return None

    def _is_better(self, value: float) -> bool:
        if self.mode == "min":
            return value < self._best_value
        return value > self._best_value

    def _cleanup_old(self) -> None:
        """Elimina checkpoints viejos manteniendo solo los últimos `keep_n`."""
        if len(self._history) > self.keep_n:
            to_remove = self._history[:-self.keep_n]
            for record in to_remove:
                p = Path(record["path"])
                if p.exists() and "best" not in p.name:
                    p.unlink(missing_ok=True)
            self._history = self._history[-self.keep_n:]
            self._save_history()

    def _load_history(self) -> list:
        if self._history_file.exists():
            with open(self._history_file) as f:
                return json.load(f)
        return []

    def _save_history(self) -> None:
        with open(self._history_file, "w") as f:
            json.dump(self._history, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Métricas de tráfico (cálculos puros, sin TraCI)
# ──────────────────────────────────────────────────────────────────────────────

def compute_avg_travel_time(
    departure_times: np.ndarray,
    arrival_times: np.ndarray,
) -> float:
    """
    Calcula el Tiempo Promedio de Viaje (ATT) para un conjunto de vehículos.

    Parámetros
    ----------
    departure_times : (N,) tiempo de salida en segundos
    arrival_times   : (N,) tiempo de llegada en segundos

    Retorna
    -------
    ATT en segundos
    """
    travel_times = arrival_times - departure_times
    travel_times = travel_times[travel_times > 0]   # filtrar inválidos
    return float(np.mean(travel_times)) if len(travel_times) > 0 else 0.0


def compute_queue_stats(queue_lengths: np.ndarray) -> Dict[str, float]:
    """
    Calcula estadísticas de longitud de cola.

    Parámetros
    ----------
    queue_lengths : (T, N_lanes) mediciones a lo largo del tiempo

    Retorna
    -------
    dict con mean, max, p95
    """
    flat = queue_lengths.flatten()
    return {
        "mean": float(np.mean(flat)),
        "max":  float(np.max(flat)),
        "p95":  float(np.percentile(flat, 95)),
        "std":  float(np.std(flat)),
    }


def compute_throughput(
    arrived: int,
    departed: int,
    duration_seconds: float,
) -> Dict[str, float]:
    """
    Calcula throughput y tasa de servicio del sistema.
    """
    return {
        "total_arrived":       arrived,
        "total_departed":      departed,
        "vehicles_per_hour":   arrived / max(duration_seconds / 3600, 1e-6),
        "service_rate":        arrived / max(departed, 1),
    }


def compute_delay(
    actual_travel_time: float,
    free_flow_travel_time: float,
) -> float:
    """Retraso vehicular = ATT - tiempo de flujo libre."""
    return max(actual_travel_time - free_flow_travel_time, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Decorador de tiempo de ejecución
# ──────────────────────────────────────────────────────────────────────────────

def timed(func):
    """Decorador que registra el tiempo de ejecución de una función."""
    import functools
    import time

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  ⏱ {func.__name__} completado en {elapsed:.2f}s")
        return result
    return wrapper


# ──────────────────────────────────────────────────────────────────────────────
# Generador de ID único de experimento
# ──────────────────────────────────────────────────────────────────────────────

def experiment_id(config: dict) -> str:
    """
    Genera un ID corto y reproducible para un experimento dado su configuración.
    Útil para nombrar carpetas de resultados.
    """
    config_str = json.dumps(config, sort_keys=True)
    h = hashlib.md5(config_str.encode()).hexdigest()[:8]
    ts = datetime.datetime.now().strftime("%m%d_%H%M")
    return f"exp_{ts}_{h}"
