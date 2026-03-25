# phase2/safety/action_mask.py
"""
Fase II – Escudo de Seguridad: Action Masking
Filtra acciones ilegales o inseguras antes de que el agente las ejecute,
garantizando cero violaciones de seguridad durante entrenamiento y despliegue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Estado de seguridad por intersección
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IntersectionSafetyState:
    """Rastrea el estado de seguridad de una intersección."""
    current_phase: int = 0
    phase_timer: int = 0          # pasos en la fase actual
    yellow_active: bool = False
    yellow_countdown: int = 0
    red_timers: Dict[int, int] = None  # fase → pasos en rojo

    def __post_init__(self):
        if self.red_timers is None:
            self.red_timers = {}


# ──────────────────────────────────────────────────────────────────────────────
# Action Masking
# ──────────────────────────────────────────────────────────────────────────────

class ActionMask:
    """
    Capa de enmascaramiento de acciones.

    Restringe las fases que el agente puede seleccionar en función de:
      1. Tiempo mínimo de verde (min_green): no puede abandonar la fase
         verde actual antes de cumplir el mínimo.
      2. Intervalo entre verdes (intergreen / IIG): la siguiente fase
         no puede activarse hasta que haya transcurrido el tiempo de
         amarillo/rojo mínimo (min_intergreen).
      3. Máximo tiempo en rojo (max_consecutive_red): prohíbe que un
         carril esté en rojo más de N pasos consecutivos.

    Parámetros
    ----------
    num_tls : int
    num_phases : int
    min_green : int        seg mínimos en verde antes de poder cambiar
    min_intergreen : int   seg mínimos de amarillo/transición
    max_consecutive_red : int  pasos máximos de rojo antes de forzar verde
    """

    def __init__(
        self,
        num_tls: int,
        num_phases: int,
        min_green: int = 5,
        min_intergreen: int = 3,
        max_consecutive_red: int = 120,
    ) -> None:
        self.num_tls = num_tls
        self.num_phases = num_phases
        self.min_green = min_green
        self.min_intergreen = min_intergreen
        self.max_consecutive_red = max_consecutive_red

        self._states: List[IntersectionSafetyState] = [
            IntersectionSafetyState(red_timers={p: 0 for p in range(num_phases)})
            for _ in range(num_tls)
        ]
        self._violation_count: int = 0

    # ── API pública ───────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reinicia el estado de seguridad al inicio de un episodio."""
        self._states = [
            IntersectionSafetyState(
                red_timers={p: 0 for p in range(self.num_phases)}
            )
            for _ in range(self.num_tls)
        ]
        self._violation_count = 0

    def get_mask(self, tl_idx: int) -> np.ndarray:
        """
        Retorna máscara booleana de forma (num_phases,).
        True = acción válida, False = acción prohibida.
        """
        state = self._states[tl_idx]
        mask = np.ones(self.num_phases, dtype=bool)

        if state.yellow_active:
            # Durante amarillo, solo se permite mantener la fase actual
            mask[:] = False
            mask[state.current_phase] = True
            return mask

        for phase in range(self.num_phases):
            if phase == state.current_phase:
                continue  # siempre válido mantener la fase actual

            # Restricción 1: min_green
            if state.phase_timer < self.min_green:
                mask[phase] = False
                continue

            # Restricción 2: min_intergreen
            red_time = state.red_timers.get(phase, 0)
            if red_time < self.min_intergreen:
                mask[phase] = False
                continue

        # Restricción 3: max_consecutive_red — forzar fase si lleva demasiado
        # tiempo en rojo (debe haber AL MENOS una fase válida siempre)
        for phase in range(self.num_phases):
            if phase != state.current_phase:
                red_t = state.red_timers.get(phase, 0)
                if red_t >= self.max_consecutive_red:
                    # Forzar esta fase (ignorar otras restricciones)
                    mask[:] = False
                    mask[phase] = True
                    return mask

        # Garantía: siempre al menos una acción válida
        if not mask.any():
            mask[state.current_phase] = True

        return mask

    def get_all_masks(self) -> np.ndarray:
        """Retorna máscaras para todos los TLs: (num_tls, num_phases)."""
        return np.stack([self.get_mask(i) for i in range(self.num_tls)])

    def get_mask_tensor(self, device: torch.device) -> torch.Tensor:
        """Retorna máscaras como tensor: (1, num_tls, num_phases)."""
        masks = self.get_all_masks()
        return torch.tensor(masks, dtype=torch.bool, device=device).unsqueeze(0)

    def update(self, actions: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Actualiza el estado interno tras aplicar `actions`.

        Parámetros
        ----------
        actions : (num_tls,) int — acciones DESPUÉS del enmascaramiento.

        Retorna
        -------
        safe_actions : (num_tls,) acciones validadas
        violations   : número de violaciones detectadas (debería ser 0)
        """
        violations = 0
        safe_actions = actions.copy()

        for i, (action, state) in enumerate(zip(actions, self._states)):
            mask = self.get_mask(i)

            if not mask[action]:
                # Violación detectada (no debería ocurrir si se usa get_mask)
                violations += 1
                self._violation_count += 1
                # Fallback: mantener fase actual
                safe_actions[i] = state.current_phase

            self._apply_action(i, int(safe_actions[i]))

        return safe_actions, violations

    def apply_to_logits(
        self,
        logits: torch.Tensor,   # (B, num_tls, num_phases)
        neg_inf: float = -1e9,
    ) -> torch.Tensor:
        """
        Aplica la máscara directamente a los logits del modelo
        (para integración con la capa de salida del DT).
        """
        masks = self.get_mask_tensor(logits.device)  # (1, num_tls, num_phases)
        masked = logits.clone()
        masked[~masks.expand_as(logits)] = neg_inf
        return masked

    @property
    def total_violations(self) -> int:
        return self._violation_count

    # ── Helpers privados ──────────────────────────────────────────────────────

    def _apply_action(self, tl_idx: int, action: int) -> None:
        state = self._states[tl_idx]

        if state.yellow_active:
            state.yellow_countdown -= 1
            if state.yellow_countdown <= 0:
                state.yellow_active = False
                state.current_phase = action
                state.phase_timer = 0
            return

        if action != state.current_phase:
            # Iniciar transición amarilla
            state.yellow_active = True
            state.yellow_countdown = self.min_intergreen
            # Actualizar timers de rojo
            for p in range(self.num_phases):
                if p == state.current_phase:
                    state.red_timers[p] = 0
                else:
                    state.red_timers[p] = state.red_timers.get(p, 0) + 1
            state.current_phase = action
            state.phase_timer = 0
        else:
            state.phase_timer += 1
            for p in range(self.num_phases):
                if p != state.current_phase:
                    state.red_timers[p] = state.red_timers.get(p, 0) + 1


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper de entorno con safety integrado
# ──────────────────────────────────────────────────────────────────────────────

class SafeEnvironmentWrapper:
    """
    Envuelve un entorno Gymnasium e intercepta las acciones
    con ActionMask antes de pasarlas al simulador.
    """

    def __init__(
        self,
        env,
        num_phases: int = 4,
        min_green: int = 5,
        min_intergreen: int = 3,
        max_consecutive_red: int = 120,
    ) -> None:
        self.env = env
        self.mask = ActionMask(
            num_tls=env._num_tls,
            num_phases=num_phases,
            min_green=min_green,
            min_intergreen=min_intergreen,
            max_consecutive_red=max_consecutive_red,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.mask.reset()
        return obs, info

    def step(self, action: np.ndarray):
        safe_action, violations = self.mask.update(action)
        if violations > 0:
            print(f"[WARN] {violations} violacion(es) de seguridad interceptada(s).")
        obs, reward, terminated, truncated, info = self.env.step(safe_action)
        info["safety_violations"] = violations
        info["total_violations"]  = self.mask.total_violations
        return obs, reward, terminated, truncated, info

    def get_current_masks(self) -> np.ndarray:
        return self.mask.get_all_masks()

    def __getattr__(self, name: str):
        return getattr(self.env, name)
