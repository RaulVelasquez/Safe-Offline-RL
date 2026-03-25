# phase3/online/online_finetuner.py
"""
Fase III – Ajuste Fino Online
Transiciona el modelo pre-entrenado offline a simulación en vivo,
actualizando pesos de forma ligera para adaptarse a demanda dinámica.
"""

from __future__ import annotations

import os
import collections
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from phase1.models.decision_transformer import DecisionTransformer
from phase2.safety.action_mask import SafeEnvironmentWrapper


# ──────────────────────────────────────────────────────────────────────────────
# Buffer de experiencia reciente (replay de ventana deslizante)
# ──────────────────────────────────────────────────────────────────────────────

class OnlineReplayBuffer:
    """
    Buffer FIFO de capacidad fija para almacenar transiciones online recientes.
    """

    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = capacity
        self._obs:    Deque[np.ndarray] = collections.deque(maxlen=capacity)
        self._act:    Deque[np.ndarray] = collections.deque(maxlen=capacity)
        self._rew:    Deque[float]      = collections.deque(maxlen=capacity)
        self._rtg:    Deque[float]      = collections.deque(maxlen=capacity)
        self._t:      Deque[int]        = collections.deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        rtg: float,
        timestep: int,
    ) -> None:
        self._obs.append(obs)
        self._act.append(action)
        self._rew.append(reward)
        self._rtg.append(rtg)
        self._t.append(timestep)

    def sample(self, batch_size: int, context_length: int) -> Optional[dict]:
        """Muestra aleatoriamente ventanas de longitud `context_length`."""
        N = len(self._obs)
        if N < context_length + 1:
            return None

        max_start = N - context_length
        starts = np.random.randint(0, max_start, size=batch_size)

        obs_arr = np.array(list(self._obs))
        act_arr = np.array(list(self._act))
        rtg_arr = np.array(list(self._rtg))

        batch_obs, batch_act, batch_rtg, batch_t = [], [], [], []
        for s in starts:
            batch_obs.append(obs_arr[s: s + context_length])
            batch_act.append(act_arr[s: s + context_length])
            batch_rtg.append(rtg_arr[s: s + context_length, None])
            # Timesteps relativos (0..K-1) para no superar max_ep_len del embedding
            batch_t.append(np.arange(context_length))

        return {
            "observations": torch.tensor(np.array(batch_obs), dtype=torch.float32),
            "actions":      torch.tensor(np.array(batch_act), dtype=torch.long),
            "rtg":          torch.tensor(np.array(batch_rtg), dtype=torch.float32),
            "timesteps":    torch.tensor(np.array(batch_t),   dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self._obs)


# ──────────────────────────────────────────────────────────────────────────────
# Online Fine-Tuner
# ──────────────────────────────────────────────────────────────────────────────

class OnlineFineTuner:
    """
    Ajuste fino del Decision Transformer usando interacción en tiempo real.

    Estrategia: sólo se actualizan los últimos `trainable_layers` bloques
    del transformer (fine-tuning ligero); el embedding y capas inferiores
    permanecen congelados para preservar el conocimiento offline.

    Parámetros
    ----------
    model : DecisionTransformer — modelo pre-entrenado
    env   : SafeEnvironmentWrapper — entorno con safety integrado
    target_return : float   RTG objetivo (negativo = menor delay)
    context_length : int    Ventana de contexto del DT
    trainable_layers : int  Últimas N capas a descongelar
    lr : float              Learning rate de fine-tuning (pequeño)
    update_freq : int       Pasos entre actualizaciones de pesos
    buffer_capacity : int   Capacidad del replay buffer online
    checkpoint_dir : str
    device : str
    """

    def __init__(
        self,
        model: DecisionTransformer,
        env: SafeEnvironmentWrapper,
        target_return: float = -50.0,
        context_length: int = 20,
        trainable_layers: int = 2,
        lr: float = 5e-5,
        update_freq: int = 10,
        buffer_capacity: int = 10_000,
        checkpoint_dir: str = "checkpoints/online",
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.env = env
        self.target_return = target_return
        self.context_length = context_length
        self.update_freq = update_freq
        self.checkpoint_dir = checkpoint_dir

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.model.to(self.device)

        # Congelar capas de embedding y primeras capas del transformer
        self._freeze_base_layers(trainable_layers)

        # Solo optimizar parámetros descongelados
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr)

        self.buffer = OnlineReplayBuffer(capacity=buffer_capacity)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, "tb"))

        self._episode_count = 0
        self._global_step = 0

    # ── API pública ───────────────────────────────────────────────────────────

    def run(self, num_episodes: int, batch_size: int = 32, start_episode: int = 1) -> None:
        action = "Reanudando" if start_episode > 1 else "Iniciando"
        print(f"\n{action} fine-tuning online ({num_episodes} episodios, "
              f"desde ep {start_episode})...")
        print(f"  Parametros entrenables: "
              f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        for ep in range(start_episode, num_episodes + 1):
            ep_reward, ep_violations, ep_steps = self._run_episode(batch_size)
            self._episode_count += 1

            self.writer.add_scalar("Online/episode_reward",     ep_reward,     ep)
            self.writer.add_scalar("Online/safety_violations",  ep_violations, ep)
            self.writer.add_scalar("Online/episode_steps",      ep_steps,      ep)

            if ep % 10 == 0:
                print(
                    f"  Ep {ep:4d}/{num_episodes} | "
                    f"reward: {ep_reward:8.1f} | "
                    f"violations: {ep_violations:3d} | "
                    f"steps: {ep_steps}"
                )
                self._save_checkpoint(f"ep_{ep:04d}.pt")

        self.writer.close()
        print("Fine-tuning online completado.")

    # ── Episodio ──────────────────────────────────────────────────────────────

    def _run_episode(self, batch_size: int) -> Tuple[float, int, int]:
        obs, _ = self.env.reset()

        # Buffers de contexto para el DT
        ctx_obs  = collections.deque(maxlen=self.context_length)
        ctx_act  = collections.deque(maxlen=self.context_length)
        ctx_rtg  = collections.deque(maxlen=self.context_length)
        ctx_t    = collections.deque(maxlen=self.context_length)

        num_tls  = self.model.num_tls
        obs_dim  = self.model.obs_dim
        phases   = self.model.act_dim // num_tls

        # Acción nula para el primer paso
        dummy_action = np.zeros(num_tls, dtype=np.int64)

        ep_reward    = 0.0
        ep_violations = 0
        ep_steps     = 0
        running_rtg  = self.target_return

        done = False
        while not done:
            ctx_obs.append(obs.copy())
            ctx_act.append(dummy_action.copy())
            ctx_rtg.append(running_rtg)
            ctx_t.append(ep_steps)

            # Construir tensores de contexto
            t_obs = torch.tensor(
                np.array(ctx_obs), dtype=torch.float32
            ).unsqueeze(0).to(self.device)              # (1,T,obs_dim)
            t_act = torch.tensor(
                np.array(ctx_act), dtype=torch.long
            ).unsqueeze(0).to(self.device)              # (1,T,num_tls)
            t_rtg = torch.tensor(
                np.array(ctx_rtg)[:, None], dtype=torch.float32
            ).unsqueeze(0).to(self.device)              # (1,T,1)
            t_ts  = torch.tensor(
                np.array(ctx_t), dtype=torch.long
            ).unsqueeze(0).to(self.device)              # (1,T)

            # Obtener máscara de seguridad
            mask = self.env.get_current_masks()
            mask_tensor = torch.tensor(
                mask, dtype=torch.bool, device=self.device
            ).unsqueeze(0)                              # (1,num_tls,phases)

            # Inferencia
            action = self.model.get_action(
                t_obs, t_act, t_rtg, t_ts, action_mask=mask_tensor
            ).cpu().numpy()

            obs_next, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            ep_reward    += reward
            ep_violations += info.get("safety_violations", 0)
            ep_steps     += 1

            running_rtg  = max(running_rtg - reward, 0)

            # Almacenar en buffer online
            self.buffer.push(obs, action, reward, running_rtg, ep_steps)

            dummy_action = action
            obs = obs_next
            self._global_step += 1

            # Actualización de pesos periódica
            if (
                self._global_step % self.update_freq == 0
                and len(self.buffer) >= self.context_length
            ):
                loss = self._update_step(batch_size)
                if loss is not None:
                    self.writer.add_scalar("Online/loss", loss, self._global_step)

        return ep_reward, ep_violations, ep_steps

    def _update_step(self, batch_size: int) -> Optional[float]:
        batch = self.buffer.sample(batch_size, self.context_length)
        if batch is None:
            return None

        self.model.train()
        states    = batch["observations"].to(self.device)
        actions   = batch["actions"].to(self.device)
        rtg       = batch["rtg"].to(self.device)
        timesteps = batch["timesteps"].to(self.device)

        preds = self.model(states, actions, rtg, timesteps)
        # preds: tuple de (B, T, phases_per_tl)

        total_loss = torch.tensor(0.0, device=self.device)
        for tl_idx, pred in enumerate(preds):
            B, T, P = pred.shape
            targets = actions[..., tl_idx]
            total_loss += nn.functional.cross_entropy(
                pred.reshape(B * T, P), targets.reshape(B * T)
            )
        total_loss = total_loss / len(preds)

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 0.5
        )
        self.optimizer.step()
        self.model.eval()

        return total_loss.item()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _freeze_base_layers(self, trainable_layers: int) -> None:
        """Congela todo excepto los últimos `trainable_layers` bloques y cabezas."""
        # Congelar embeddings
        for param in self.model.embed_timestep.parameters():
            param.requires_grad = False
        for param in self.model.embed_return.parameters():
            param.requires_grad = False
        for param in self.model.embed_state.parameters():
            param.requires_grad = False
        for param in self.model.embed_action.parameters():
            param.requires_grad = False

        # Congelar bloques menos los últimos N
        n_blocks = len(self.model.blocks)
        for i, block in enumerate(self.model.blocks):
            requires_grad = i >= (n_blocks - trainable_layers)
            for param in block.parameters():
                param.requires_grad = requires_grad

        # Siempre descongelar cabezas de acción
        for head in self.model.action_heads:
            for param in head.parameters():
                param.requires_grad = True

    def _save_checkpoint(self, name: str) -> None:
        path = os.path.join(self.checkpoint_dir, name)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "episode": self._episode_count,
            },
            path,
        )
