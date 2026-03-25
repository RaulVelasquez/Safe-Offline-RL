# phase1/train_offline.py
"""
Fase I – Script de Entrenamiento Offline
Pre-entrena el Decision Transformer sobre el corpus de trayectorias históricas.
"""

from __future__ import annotations

import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from phase1.models.decision_transformer import DecisionTransformer
from phase1.data.dataset_generator import load_offline_dataset


# ──────────────────────────────────────────────────────────────────────────────
# Dataset PyTorch
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    """
    Dataset de secuencias de longitud K para el Decision Transformer.
    Cada muestra es una ventana deslizante del corpus offline.
    """

    def __init__(
        self,
        data: dict,
        context_length: int = 20,
        num_tls: int = 4,
    ) -> None:
        self.obs        = data["observations"]     # (N, obs_dim)
        self.actions    = data["actions"]           # (N, num_tls)
        self.rtg        = data["returns_to_go"]    # (N,)
        self.terminals  = data["terminals"]        # (N,)
        self.K          = context_length
        self.num_tls    = num_tls

        # Construir índices de inicio de ventanas (no cruzan terminales)
        self._indices = self._build_indices()

    def _build_indices(self):
        indices = []
        N = len(self.terminals)
        i = 0
        while i < N - self.K:
            # Si hay un terminal dentro de la ventana, saltar
            window_end = i + self.K
            term_in_window = self.terminals[i:window_end].any()
            if not term_in_window:
                indices.append(i)
                i += 1
            else:
                # Saltar al siguiente episodio
                term_pos = np.argmax(self.terminals[i:window_end]) + i
                i = term_pos + 1
        return indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        start = self._indices[idx]
        end   = start + self.K

        states   = torch.tensor(self.obs[start:end], dtype=torch.float32)
        actions  = torch.tensor(self.actions[start:end], dtype=torch.long)
        rtg      = torch.tensor(self.rtg[start:end, None], dtype=torch.float32)
        # Timesteps relativos al inicio de la ventana (0…K-1)
        # Los índices absolutos pueden superar max_ep_len del embedding.
        timesteps= torch.arange(0, self.K, dtype=torch.long)

        return states, actions, rtg, timesteps


# ──────────────────────────────────────────────────────────────────────────────
# Funciones de entrenamiento y evaluación
# ──────────────────────────────────────────────────────────────────────────────

def compute_loss(
    model: DecisionTransformer,
    states: torch.Tensor,
    actions: torch.Tensor,
    rtg: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy entre las acciones predichas y las acciones del dataset.
    Predicción: a_t dado (R̂_t, s_t, …, a_{t-1}).
    """
    # Desplazar acciones: predecimos a_t usando contexto hasta a_{t-1}
    # DT predice acción en la posición de estado
    action_preds = model(states, actions, rtg, timesteps)
    # action_preds: num_tls × (B, T, phases)
    # actions: (B, T, num_tls)

    total_loss = torch.tensor(0.0, device=states.device)
    for tl_idx, preds in enumerate(action_preds):
        # preds: (B, T, phases) — flatten para cross-entropy
        B, T, P = preds.shape
        targets = actions[..., tl_idx]  # (B, T)
        loss = nn.functional.cross_entropy(
            preds.reshape(B * T, P),
            targets.reshape(B * T),
        )
        total_loss += loss

    return total_loss / len(action_preds)


def evaluate(
    model: DecisionTransformer,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for states, actions, rtg, timesteps in loader:
            states    = states.to(device)
            actions   = actions.to(device)
            rtg       = rtg.to(device)
            timesteps = timesteps.to(device)
            loss = compute_loss(model, states, actions, rtg, timesteps)
            total_loss += loss.item()
    model.train()
    return total_loss / len(loader)


# ──────────────────────────────────────────────────────────────────────────────
# Entrenador principal
# ──────────────────────────────────────────────────────────────────────────────

class OfflineTrainer:

    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg = cfg
        self._set_seed(cfg.seed)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
        )
        print(f"Dispositivo: {self.device}")

        # ── Dataset ───────────────────────────────────────────────────────────
        print("Cargando dataset offline…")
        data = load_offline_dataset(cfg.dataset_dir)
        n_total = len(data["observations"])
        n_val = int(n_total * 0.1)

        train_data = {k: v[:-n_val] for k, v in data.items()}
        val_data   = {k: v[-n_val:] for k, v in data.items()}

        train_ds = TrajectoryDataset(train_data, cfg.context_length, cfg.num_tls)
        val_ds   = TrajectoryDataset(val_data,   cfg.context_length, cfg.num_tls)

        # num_workers=0 en Windows: evita que cada proceso hijo recargue
        # el dataset completo (3.58M entradas) causando bloqueo en el inicio.
        self.train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
        )

        # ── Modelo ────────────────────────────────────────────────────────────
        self.model = DecisionTransformer(
            obs_dim       = data["observations"].shape[1],
            act_dim       = cfg.num_phases * cfg.num_tls,
            num_tls       = cfg.num_tls,
            context_length= cfg.context_length,
            d_model       = cfg.d_model,
            n_layer       = cfg.n_layer,
            n_head        = cfg.n_head,
            d_inner       = cfg.d_inner,
            dropout       = cfg.dropout,
            max_ep_len    = cfg.max_ep_len,
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Parámetros del modelo: {total_params:,}")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.num_epochs
        )

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.checkpoint_dir, "tb"))
        self.best_val_loss = float("inf")

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def train(self) -> None:
        cfg = self.cfg
        # steps_per_epoch=0 significa usar el dataset completo
        spe = getattr(cfg, "steps_per_epoch", 0)
        effective = spe if spe > 0 else len(self.train_loader)
        print(f"\nIniciando entrenamiento offline ({cfg.num_epochs} epocas, "
              f"{effective} pasos/epoca)...")

        train_iter = iter(self.train_loader)

        for epoch in range(1, cfg.num_epochs + 1):
            self.model.train()
            running_loss = 0.0
            steps_done = 0

            pbar = tqdm(range(effective), desc=f"Epoch {epoch}/{cfg.num_epochs}")
            for _ in pbar:
                try:
                    states, actions, rtg, timesteps = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    states, actions, rtg, timesteps = next(train_iter)

                states    = states.to(self.device)
                actions   = actions.to(self.device)
                rtg       = rtg.to(self.device)
                timesteps = timesteps.to(self.device)

                self.optimizer.zero_grad()
                loss = compute_loss(self.model, states, actions, rtg, timesteps)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.grad_clip
                )
                self.optimizer.step()

                running_loss += loss.item()
                steps_done += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_train = running_loss / steps_done
            self.scheduler.step()

            self.writer.add_scalar("Loss/train", avg_train, epoch)
            self.writer.add_scalar(
                "LR", self.optimizer.param_groups[0]["lr"], epoch
            )

            # Evaluación periódica
            if epoch % cfg.eval_interval == 0:
                val_loss = evaluate(self.model, self.val_loader, self.device)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                print(
                    f"  -> Epoch {epoch:4d} | train: {avg_train:.4f} | "
                    f"val: {val_loss:.4f}"
                )
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best_model.pt")
                    print(f"  [OK] Nuevo mejor modelo guardado (val={val_loss:.4f})")

        self._save_checkpoint("final_model.pt")
        self.writer.close()
        print("\nEntrenamiento completado.")

    def _save_checkpoint(self, name: str) -> None:
        path = os.path.join(self.cfg.checkpoint_dir, name)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "cfg": vars(self.cfg),
                "best_val_loss": self.best_val_loss,
            },
            path,
        )

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenamiento Offline – Decision Transformer")
    p.add_argument("--dataset_dir",    default="data/offline_dataset")
    p.add_argument("--checkpoint_dir", default="checkpoints/offline")
    p.add_argument("--num_tls",        type=int, default=4)
    p.add_argument("--num_phases",     type=int, default=4)
    p.add_argument("--context_length", type=int, default=20)
    p.add_argument("--d_model",        type=int, default=128)
    p.add_argument("--n_layer",        type=int, default=4)
    p.add_argument("--n_head",         type=int, default=4)
    p.add_argument("--d_inner",        type=int, default=512)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--max_ep_len",     type=int, default=3600)
    p.add_argument("--batch_size",     type=int, default=64)
    p.add_argument("--num_epochs",     type=int, default=100)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--eval_interval",  type=int, default=10)
    p.add_argument("--steps_per_epoch",type=int, default=500,
                   help="Pasos de gradiente por epoca (0=dataset completo, ~713h en CPU)")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--device",         default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = OfflineTrainer(args)
    trainer.train()
