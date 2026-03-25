# phase1/models/decision_transformer.py
"""
Fase I – Modelado de Secuencias
Arquitectura Decision Transformer (DT) para control de señales de tráfico.
Referencia: Chen et al. (2021) "Decision Transformer: Reinforcement Learning
            via Sequence Modeling".
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Componentes base del Transformer
# ──────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Multi-head attention causal (no atiende a pasos futuros)."""

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_head == 0, "d_model debe ser divisible por n_head"
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, H, T, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) / scale    # (B, H, T, T)

        # Máscara causal
        causal = torch.tril(torch.ones(T, T, device=x.device)).bool()
        attn = attn.masked_fill(~causal, float("-inf"))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.proj(out))


class TransformerBlock(nn.Module):
    """Bloque Transformer con pre-LayerNorm."""

    def __init__(self, d_model: int, n_head: int, d_inner: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Decision Transformer
# ──────────────────────────────────────────────────────────────────────────────

class DecisionTransformer(nn.Module):
    """
    Decision Transformer para control adaptativo de señales de tráfico.

    El DT modela la distribución condicional:
        a_t ~ π(· | R̂_t, s_t, a_{t-1}, …, R̂_{t-K+1}, s_{t-K+1})

    donde R̂_t es el return-to-go deseado.

    Parámetros
    ----------
    obs_dim : int
        Dimensión del vector de observación.
    act_dim : int
        Número total de acciones posibles (MultiDiscrete → suma de dims).
    num_tls : int
        Número de intersecciones controladas.
    context_length : int
        Número de timesteps de contexto (K).
    d_model : int
        Dimensión interna del transformer.
    n_layer : int
        Número de bloques transformer.
    n_head : int
        Cabezas de atención.
    d_inner : int
        Dimensión de la FFN interna.
    dropout : float
        Tasa de dropout.
    max_ep_len : int
        Longitud máxima de episodio (para embedding posicional).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_tls: int,
        context_length: int = 20,
        d_model: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        d_inner: int = 512,
        dropout: float = 0.1,
        max_ep_len: int = 3600,
    ) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_tls = num_tls
        self.context_length = context_length
        self.d_model = d_model

        # ── Embeddings ────────────────────────────────────────────────────────
        # Cada timestep aporta 3 tokens: (R̂, s, a)
        self.embed_timestep = nn.Embedding(max_ep_len, d_model)
        self.embed_return   = nn.Linear(1, d_model)
        self.embed_state    = nn.Linear(obs_dim, d_model)
        self.embed_action   = nn.Embedding(act_dim, d_model)  # acción global

        self.embed_ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        # ── Backbone Transformer ──────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_inner, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # ── Cabeza de predicción de acción por TL ─────────────────────────────
        # Predicción independiente por intersección
        phases_per_tl = act_dim // num_tls if num_tls > 0 else act_dim
        self.action_heads = nn.ModuleList([
            nn.Linear(d_model, phases_per_tl)
            for _ in range(num_tls)
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        states: torch.Tensor,       # (B, T, obs_dim)
        actions: torch.Tensor,      # (B, T, num_tls)  — int
        returns_to_go: torch.Tensor,# (B, T, 1)
        timesteps: torch.Tensor,    # (B, T)            — int
    ) -> Tuple[torch.Tensor, ...]:
        """
        Retorna
        -------
        action_preds : tuple de (B, T, phases_per_tl) por cada TL
        """
        B, T, _ = states.shape

        # ── Embeddings ────────────────────────────────────────────────────────
        t_emb   = self.embed_timestep(timesteps)                 # (B,T,d)
        r_emb   = self.embed_return(returns_to_go.float())       # (B,T,d)
        s_emb   = self.embed_state(states.float())               # (B,T,d)

        # Acción: suma de embeddings por TL (estrategia de fusión simple)
        a_emb = torch.zeros(B, T, self.d_model, device=states.device)
        for i in range(self.num_tls):
            a_emb += self.embed_action(actions[..., i])
        a_emb = a_emb / self.num_tls

        # Posicional: misma embedding para los 3 tokens del mismo timestep
        r_emb = r_emb + t_emb
        s_emb = s_emb + t_emb
        a_emb = a_emb + t_emb

        # Interleave: [R̂_1, s_1, a_1, R̂_2, s_2, a_2, …]
        # Shape → (B, 3*T, d_model)
        tokens = torch.stack([r_emb, s_emb, a_emb], dim=2)   # (B,T,3,d)
        tokens = tokens.reshape(B, 3 * T, self.d_model)
        tokens = self.drop(self.embed_ln(tokens))

        # ── Transformer ───────────────────────────────────────────────────────
        x = tokens
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)                                       # (B, 3*T, d)

        # Extraer representaciones de estado (posición 1, 4, 7, …)
        # índices impares en el tripleto: posición 1 de cada (R,s,a)
        state_repr = x[:, 1::3, :]                             # (B, T, d)

        # ── Predicciones de acción ────────────────────────────────────────────
        action_preds = tuple(
            head(state_repr) for head in self.action_heads
        )   # num_tls × (B, T, phases_per_tl)

        return action_preds

    # ── Inferencia greedy ─────────────────────────────────────────────────────

    @torch.no_grad()
    def get_action(
        self,
        states: torch.Tensor,       # (1, T, obs_dim)
        actions: torch.Tensor,      # (1, T, num_tls)
        returns_to_go: torch.Tensor,# (1, T, 1)
        timesteps: torch.Tensor,    # (1, T)
        action_mask: Optional[torch.Tensor] = None,  # (1, num_tls, phases)
    ) -> torch.Tensor:              # (num_tls,)
        """
        Devuelve la acción para el último timestep.
        action_mask: máscara booleana de acciones válidas (True = válida).
        """
        action_preds = self.forward(states, actions, returns_to_go, timesteps)
        # Solo último timestep
        acts = []
        for tl_idx, logits in enumerate(action_preds):
            last_logits = logits[0, -1, :]          # (phases_per_tl,)
            if action_mask is not None:
                mask = action_mask[0, tl_idx, :]    # (phases_per_tl,)
                last_logits = last_logits.masked_fill(~mask, float("-inf"))
            acts.append(last_logits.argmax())
        return torch.stack(acts)                     # (num_tls,)
