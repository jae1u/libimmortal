from __future__ import annotations

from typing import Dict

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from torch import nn


class TransformerDictExtractor(BaseFeaturesExtractor):
    """
    Transformer-based feature extractor for dict observations with image + vector.

    Vector tokenization:
      - player token: x_p in R^{13} (see VectorObservationPlayerIndex)
      - enemy tokens: x_e_i in R^{9} (see VectorObservationEnemyIndex), i=1..10
      - tokens are projected to d_model and encoded with self-attention

    Attention block computes:
      Q = X W_Q, K = X W_K, V = X W_V
      Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V

    Positional and entity-id embeddings are added to each token for stable ordering and
    identity conditioning. Enemy-type embeddings are inferred from the first three
    enemy features (skeleton/bombkid/turret), and use a "none" type when padded.
    If use_image=False, image features are ignored.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        image_features_dim: int = 256,
        vector_features_dim: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        player_dim: int | None = None,
        enemy_dim: int | None = None,
        num_enemies: int | None = None,
        use_image: bool = True,
    ) -> None:
        self.image_space = observation_space["image"]
        self.vector_space = observation_space["vector"]

        vector_len = int(self.vector_space.shape[0])
        if player_dim is None or enemy_dim is None or num_enemies is None:
            if vector_len == 103:
                player_dim = 13
                enemy_dim = 9
                num_enemies = 10
            elif vector_len == 99:
                player_dim = 9
                enemy_dim = 9
                num_enemies = 10
            else:
                raise ValueError(
                    "Vector observation size not recognized. Provide player_dim, "
                    "enemy_dim, and num_enemies explicitly."
                )

        vector_dim = player_dim + num_enemies * enemy_dim
        if int(self.vector_space.shape[0]) != vector_dim:
            raise ValueError(
                "Vector observation size mismatch: expected "
                f"{vector_dim}, got {self.vector_space.shape[0]}"
            )

        features_dim = (
            image_features_dim + vector_features_dim
            if use_image
            else vector_features_dim
        )
        super().__init__(observation_space, features_dim=features_dim)

        self.player_dim = player_dim
        self.enemy_dim = enemy_dim
        self.num_enemies = num_enemies
        self.d_model = d_model
        self.use_image = use_image

        if self.use_image:
            self.image_extractor = NatureCNN(
                self.image_space, features_dim=image_features_dim
            )
        else:
            self.image_extractor = None

        self.player_proj = nn.Linear(player_dim, d_model)
        self.enemy_proj = nn.Linear(enemy_dim, d_model)
        self.type_embed = nn.Embedding(2, d_model)
        self.pos_embed = nn.Embedding(1 + num_enemies, d_model)
        self.entity_id_embed = nn.Embedding(1 + num_enemies, d_model)
        self.enemy_type_embed = nn.Embedding(4, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.vector_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vector_features_dim),
            nn.GELU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.use_image and self.image_extractor is not None:
            image_features = self.image_extractor(observations["image"])
        vector = observations["vector"]

        player = vector[:, : self.player_dim].unsqueeze(1)
        enemies = vector[:, self.player_dim :].view(
            vector.shape[0], self.num_enemies, self.enemy_dim
        )

        player_tokens = self.player_proj(player)
        enemy_tokens = self.enemy_proj(enemies)

        enemy_types = self._infer_enemy_types(enemies)
        enemy_tokens = enemy_tokens + self.enemy_type_embed(enemy_types)

        tokens = torch.cat([player_tokens, enemy_tokens], dim=1)

        token_types = torch.zeros(
            tokens.shape[:2], dtype=torch.long, device=tokens.device
        )
        token_types[:, 1:] = 1
        positions = torch.arange(
            tokens.shape[1], device=tokens.device, dtype=torch.long
        ).unsqueeze(0)
        entity_ids = positions
        tokens = (
            tokens
            + self.type_embed(token_types)
            + self.pos_embed(positions)
            + self.entity_id_embed(entity_ids)
        )

        encoded = self.transformer(tokens)
        pooled = encoded.mean(dim=1)
        vector_features = self.vector_head(pooled)

        if self.use_image and self.image_extractor is not None:
            return torch.cat([image_features, vector_features], dim=1)
        return vector_features

    @staticmethod
    def _infer_enemy_types(enemies: torch.Tensor) -> torch.Tensor:
        enemy_type_logits = enemies[..., :3]
        max_vals, max_ids = enemy_type_logits.max(dim=-1)
        none_mask = max_vals <= 0
        type_ids = max_ids + 1
        type_ids = torch.where(none_mask, torch.zeros_like(type_ids), type_ids)
        return type_ids
