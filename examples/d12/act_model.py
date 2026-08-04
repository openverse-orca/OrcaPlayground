import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def act_lite_loss(pred_actions, target_actions, mu, logvar, kl_weight=10.0):
    recon_loss = F.l1_loss(pred_actions, target_actions)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss


def act_vision_loss(pred_actions, target_actions, mu, logvar, kl_weight=10.0):
    recon_loss = F.l1_loss(pred_actions, target_actions)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss


def act_deterministic_loss(pred_actions, target_actions, **kwargs):
    return F.l1_loss(pred_actions, target_actions), F.l1_loss(pred_actions, target_actions), torch.tensor(0.0)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ACTLite(nn.Module):
    def __init__(
        self,
        state_dim: int = 14,
        action_dim: int = 30,
        chunk_size: int = 20,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.latent_dim = latent_dim

        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.cvae_encoder = nn.Sequential(
            nn.Linear(action_dim * chunk_size + state_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.cvae_mu = nn.Linear(d_model, latent_dim)
        self.cvae_logvar = nn.Linear(d_model, latent_dim)
        self.latent_proj = nn.Linear(latent_dim, d_model)

        self.pos_encoding = SinusoidalPositionEncoding(d_model, max_len=chunk_size + 2)
        self.action_query = nn.Embedding(chunk_size, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim),
        )

    def encode_cvae(self, state: torch.Tensor, action_chunk: torch.Tensor):
        B = state.size(0)
        action_flat = action_chunk.reshape(B, -1)
        cvae_input = torch.cat([state, action_flat], dim=-1)
        h = self.cvae_encoder(cvae_input)
        mu = self.cvae_mu(h)
        logvar = self.cvae_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B = state.size(0)
        z_emb = self.latent_proj(z).unsqueeze(1)
        state_emb = self.state_proj(state).unsqueeze(1)
        memory = torch.cat([z_emb, state_emb], dim=1)

        query_indices = torch.arange(self.chunk_size, device=state.device)
        queries = self.action_query(query_indices).unsqueeze(0).expand(B, -1, -1)
        queries = self.pos_encoding(queries)

        decoded = self.transformer_decoder(queries, memory)
        actions = self.action_head(decoded)
        return actions

    def forward(self, state: torch.Tensor, action_chunk: torch.Tensor = None):
        if action_chunk is not None:
            mu, logvar = self.encode_cvae(state, action_chunk)
            z = self.reparameterize(mu, logvar)
            pred_actions = self.decode(state, z)
            return pred_actions, mu, logvar
        else:
            B = state.size(0)
            z = torch.zeros(B, self.latent_dim, device=state.device)
            pred_actions = self.decode(state, z)
            return pred_actions, None, None


class LightweightVisionEncoder(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(256, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.pool(feat).flatten(1)
        return self.proj(feat)


class ACTLiteVision(nn.Module):
    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 30,
        num_cameras: int = 1,
        img_size: int = 128,
        vision_backbone: str = "lightweight",
        chunk_size: int = 20,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_cameras = num_cameras
        self.img_size = img_size
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.latent_dim = latent_dim

        self.vision_encoder = LightweightVisionEncoder(d_model=d_model)

        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        cvae_input_dim = action_dim * chunk_size + state_dim + d_model * num_cameras
        self.cvae_encoder = nn.Sequential(
            nn.Linear(cvae_input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.cvae_mu = nn.Linear(d_model, latent_dim)
        self.cvae_logvar = nn.Linear(d_model, latent_dim)
        self.latent_proj = nn.Linear(latent_dim, d_model)

        self.pos_encoding = SinusoidalPositionEncoding(d_model, max_len=chunk_size + 2)
        self.action_query = nn.Embedding(chunk_size, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim),
        )

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = images.shape
        imgs_flat = images.reshape(B * N, C, H, W)
        feats = self.vision_encoder(imgs_flat)
        return feats.reshape(B, N, self.d_model)

    def encode_cvae(self, state: torch.Tensor, action_chunk: torch.Tensor, vision_feats: torch.Tensor):
        B = state.size(0)
        action_flat = action_chunk.reshape(B, -1)
        vision_flat = vision_feats.reshape(B, -1)
        cvae_input = torch.cat([state, action_flat, vision_flat], dim=-1)
        h = self.cvae_encoder(cvae_input)
        mu = self.cvae_mu(h)
        logvar = self.cvae_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, state: torch.Tensor, z: torch.Tensor, vision_feats: torch.Tensor) -> torch.Tensor:
        B = state.size(0)
        z_emb = self.latent_proj(z).unsqueeze(1)
        state_emb = self.state_proj(state).unsqueeze(1)
        memory = torch.cat([z_emb, state_emb, vision_feats], dim=1)

        query_indices = torch.arange(self.chunk_size, device=state.device)
        queries = self.action_query(query_indices).unsqueeze(0).expand(B, -1, -1)
        queries = self.pos_encoding(queries)

        decoded = self.transformer_decoder(queries, memory)
        actions = self.action_head(decoded)
        return actions

    def forward(self, state: torch.Tensor, images: torch.Tensor = None, action_chunk: torch.Tensor = None):
        if images is not None:
            vision_feats = self.encode_vision(images)
        else:
            vision_feats = torch.zeros(state.size(0), self.num_cameras, self.d_model, device=state.device)

        if action_chunk is not None:
            mu, logvar = self.encode_cvae(state, action_chunk, vision_feats)
            z = self.reparameterize(mu, logvar)
            pred_actions = self.decode(state, z, vision_feats)
            return pred_actions, mu, logvar
        else:
            B = state.size(0)
            z = torch.zeros(B, self.latent_dim, device=state.device)
            pred_actions = self.decode(state, z, vision_feats)
            return pred_actions, None, None


class ACTDet(nn.Module):
    """确定性 Action Chunking Transformer — 无 CVAE，推理时无均值偏向。

    与 ACTLite 的区别:
    - 去掉 CVAE encoder / reparameterize / latent_proj
    - Transformer memory 只有 [state_emb]，无 z_emb
    - 训练时直接用 L1 loss，无 KL 散度
    - 推理时输出完全由 state 决定，不存在 z=0 均值偏向
    """

    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 30,
        chunk_size: int = 20,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.d_model = d_model

        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.pos_encoding = SinusoidalPositionEncoding(d_model, max_len=chunk_size + 1)
        self.action_query = nn.Embedding(chunk_size, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim),
        )

    def forward(self, state: torch.Tensor, action_chunk: torch.Tensor = None):
        B = state.size(0)
        state_emb = self.state_proj(state).unsqueeze(1)
        memory = state_emb

        query_indices = torch.arange(self.chunk_size, device=state.device)
        queries = self.action_query(query_indices).unsqueeze(0).expand(B, -1, -1)
        queries = self.pos_encoding(queries)

        decoded = self.transformer_decoder(queries, memory)
        pred_actions = self.action_head(decoded)

        if action_chunk is not None:
            return pred_actions, None, None
        return pred_actions, None, None


class ACTDetVision(nn.Module):
    """确定性 ACT + 视觉 — 无 CVAE，推理时无均值偏向。"""

    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 30,
        num_cameras: int = 1,
        img_size: int = 128,
        vision_backbone: str = "lightweight",
        chunk_size: int = 20,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_cameras = num_cameras
        self.img_size = img_size
        self.chunk_size = chunk_size
        self.d_model = d_model

        self.vision_encoder = LightweightVisionEncoder(d_model=d_model)

        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.pos_encoding = SinusoidalPositionEncoding(d_model, max_len=chunk_size + 1 + num_cameras)
        self.action_query = nn.Embedding(chunk_size, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim),
        )

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = images.shape
        imgs_flat = images.reshape(B * N, C, H, W)
        feats = self.vision_encoder(imgs_flat)
        return feats.reshape(B, N, self.d_model)

    def forward(self, state: torch.Tensor, images: torch.Tensor = None, action_chunk: torch.Tensor = None):
        B = state.size(0)

        if images is not None:
            vision_feats = self.encode_vision(images)
        else:
            vision_feats = torch.zeros(B, self.num_cameras, self.d_model, device=state.device)

        state_emb = self.state_proj(state).unsqueeze(1)
        memory = torch.cat([state_emb, vision_feats], dim=1)

        query_indices = torch.arange(self.chunk_size, device=state.device)
        queries = self.action_query(query_indices).unsqueeze(0).expand(B, -1, -1)
        queries = self.pos_encoding(queries)

        decoded = self.transformer_decoder(queries, memory)
        pred_actions = self.action_head(decoded)

        if action_chunk is not None:
            return pred_actions, None, None
        return pred_actions, None, None
