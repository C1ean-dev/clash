"""Masked policy, behavioral cloning, and PPO losses shared by both decks."""

from collections import Counter
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile, TemporaryFile
import zlib

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from royaleimitate.replays import CARD_NAMES
from xbow_arena import ArenaCapture, arena_enabled
from xbow_environment import (
    ID_PLANE_CHANNELS, RIVAL_DECKS, obs_profile_for_architecture, pack_id_planes,
)
from xbow_metrics import check_training_stop

from xbow_replays import DECKS, reconstruct, winning_replays


PPO_REFERENCE_CACHE_BYTES = 1024 * 1024**2
EMPTY_ID_PLANES = np.empty(0, dtype=np.uint8)
DEFAULT_CARD_VOCAB = 147
DEFAULT_UNIT_VOCAB = 256
DEFAULT_V3_CHANNELS = 64
BC_VALUE_LOSS_WEIGHT = 0.25
BC_DEFAULT_EARLY_STOP_PATIENCE = 4
RAW_FRAME_ESTIMATE_BYTES = 45 * 1024**2  # Conservative per-match budget (~450 v3 frames).


class ResBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(x + h)


class XBowPolicy(nn.Module):
    def __init__(self, spatial_channels: int, vector_size: int, actions: int, architecture=3,
                 placement_prior=None, hand_start=0, card_vocab=DEFAULT_CARD_VOCAB,
                 unit_vocab=DEFAULT_UNIT_VOCAB, channels=DEFAULT_V3_CHANNELS,
                 obs_profile: str | None = None):
        super().__init__()
        self.spec = (spatial_channels, vector_size, actions)
        self.architecture = architecture
        self.hand_start = hand_start
        self.card_vocab = card_vocab
        self.unit_vocab = unit_vocab
        self.channels = channels
        self._obs_profile = obs_profile_for_architecture(architecture, obs_profile)
        self.register_buffer("placement_prior", placement_prior)

        if architecture == 3:
            self._init_v3(spatial_channels, vector_size, actions, card_vocab, unit_vocab, channels)
            return

        conditioned = architecture == 2
        self.board = nn.Sequential(
            nn.Conv2d(spatial_channels + 2 * conditioned, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(),
        )
        self.context = nn.Sequential(
            nn.Linear(vector_size + 24 * 8 * 4, 128), nn.LayerNorm(128), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((8, 4))
        self.placement = nn.Conv2d(24 if conditioned else 40, 4, 1)
        self.condition = nn.Linear(128, 4 * 24 if conditioned else 16)
        self.slot_bias = nn.Linear(128, 4)
        self.special = nn.Linear(128, actions - 4 * 32 * 18)
        self.tile_bias = nn.Parameter(torch.zeros(4, 32, 18))
        self.value = nn.Linear(128, 1)
        if conditioned:
            y, x = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 18),
                                  indexing="ij")
            self.register_buffer("coordinates", torch.stack((x, y)), persistent=False)

    def _init_v3(self, spatial_channels, vector_size, actions, card_vocab, unit_vocab, channels):
        card_dim = 12
        spell_dim = 8
        unit_dim = 8
        broadcast_dim = 32
        groups = 8 if channels % 8 == 0 else 4
        n_buttons = actions - 1 - 4 * 32 * 18

        self.card_embed = nn.Embedding(card_vocab, card_dim, padding_idx=0)
        self.spell_proj = nn.Linear(card_dim, spell_dim, bias=False)
        self.unit_embed = nn.Embedding(unit_vocab, unit_dim, padding_idx=0)

        self.vector_encoder = nn.Sequential(
            nn.Linear(vector_size, 192), nn.LayerNorm(192), nn.SiLU(),
            nn.Linear(192, 128), nn.LayerNorm(128), nn.SiLU(),
        )
        self.vector_to_board = nn.Linear(128, broadcast_dim)

        stem_in = (
            spatial_channels
            + 4  # legal placement mask planes per hand slot
            + 2  # normalized (x, y) coordinates
            + broadcast_dim
            + 2 * card_dim
            + 4 * spell_dim
            + 2 * unit_dim
        )
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
        )
        self.fine_block = ResBlock(channels, groups)
        self.down = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
        )
        self.coarse_blocks = nn.Sequential(
            ResBlock(channels, groups),
            ResBlock(channels, groups),
            ResBlock(channels, groups),
        )
        self.fuse_block = ResBlock(channels, groups)

        self.context = nn.Sequential(
            nn.Linear(2 * channels + 128, 192), nn.LayerNorm(192), nn.SiLU(),
        )
        self.tile_proj = nn.Conv2d(channels, channels, 1)
        self.slot_encoder = nn.Sequential(
            nn.Linear(card_dim + 192, channels), nn.LayerNorm(channels), nn.SiLU(),
        )
        self.tile_query = nn.Linear(channels, channels)
        self.slot_head = nn.Linear(channels + 1, 1)
        self.button_head = nn.Linear(192, n_buttons) if n_buttons > 0 else None
        self.wait_head = nn.Linear(192, 1)
        self.prior_weight = nn.Parameter(torch.tensor(0.5))
        self.value = nn.Sequential(
            nn.Linear(192, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, 1),
        )
        y, x = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 18), indexing="ij")
        self.register_buffer("coordinates", torch.stack((x, y)), persistent=False)
        self._init_v3_weights()

    def _init_v3_weights(self):
        hidden_gain = math.sqrt(2.0)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=hidden_gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.card_embed.weight.normal_(0.0, 0.05)
            self.card_embed.weight[0].zero_()
            self.unit_embed.weight.normal_(0.0, 0.05)
            self.unit_embed.weight[0].zero_()
        for head in (self.tile_query, self.slot_head, self.wait_head, self.button_head):
            if head is not None:
                nn.init.orthogonal_(head.weight, gain=0.01)
                nn.init.zeros_(head.bias)
        with torch.no_grad():
            # Start near Clash Royale's ~10% action rate (sigmoid(-2.2) ~= 0.10).
            self.wait_head.bias.fill_(2.2)
        nn.init.orthogonal_(self.value[-1].weight, gain=0.1)
        nn.init.zeros_(self.value[-1].bias)

    def encode_context(self, spatial, vector, mask, id_planes=None):
        """Return the trunk's global context embedding consumed by self.value."""
        if self.architecture == 3:
            batch = spatial.shape[0]
            dtype = spatial.dtype
            legal_tiles = mask[:, 1:1 + 4 * 32 * 18].reshape(batch, 4, 32, 18)
            coords = self.coordinates.to(dtype=dtype).expand(batch, -1, -1, -1)
            vec_ctx = self.vector_encoder(vector)
            board_ctx = self.vector_to_board(vec_ctx).to(dtype=dtype)[:, :, None, None].expand(-1, -1, 32, 18)
            if id_planes is not None:
                card_ids = id_planes[:, 0:2].long().clamp(0, self.card_vocab - 1)
                spell_ids = id_planes[:, 2:6].long().clamp(0, self.card_vocab - 1)
                unit_ids = id_planes[:, 6:8].long().clamp(0, self.unit_vocab - 1)
                card_emb = self.card_embed(card_ids).permute(0, 1, 4, 2, 3).flatten(1, 2).to(dtype=dtype)
                spell_emb = self.spell_proj(self.card_embed(spell_ids)).permute(0, 1, 4, 2, 3).flatten(1, 2).to(dtype=dtype)
                unit_emb = self.unit_embed(unit_ids).permute(0, 1, 4, 2, 3).flatten(1, 2).to(dtype=dtype)
            else:
                card_emb = spatial.new_zeros((batch, 2 * self.card_embed.embedding_dim, 32, 18))
                spell_emb = spatial.new_zeros((batch, 4 * self.spell_proj.out_features, 32, 18))
                unit_emb = spatial.new_zeros((batch, 2 * self.unit_embed.embedding_dim, 32, 18))
            stem_in = torch.cat(
                (spatial, legal_tiles.to(dtype=dtype), coords, board_ctx, card_emb, spell_emb, unit_emb),
                dim=1,
            )
            h_fine = self.fine_block(self.stem(stem_in))
            g = self.coarse_blocks(self.down(h_fine))
            g_up = F.interpolate(g, size=(32, 18), mode="bilinear", align_corners=False)
            board = self.fuse_block(h_fine + g_up)
            pooled = torch.cat((board.mean(dim=(2, 3)), board.amax(dim=(2, 3))), dim=1)
            return self.context(torch.cat((pooled, vec_ctx), dim=1))
        if self.architecture == 2:
            spatial = torch.cat((spatial, self.coordinates.expand(len(spatial), -1, -1, -1)), 1)
        board = self.board(spatial)
        return self.context(torch.cat((self.pool(board).flatten(1), vector), dim=1))

    @property
    def obs_profile(self) -> str:
        return self._obs_profile

    def forward(self, spatial, vector, mask, id_planes=None):
        logits, value = self.logits_and_value(spatial, vector, mask, id_planes=id_planes)
        return Categorical(logits=logits, validate_args=False), value

    def _logits_and_value_v3(self, spatial, vector, mask, id_planes=None):
        batch = spatial.shape[0]
        dtype = spatial.dtype
        legal_tiles = mask[:, 1:1 + 4 * 32 * 18].reshape(batch, 4, 32, 18)
        coords = self.coordinates.to(dtype=dtype).expand(batch, -1, -1, -1)
        vec_ctx = self.vector_encoder(vector)
        board_ctx = self.vector_to_board(vec_ctx).to(dtype=dtype)[:, :, None, None].expand(-1, -1, 32, 18)

        if id_planes is not None:
            card_ids = id_planes[:, 0:2].long().clamp(0, self.card_vocab - 1)
            spell_ids = id_planes[:, 2:6].long().clamp(0, self.card_vocab - 1)
            unit_ids = id_planes[:, 6:8].long().clamp(0, self.unit_vocab - 1)
            card_emb = self.card_embed(card_ids).permute(0, 1, 4, 2, 3).flatten(1, 2).to(dtype=dtype)
            spell_emb = self.spell_proj(self.card_embed(spell_ids)).permute(0, 1, 4, 2, 3).flatten(1, 2).to(dtype=dtype)
            unit_emb = self.unit_embed(unit_ids).permute(0, 1, 4, 2, 3).flatten(1, 2).to(dtype=dtype)
        else:
            card_emb = spatial.new_zeros((batch, 2 * self.card_embed.embedding_dim, 32, 18))
            spell_emb = spatial.new_zeros((batch, 4 * self.spell_proj.out_features, 32, 18))
            unit_emb = spatial.new_zeros((batch, 2 * self.unit_embed.embedding_dim, 32, 18))

        stem_in = torch.cat(
            (spatial, legal_tiles.to(dtype=dtype), coords, board_ctx, card_emb, spell_emb, unit_emb),
            dim=1,
        )
        h_fine = self.fine_block(self.stem(stem_in))
        g = self.coarse_blocks(self.down(h_fine))
        g_up = F.interpolate(g, size=(32, 18), mode="bilinear", align_corners=False)
        board = self.fuse_block(h_fine + g_up)

        pooled = torch.cat((board.mean(dim=(2, 3)), board.amax(dim=(2, 3))), dim=1)
        context = self.context(torch.cat((pooled, vec_ctx), dim=1))

        card_classes = self.card_vocab - 1
        hand = vector[:, self.hand_start:self.hand_start + 4 * card_classes].reshape(batch, 4, card_classes)
        raw_cards = hand.argmax(dim=-1)
        has_card = hand.sum(dim=-1) > 0.5
        # In card_ids: 0=empty, 1=tower, 2+card_id=card; in own_hand_cards: 0..num_cards-1=card_id, last=empty.
        hand_card_ids = torch.where(has_card & (raw_cards < self.card_vocab - 2), raw_cards + 2, 0)
        hand_emb = self.card_embed(hand_card_ids)

        slot_in = torch.cat((hand_emb, context[:, None, :].expand(-1, 4, -1)), dim=-1)
        slot_feat = self.slot_encoder(slot_in)
        tile_query = self.tile_query(slot_feat)
        tile_map = self.tile_proj(board)
        tiles = (torch.einsum("bsc,bcyx->bsyx", tile_query, tile_map) / (self.channels ** 0.5)).flatten(2)
        if self.placement_prior is not None:
            prior_idx = raw_cards.clamp(0, len(self.placement_prior) - 1)
            tiles = tiles + self.prior_weight * self.placement_prior[prior_idx]

        legal_flat = legal_tiles.flatten(2)
        slot_legal = legal_flat.any(dim=-1)
        best_tile = tiles.masked_fill(~legal_flat, -1e4).amax(dim=-1, keepdim=True)
        safe_best = torch.where(slot_legal.unsqueeze(-1), best_tile.clamp(-20.0, 20.0), 0.0)
        tile_logp = tiles.masked_fill(~legal_flat, -1e9).log_softmax(dim=-1)

        slot_logits = self.slot_head(torch.cat((slot_feat, safe_best), dim=-1)).squeeze(-1)
        if self.button_head is not None:
            button_logits = self.button_head(context)
            cand_logits = torch.cat((slot_logits, button_logits), dim=1)
            cand_legal = torch.cat((slot_legal, mask[:, 1 + 4 * 32 * 18:]), dim=1)
        else:
            cand_logits = slot_logits
            cand_legal = slot_legal
        cand_logp = cand_logits.masked_fill(~cand_legal, -1e9).log_softmax(dim=1)

        any_legal = cand_legal.any(dim=1)
        act_logit = -self.wait_head(context).squeeze(-1)
        act_logit = act_logit.masked_fill(~any_legal, -1e9)
        gate_logp = F.log_softmax(torch.stack((torch.zeros_like(act_logit), act_logit), dim=-1), dim=-1)
        wait_logp = gate_logp[:, 0]
        act_logp = gate_logp[:, 1]

        card_logp = (act_logp[:, None, None] + cand_logp[:, :4, None] + tile_logp).flatten(1)
        if self.button_head is not None:
            btn_logp = act_logp[:, None] + cand_logp[:, 4:]
            logits = torch.cat((wait_logp[:, None], card_logp, btn_logp), dim=1)
        else:
            logits = torch.cat((wait_logp[:, None], card_logp), dim=1)
        return logits.masked_fill(~mask, -1e9), self.value(context).squeeze(-1)

    def logits_and_value(self, spatial, vector, mask, id_planes=None):
        if self.architecture == 3:
            return self._logits_and_value_v3(spatial, vector, mask, id_planes=id_planes)
        if self.architecture == 2:
            spatial = torch.cat((spatial, self.coordinates.expand(len(spatial), -1, -1, -1)), 1)
        board = self.board(spatial)
        context = self.context(torch.cat((self.pool(board).flatten(1), vector), dim=1))
        if self.architecture == 2:
            condition = self.condition(context).reshape(-1, 4, 24)
            compatibility = torch.einsum("bsc,bcyx->bsyx", condition, board) / 24**0.5
            tiles = (self.placement(board) + self.tile_bias + compatibility).flatten(2)
            if self.placement_prior is not None:
                classes = len(self.placement_prior)
                hand = vector[:, self.hand_start:self.hand_start + 4 * classes]
                cards = hand.reshape(-1, 4, classes).argmax(-1)
                tiles = tiles + self.placement_prior[cards]
            legal_tiles = mask[:, 1:1 + 4 * 32 * 18].reshape(-1, 4, 32 * 18)
            tiles = tiles.masked_fill(~legal_tiles, -1e9).log_softmax(dim=-1)
        else:
            condition = self.condition(context)[:, :, None, None].expand(-1, -1, 32, 18)
            tiles = (self.placement(torch.cat((board, condition), dim=1)) + self.tile_bias).flatten(2)
        tiles = tiles + self.slot_bias(context).unsqueeze(-1)
        special = self.special(context)
        plays = torch.cat((tiles.flatten(1), special[:, 1:]), dim=1)
        plays = plays.masked_fill(~mask[:, 1:], -1e9).log_softmax(dim=1)
        logits = torch.cat((special[:, :1], plays), dim=1)
        return logits.masked_fill(~mask, -1e9), self.value(context).squeeze(-1)

    @classmethod
    def from_observation(cls, obs, placement_prior=None, hand_start=0, *, env=None,
                         architecture=None, obs_profile=None):
        has_ids = pack_id_planes(obs) is not None
        if architecture is None:
            architecture = 3 if has_ids else 2
        card_vocab = DEFAULT_CARD_VOCAB
        unit_vocab = DEFAULT_UNIT_VOCAB
        if env is not None:
            builder = env.obs_builder
            if hasattr(builder, "num_cards") and builder.num_cards:
                card_vocab = int(builder.num_cards) + 2
            if hand_start == 0:
                hand_start = own_hand_field(env).start
            if obs_profile is None and architecture >= 3:
                if getattr(builder, "card_identity", False) and not getattr(builder, "unit_status", False):
                    obs_profile = "live"
                else:
                    obs_profile = "v3"
        elif placement_prior is not None:
            card_vocab = len(placement_prior) + 1
        if obs_profile is None and architecture >= 3 and obs["spatial"].shape[0] <= 40:
            obs_profile = "live"
        return cls(
            obs["spatial"].shape[0], obs["vector"].size, obs["action_mask"].size,
            architecture=architecture, placement_prior=placement_prior, hand_start=hand_start,
            card_vocab=card_vocab, unit_vocab=unit_vocab, obs_profile=obs_profile,
        )

    @torch.no_grad()
    def decide(self, obs, device, deterministic=False, generator=None, temperature=1.0):
        id_planes = pack_id_planes(obs) if self.architecture >= 3 else None
        id_tensor = torch.as_tensor(id_planes, device=device).unsqueeze(0) if id_planes is not None else None
        mask = torch.as_tensor(obs["action_mask"], device=device, dtype=torch.bool).unsqueeze(0)
        logits, value = self.logits_and_value(
            torch.as_tensor(obs["spatial"], device=device).unsqueeze(0),
            torch.as_tensor(obs["vector"], device=device).unsqueeze(0),
            mask,
            id_planes=id_tensor,
        )
        distribution = Categorical(logits=logits, validate_args=False)
        if deterministic:
            choice = distribution.probs[:, 1:].argmax(-1) + 1
            action = torch.where(distribution.probs[:, 0] >= 0.5, 0, choice)
        else:
            if temperature != 1.0 and mask[:, 1:].any():
                # Keep exact P(wait) vs P(play) so sharpening never paralyzes the policy;
                # apply temperature strictly to the conditional P(card, tile | play) distribution.
                p_wait = distribution.probs[:, :1]
                p_play = (1.0 - p_wait).clamp_min(0.0)
                play_logits = (logits[:, 1:] / float(temperature)).masked_fill(~mask[:, 1:], -1e9)
                play_probs = F.softmax(play_logits, dim=-1)
                probs = torch.cat([p_wait, p_play * play_probs], dim=-1)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                sample_dist = Categorical(probs=probs, validate_args=False)
            else:
                sample_dist = distribution
            if generator is not None:
                action = torch.multinomial(sample_dist.probs, 1, generator=generator).squeeze(-1)
            else:
                action = sample_dist.sample()
        return int(action.item()), float(distribution.log_prob(action).item()), float(value.item())


def checkpoint_data(policy, deck, phase, metrics, trainer=None):
    data = {
        "version": 1, "deck": deck, "phase": phase, "spec": policy.spec,
        "architecture": policy.architecture, "policy": policy.state_dict(), "metrics": metrics,
        "hand_start": policy.hand_start, "trainer": trainer,
    }
    if policy.architecture >= 3:
        data["arch_config"] = {
            "card_vocab": policy.card_vocab,
            "unit_vocab": policy.unit_vocab,
            "channels": policy.channels,
            "obs_profile": policy.obs_profile,
        }
    return data


def save_checkpoint(path, policy, deck, phase, metrics, trainer=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as temporary:
        local = Path(temporary.name)
    try:
        torch.save(checkpoint_data(policy, deck, phase, metrics, trainer), local)
        os.replace(local, path)
    finally:
        local.unlink(missing_ok=True)


def load_checkpoint(path, deck, device, *, memory_format=torch.preserve_format):
    saved = torch.load(path, map_location=device, weights_only=True)
    return policy_from_checkpoint(saved, deck, device, memory_format=memory_format), saved


def policy_from_checkpoint(saved, deck, device, *, memory_format=torch.preserve_format):
    if saved.get("version") != 1 or saved.get("deck") != deck:
        raise ValueError(f"Checkpoint incompativel com {deck}")
    architecture = saved.get("architecture", 1)
    if architecture not in (1, 2, 3):
        raise ValueError(f"Arquitetura desconhecida no checkpoint: {architecture}")
    arch_config = saved.get("arch_config") or {}
    policy = XBowPolicy(
        *saved["spec"], architecture=architecture,
        placement_prior=saved["policy"].get("placement_prior"),
        hand_start=saved.get("hand_start", 0),
        card_vocab=arch_config.get("card_vocab", DEFAULT_CARD_VOCAB),
        unit_vocab=arch_config.get("unit_vocab", DEFAULT_UNIT_VOCAB),
        channels=arch_config.get("channels", DEFAULT_V3_CHANNELS),
        obs_profile=arch_config.get("obs_profile"),
    ).to(device=device, memory_format=memory_format)
    policy.load_state_dict(saved["policy"])
    return policy.eval()


def kata_weights(labels: np.ndarray, action_rate: float) -> np.ndarray:
    """KataCR Eq. 8, with the wait-frame weight before the first play."""
    ticks = np.arange(len(labels))
    latest = np.maximum.accumulate(np.where(labels != 0, ticks, -1))
    wait_weight = 1.0 / (1.0 - action_rate)
    return np.where(
        latest < 0, wait_weight,
        np.maximum(wait_weight, 1.0 / (action_rate * (ticks - latest + 1))),
    )


@dataclass(frozen=True)
class MirrorSpec:
    """Indices that swap when mirroring the arena horizontally (x -> 17 - x)."""

    spatial_swaps: tuple[tuple[int, int], ...] = ()
    vector_swaps: tuple[tuple[int, int], ...] = ()

    @classmethod
    def from_env(cls, env):
        names = list(env.obs_builder.channel_names())
        name_to_idx = {name: idx for idx, name in enumerate(names)}
        spatial_swaps = []
        for left, right in (("own_dx_plus", "own_dx_minus"), ("enemy_dx_plus", "enemy_dx_minus")):
            if left in name_to_idx and right in name_to_idx:
                spatial_swaps.append((name_to_idx[left], name_to_idx[right]))
        vector_swaps = []
        offset = 0
        for field in env.obs_builder.vector_layout():
            if field.key in ("own_tower_hp", "enemy_tower_hp") and field.size >= 3:
                # Tower order is [king, left_princess, right_princess].
                vector_swaps.append((offset + 1, offset + 2))
            offset += field.size
        return cls(tuple(spatial_swaps), tuple(vector_swaps))


def mirror_sample(spatial, vector, mask, id_planes, target, spec: MirrorSpec):
    """Exact horizontal mirror (x -> 17 - x) of a single observation and target action."""
    spatial = spatial[..., ::-1].copy()
    for a, b in spec.spatial_swaps:
        tmp = spatial[a].copy()
        spatial[a] = spatial[b]
        spatial[b] = tmp
    vector = vector.copy()
    for a, b in spec.vector_swaps:
        vector[a], vector[b] = vector[b], vector[a]
    mask = mask.copy()
    tile_end = 1 + 4 * 32 * 18
    tiles_mask = mask[1:tile_end].reshape(4, 32, 18)
    mask[1:tile_end] = tiles_mask[..., ::-1].reshape(-1)
    if id_planes.ndim == 3:
        id_planes = id_planes[..., ::-1].copy()
    if 1 <= target <= 4 * 32 * 18:
        slot, cell = divmod(int(target) - 1, 32 * 18)
        y, x = divmod(cell, 18)
        target = 1 + slot * (32 * 18) + y * 18 + (17 - x)
    return spatial, vector, mask, id_planes, target


class ReplayFrames(Dataset):
    def __init__(self, matches, action_count, *, mirror_spec=None, mirror_prob=0.0,
                 include_values=False, seed=42):
        self.matches = matches
        self.ends = np.cumsum([len(match.actions) for match in matches])
        self.action_count = action_count
        self.timing_correction = None
        self.mirror_spec = mirror_spec
        self.mirror_prob = float(mirror_prob)
        self.include_values = bool(include_values)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        match_id = int(np.searchsorted(self.ends, index, side="right"))
        frame = index - (self.ends[match_id - 1] if match_id else 0)
        match = self.matches[match_id]
        spatial, vector, mask, id_planes = match.frame(frame)
        spatial = np.array(spatial)
        vector = np.array(vector)
        unpacked_mask = np.unpackbits(mask, count=self.action_count).astype(bool)
        ids = np.array(id_planes) if id_planes is not None else EMPTY_ID_PLANES
        target = int(match.actions[frame])
        if self.mirror_spec is not None and self.mirror_prob > 0.0 and self.rng.random() < self.mirror_prob:
            spatial, vector, unpacked_mask, ids, target = mirror_sample(
                spatial, vector, unpacked_mask, ids, target, self.mirror_spec,
            )
        data = (spatial, vector, unpacked_mask, ids, target)
        if self.timing_correction is not None:
            data = (*data, self.timing_correction[index])
        if self.include_values:
            val = float(match.values[frame]) if getattr(match, "values", None) is not None else 0.0
            data = (*data, np.float32(val))
        return data

    def weights(self):
        plays = sum(np.count_nonzero(match.actions) for match in self.matches)
        rate = plays / len(self)
        if not 0 < rate < 1:
            return np.ones(len(self))
        return np.concatenate([kata_weights(match.actions, rate) for match in self.matches])


class DiskMatchSamples:
    """Store and read each compressed observation independently for random sampling."""

    def __init__(self, match, store):
        self.tag = match.tag
        self.actions = match.actions
        self.values = getattr(match, "values", None)
        self.store = store
        self.has_id_planes = match.id_planes is not None
        fields = [
            ("spatial", match.spatial),
            ("vector", match.vector),
            ("mask", match.mask),
        ]
        if self.has_id_planes:
            fields.append(("id_planes", match.id_planes))
        self.dtype = np.dtype([(name, values.dtype, values.shape[1:]) for name, values in fields])
        self.offsets = np.empty(len(self.actions) + 1, dtype=np.int64)
        offset = store.tell()
        row = np.empty(1, dtype=self.dtype)
        columns = (
            (match.spatial, match.vector, match.mask, match.id_planes)
            if self.has_id_planes else (match.spatial, match.vector, match.mask)
        )
        for index, frame in enumerate(zip(*columns, strict=True)):
            if index % 64 == 0:
                check_training_stop()
            self.offsets[index] = offset
            row[0] = frame
            offset += store.write(zlib.compress(row, level=1))
        self.offsets[-1] = offset

    def frame(self, index):
        offset, end = self.offsets[index:index + 2]
        self.store.seek(offset)
        raw = zlib.decompress(self.store.read(end - offset))
        item = np.frombuffer(raw, dtype=self.dtype, count=1)[0]
        if self.has_id_planes:
            return item["spatial"], item["vector"], item["mask"], item["id_planes"]
        return item["spatial"], item["vector"], item["mask"], None

    def hands(self, acted, hand):
        indices = np.flatnonzero(acted)
        vectors = np.empty((len(indices), hand.stop - hand.start),
                           dtype=self.dtype["vector"].base)
        for row, index in enumerate(indices):
            _, vector, _, _ = self.frame(index)
            vectors[row] = vector[hand]
        return vectors


class RawDiskMatchSamples:
    """Fixed-stride uncompressed binary frame store: ~5x faster random reads than per-frame zlib."""

    def __init__(self, match, store):
        self.tag = match.tag
        self.actions = match.actions
        self.values = getattr(match, "values", None)
        self.store = store
        self.has_id_planes = match.id_planes is not None
        fields = [
            ("spatial", match.spatial),
            ("vector", match.vector),
            ("mask", match.mask),
        ]
        if self.has_id_planes:
            fields.append(("id_planes", match.id_planes))
        self.dtype = np.dtype([(name, values.dtype, values.shape[1:]) for name, values in fields])
        self.itemsize = self.dtype.itemsize
        self.start = store.tell()
        buf = np.empty(len(self.actions), dtype=self.dtype)
        buf["spatial"] = match.spatial
        buf["vector"] = match.vector
        buf["mask"] = match.mask
        if self.has_id_planes:
            buf["id_planes"] = match.id_planes
        store.write(buf.tobytes())

    def frame(self, index):
        self.store.seek(self.start + int(index) * self.itemsize)
        raw = self.store.read(self.itemsize)
        item = np.frombuffer(raw, dtype=self.dtype, count=1)[0]
        if self.has_id_planes:
            return item["spatial"], item["vector"], item["mask"], item["id_planes"]
        return item["spatial"], item["vector"], item["mask"], None

    def hands(self, acted, hand):
        indices = np.flatnonzero(acted)
        vectors = np.empty((len(indices), hand.stop - hand.start),
                           dtype=self.dtype["vector"].base)
        for row, index in enumerate(indices):
            _, vector, _, _ = self.frame(index)
            vectors[row] = vector[hand]
        return vectors


def collect_replays(env, parquet_path, deck, limit, seed, frame_store=None, validation_tags=None,
                    report=None, arena_output=None, examples=None, *,
                    competitive_only=True, use_raw_disk=False):
    counts, rejected = Counter(), Counter()
    matches = []
    fixed_validation = validation_tags is not None
    heldout = []
    remaining = set(validation_tags or ())
    if fixed_validation and (not remaining or len(remaining) >= limit):
        raise ValueError("O limite deve acomodar a validação fixa e pelo menos um replay de treino.")
    training_limit = limit - len(remaining)
    disk_cls = RawDiskMatchSamples if use_raw_disk else DiskMatchSamples
    all_decks = DECKS | RIVAL_DECKS
    for replay in winning_replays(
        parquet_path, deck, counts,
        competitive_only=competitive_only, keep_tags=remaining or None,
        deck_cards=all_decks.get(deck),
    ):
        check_training_stop()
        is_validation = fixed_validation and replay.tag in remaining
        if fixed_validation and not is_validation and len(matches) >= training_limit:
            continue
        recording = (ArenaCapture(arena_output, "replays", env, replay=replay.tag,
                                  seat=replay.seat, number=len(matches) + len(heldout) + 1)
                     if arena_output is not None else nullcontext())
        with recording as capture:
            samples = reconstruct(env, replay, seed, capture)
            if capture and not isinstance(samples, str) and capture.example and examples is not None:
                examples[replay.tag] = (replay.seat, *capture.example)
        if isinstance(samples, str):
            if is_validation:
                raise ValueError(f"Replay reservado {replay.tag}: {samples}")
            rejected[samples] += 1
            continue
        if frame_store is not None:
            samples = disk_cls(samples, frame_store)
        if is_validation:
            heldout.append(samples)
            remaining.remove(replay.tag)
        else:
            matches.append(samples)
        total = len(matches) + len(heldout)
        if total % 10 == 0:
            print(f"{deck}: {total} replays reconstruidos", flush=True)
        if report:
            report("replays", done=total, total=limit, rejected=sum(rejected.values()))
        if total == limit:
            break
    total = len(matches) + len(heldout)
    if report:
        report("replays", done=total, total=limit, rejected=sum(rejected.values()),
               complete=True)
    print(f"{deck}: {dict(counts)} | reconstruidos={total} | recusas={dict(rejected)}",
          flush=True)
    if fixed_validation:
        if remaining or not matches:
            raise ValueError(f"Conjunto fixo incompleto: faltam {len(remaining)} replays reservados.")
        return matches, heldout
    if len(matches) < 2:
        raise ValueError("Sao necessarios dois replays reconstruidos para treino e validacao.")
    order = np.random.default_rng(seed).permutation(len(matches))
    validation_count = max(1, round(len(matches) * 0.2))
    return (
        [matches[i] for i in order[validation_count:]],
        [matches[i] for i in order[:validation_count]],
    )


@torch.no_grad()
def evaluate_clone(policy, batches, device, tactics=None):
    nll = count = play = expert_actions = play_probability = correct_slots = correct_tiles = 0
    value_ae = 0.0
    has_values = False
    tactic_totals, tactic_hits, tactic_placements = Counter(), Counter(), Counter()
    for batch in batches:
        if len(batch) == 6:
            spatial, vector, mask, id_planes, target, value_target = batch
            has_values = True
        else:
            spatial, vector, mask, id_planes, target = batch
            value_target = None
        target = target.to(device)
        ids_tensor = id_planes.to(device) if id_planes.ndim == 4 else None
        dist, value_pred = policy(spatial.to(device, torch.float32), vector.to(device, torch.float32),
                                  mask.to(device), id_planes=ids_tensor)
        if value_target is not None:
            value_ae += (value_pred - value_target.to(device, torch.float32)).abs().sum().item()
        placement = dist.probs[:, 1:1 + 4 * 32 * 18].reshape(-1, 4, 32 * 18)
        predicted_slot = placement.sum(-1).argmax(-1)
        likely_tile = placement[torch.arange(len(target), device=device), predicted_slot].argmax(-1)
        predicted_card_action = 1 + predicted_slot * (32 * 18) + likely_tile
        nll -= dist.log_prob(target).sum().item()
        count += target.numel()
        expert_actions += (target != 0).sum().item()
        play_probability += (1 - dist.probs[:, 0]).sum().item()
        card_target = (target > 0) & (target <= 4 * 32 * 18)
        play += card_target.sum().item()
        correct_slots += (card_target & (predicted_slot == (target - 1) // (32 * 18))).sum().item()
        correct_tiles += (card_target & (predicted_card_action == target)).sum().item()
        if tactics is not None:
            hand, groups = tactics
            hold = vector[:, hand].reshape(len(target), 4, -1).argmax(-1).to(device)
            actual_slot = ((target - 1) // (32 * 18)).clamp(0, 3)
            trained_card = hold.gather(1, actual_slot[:, None]).squeeze(1)
            actual_tile = (target - 1) % (32 * 18)
            placement_with_card = placement[
                torch.arange(len(target), device=device), actual_slot,
            ].argmax(-1)
            distance = (
                (actual_tile // 18 - likely_tile // 18).abs()
                + (actual_tile % 18 - likely_tile % 18).abs()
            )
            conditional_distance = (
                (actual_tile // 18 - placement_with_card // 18).abs()
                + (actual_tile % 18 - placement_with_card % 18).abs()
            )
            for name, ids in groups.items():
                group = card_target & torch.isin(trained_card, ids)
                tactic_totals[name] += group.sum().item()
                hit = group & (predicted_slot == actual_slot) & (distance <= 2)
                tactic_hits[name] += hit.sum().item()
                tactic_placements[name] += (group & (conditional_distance <= 2)).sum().item()
    metrics = {
        "nll": nll / count, "frames": count, "card_plays": play,
        "conditional_card_accuracy": correct_slots / max(1, play),
        "conditional_card_and_tile_accuracy": correct_tiles / max(1, play),
        "mean_play_probability": play_probability / count, "expert_play_rate": expert_actions / count,
    }
    if has_values:
        metrics["value_mae"] = value_ae / count
    if tactics is not None:
        metrics["card_and_within_two_tiles"] = {
            name: {"correct": tactic_hits[name], "examples": total}
            for name, total in tactic_totals.items()
        }
        metrics["within_two_tiles_given_card"] = {
            name: {"correct": tactic_placements[name], "examples": total}
            for name, total in tactic_totals.items()
        }
    return metrics


def own_hand_field(env):
    offset = 0
    for field in env.obs_builder.vector_layout():
        if field.key == "own_hand_cards":
            return slice(offset, offset + field.size)
        offset += field.size
    raise ValueError("O ambiente nao informa o campo da mao propria.")


def fitted_placement_prior(matches, hand):
    """Smoothed per-card placement frequencies fitted on training matches only."""
    classes = (hand.stop - hand.start) // 4
    tiles = 32 * 18
    counts = np.zeros((classes, tiles), dtype=np.float32)
    for match in matches:
        acted = (match.actions > 0) & (match.actions <= 4 * tiles)
        slots, cells = np.divmod(match.actions[acted] - 1, tiles)
        hands = match.hands(acted, hand).reshape(-1, 4, classes)
        cards = hands[np.arange(len(slots)), slots].argmax(-1)
        np.add.at(counts, (cards, cells), 1)
    grid = torch.from_numpy(counts.reshape(classes, 1, 32, 18))
    grid = (grid + grid.flip(-1)) / 2
    kernel = torch.tensor([[[[1, 2, 1], [2, 8, 2], [1, 2, 1]]]], dtype=torch.float32) / 20
    smooth = nn.functional.conv2d(grid, kernel, padding=1).flatten(1)
    return 0.8 * (smooth + 0.05).log()


def tactical_fields(env, device):
    hand = own_hand_field(env)
    cards = env.engine.cards()
    groups = {
        "x-bow": [card.card_id for card in cards if card.name == CARD_NAMES["x-bow"]],
        "tesla": [card.card_id for card in cards if card.name == CARD_NAMES["tesla"]],
        "one_elixir": [card.card_id for card in cards if card.elixir == 1],
    }
    return hand, {name: torch.tensor(ids, device=device) for name, ids in groups.items()}


def replay_validation(env, args, device, output):
    reference, saved = load_checkpoint(args.bc_checkpoint, args.deck, device)
    if saved["phase"] != "bc":
        raise ValueError("A validacao de replays requer o checkpoint de imitação.")
    metrics = saved["metrics"]
    remaining = set(metrics["validation_matches"])
    if not remaining or remaining.intersection(metrics["training_matches"]):
        raise ValueError("Checkpoint sem divisão de partidas de validação válida.")
    matches = []
    for replay in winning_replays(args.parquet_path, args.deck, Counter()):
        if replay.tag not in remaining:
            continue
        reconstructed = reconstruct(env, replay, metrics.get("seed", args.seed))
        if isinstance(reconstructed, str):
            raise ValueError(f"{replay.tag}: {reconstructed}")
        matches.append(reconstructed)
        remaining.remove(replay.tag)
        if not remaining:
            break
    if remaining:
        raise ValueError(f"Faltam {len(remaining)} partidas de validação nos arquivos fornecidos.")
    batches = DataLoader(ReplayFrames(matches, reference.spec[-1]), batch_size=args.batch_size)
    tactics = tactical_fields(env, device)
    results = {"bc": evaluate_clone(reference, batches, device, tactics)}
    if args.evaluate_checkpoint:
        policy, ppo_checkpoint = load_checkpoint(args.evaluate_checkpoint, args.deck, device)
        if ppo_checkpoint["phase"] != "ppo" or policy.spec != reference.spec:
            raise ValueError("Checkpoint PPO incompatível com a referência BC.")
        results["ppo"] = evaluate_clone(policy, batches, device, tactics)
    output.mkdir(parents=True, exist_ok=True)
    (output / "replay_validation.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8",
    )
    print(json.dumps(results, indent=2), flush=True)


def clone_loss(dist, targets, timing_correction):
    """Use weighted samples for placement and restore the original play/wait rate."""
    played = targets.ne(0)
    log_wait = dist.logits[:, 0]
    log_play = dist.logits[:, 1:].logsumexp(dim=1)
    log_timing = torch.where(played, log_play, log_wait)
    log_choice = dist.log_prob(targets) - log_timing
    return -(timing_correction * log_timing + log_choice).mean()


def bc_optimizer(policy, lr=3e-4, weight_decay=2e-2):
    """AdamW with decoupled weight decay excluding biases, norms, and embeddings."""
    decay, no_decay = [], []
    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "embed" in name or "norm" in name or name.endswith("bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr)


def bc_lr_scheduler(optimizer, total_steps: int, min_lr_ratio: float = 0.05):
    """Linear warmup followed by cosine decay down to min_lr_ratio * base_lr."""
    total = max(1, int(total_steps))
    warmup = max(1, min(20, int(0.05 * total))) if total > 2 else 0

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        if total <= warmup:
            return 1.0
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def use_raw_disk_store(folder: Path, match_limit: int) -> bool:
    """Use uncompressed fixed-stride disk I/O when plenty of free space is available."""
    try:
        free = shutil.disk_usage(folder).free
    except OSError:
        return False
    return int(match_limit) * RAW_FRAME_ESTIMATE_BYTES <= free // 3


def train_clone(env, args, device, path, report=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.disk_frames:
        raw_disk = use_raw_disk_store(path.parent, args.matches)
        with TemporaryFile(prefix="frames-", dir=path.parent) as frame_store:
            fit_clone(env, args, device, path, frame_store, report, use_raw_disk=raw_disk)
    else:
        fit_clone(env, args, device, path, report=report)


def fit_clone(env, args, device, path, frame_store=None, report=None, *, use_raw_disk=False):
    validation_tags = None
    if args.holdout_checkpoint:
        _, previous = load_checkpoint(args.holdout_checkpoint, args.deck, "cpu")
        if previous["phase"] != "bc":
            raise ValueError("A validação fixa deve vir de um checkpoint de imitação.")
        previous_metrics = previous["metrics"]
        if previous_metrics.get("seed", args.seed) != args.seed:
            raise ValueError("Use a mesma semente do checkpoint para reconstituir a validação fixa.")
        validation_tags = previous_metrics["validation_matches"]
    first_obs, _ = env.reset(seed=args.seed)
    if report:
        report("replays", done=0, total=args.matches, rejected=0)
    view_output = path.parent if arena_enabled() else None
    examples = {}
    competitive_only = not getattr(args, "include_casual", False)
    train_matches, val_matches = collect_replays(
        env, args.parquet_path, args.deck, args.matches, args.seed, frame_store, validation_tags, report,
        view_output, examples, competitive_only=competitive_only, use_raw_disk=use_raw_disk,
    )
    if report:
        report("bc_start", total=args.epochs, training=len(train_matches),
               validation=len(val_matches))
    hand = own_hand_field(env)
    prior = fitted_placement_prior(train_matches, hand)
    layout = torch.channels_last if device == "cpu" else torch.preserve_format
    policy = XBowPolicy.from_observation(
        first_obs["blue"], prior, hand.start, env=env,
    ).to(device=device, memory_format=layout)
    use_mirror = not getattr(args, "no_mirror", False)
    mirror_spec = MirrorSpec.from_env(env) if use_mirror else None
    has_values = any(getattr(m, "values", None) is not None for m in train_matches)
    train = ReplayFrames(
        train_matches, policy.spec[-1],
        mirror_spec=mirror_spec, mirror_prob=0.5 if use_mirror else 0.0,
        include_values=has_values, seed=args.seed,
    )
    val = ReplayFrames(val_matches, policy.spec[-1], include_values=has_values)
    weights = train.weights()
    train.timing_correction = (weights.mean() / weights).astype(np.float32)
    sampler = WeightedRandomSampler(weights, len(train), replacement=True)
    training_batches = DataLoader(train, batch_size=args.batch_size, sampler=sampler)
    val_batches = DataLoader(val, batch_size=args.batch_size)
    optimizer = bc_optimizer(policy, lr=3e-4, weight_decay=2e-2)
    scheduler = bc_lr_scheduler(optimizer, total_steps=args.epochs * len(training_batches))
    with ExitStack() as recordings:
        probes = []
        for match, card in reversed(clone_examples(train_matches, examples)):
            seat, index, board = examples[match.tag]
            probe = clone_probe(match, index, board, seat, device, policy.spec[-1])
            capture = recordings.enter_context(
                ArenaCapture(view_output, "bc", env, replay=match.tag, seat=seat, probe_card=card)
            )
            probes.append((capture, probe))
        fit_clone_epochs(policy, train_matches, val_matches, training_batches, val_batches,
                         optimizer, args, device, path, report, probes,
                         scheduler=scheduler, mirror_enabled=use_mirror,
                         competitive_only=competitive_only)


def clone_examples(matches, examples):
    """Prefer different opening plays while keeping each comparison on a fixed board."""
    probe_limit = 3
    distinct, repeats = {}, []
    for match in matches:
        if match.tag not in examples:
            continue
        seat, _, board = examples[match.tag]
        card = next(move["card"] for move in board["moves"] if move["seat"] == seat)
        if card in distinct:
            if len(repeats) < probe_limit:
                repeats.append((match, card))
        else:
            distinct[card] = (match, card)
            if len(distinct) == probe_limit:
                break
    return (*distinct.values(), *repeats)[:probe_limit]


def clone_probe(match, index, board, seat, device, action_count):
    spatial, vector, mask, id_planes, target = ReplayFrames([match], action_count)[index]
    ids_tensor = (
        torch.as_tensor(id_planes, device=device).unsqueeze(0)
        if id_planes.ndim == 3 else None
    )
    features = (
        torch.as_tensor(spatial, dtype=torch.float32, device=device).unsqueeze(0),
        torch.as_tensor(vector, dtype=torch.float32, device=device).unsqueeze(0),
        torch.as_tensor(mask, device=device).unsqueeze(0),
        ids_tensor,
    )
    return features, int(target), board, seat


@torch.no_grad()
def show_clone_example(capture, policy, probe, **metrics):
    features, target, board, seat = probe
    dist, _ = policy(*features)
    probs = dist.probs[0]
    best_play = int(probs[1:].argmax().item()) + 1
    chosen = 0 if probs[0] >= .5 else best_play
    hand = board["sides"][seat]["hand"]
    capture.frame(board, comparison={
        "human": capture.action(target, seat, hand), "model": capture.action(chosen, seat, hand),
        "play": float(1 - probs[0]), "target": float(probs[target]), "exact": chosen == target,
        "alternative": capture.action(best_play, seat, hand) if not chosen and probs[best_play] > 0 else None,
        **metrics,
    })


def fit_clone_epochs(policy, train_matches, val_matches, training_batches, val_batches,
                     optimizer, args, device, path, report, probes, *,
                     scheduler=None, mirror_enabled=False, competitive_only=False):
    best = float("inf")
    stale_epochs = 0
    patience = getattr(args, "early_stop_patience", BC_DEFAULT_EARLY_STOP_PATIENCE)
    view_stride = max(15, len(training_batches) // 24)
    for epoch in range(args.epochs):
        policy.train()
        for batch, items in enumerate(training_batches, 1):
            check_training_stop()
            if len(items) == 7:
                spatial, vector, mask, id_planes, target, correction, value_target = items
            else:
                spatial, vector, mask, id_planes, target, correction = items
                value_target = None
            ids_tensor = id_planes.to(device) if id_planes.ndim == 4 else None
            dist, value_pred = policy(spatial.to(device, torch.float32), vector.to(device, torch.float32),
                                      mask.to(device), id_planes=ids_tensor)
            loss = clone_loss(dist, target.to(device), correction.to(device))
            if value_target is not None:
                loss = loss + BC_VALUE_LOSS_WEIGHT * F.smooth_l1_loss(
                    value_pred, value_target.to(device, torch.float32),
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            edge = batch in (1, len(training_batches))
            if report and (edge or batch % 15 == 0):
                report("bc_batch", epoch=epoch + 1, total=args.epochs, done=batch,
                       batches=len(training_batches), loss=loss.item())
            if probes and (edge or batch % view_stride == 0):
                for capture, probe in probes:
                    show_clone_example(capture, policy, probe, epoch=epoch + 1, batch=batch,
                                       batches=len(training_batches), loss=loss.item())
        policy.eval()
        check_training_stop()
        metrics = evaluate_clone(policy, val_batches, device)
        metrics["lr"] = float(optimizer.param_groups[0]["lr"])
        print(f"BC {epoch + 1}/{args.epochs}: {metrics}", flush=True)
        if report:
            report("bc_epoch", done=epoch + 1, total=args.epochs, metrics=metrics)
        if metrics["nll"] < best:
            best = metrics["nll"]
            stale_epochs = 0
            save_checkpoint(path, policy, args.deck, "bc", {
                **metrics, "training_matches": [match.tag for match in train_matches],
                "validation_matches": [match.tag for match in val_matches],
                "epoch": epoch + 1, "seed": args.seed, "timing_importance_corrected": True,
                "mirror_augmentation": mirror_enabled, "competitive_only": competitive_only,
            })
        else:
            stale_epochs += 1
            if patience and stale_epochs >= patience:
                print(
                    f"BC early stopping na epoca {epoch + 1}/{args.epochs} "
                    f"(sem melhora ha {stale_epochs} epocas; melhor NLL={best:.4f}).",
                    flush=True,
                )
                break


def warmup_value_head(policy, data, batch_size, updates=5, lr=1e-3):
    """Calibrate only the critic parameters on on-policy returns using cached trunk features."""
    if len(data) == 8:
        spatial, vector, mask, _, _, id_planes, _, returns = data
    else:
        spatial, vector, mask, _, _, _, returns = data
        id_planes = None
    value_params = list(policy.value.parameters())
    if not value_params:
        return None
    # Trunk is frozen during critic warmup: extract features once in a single no_grad pass.
    chunks = []
    with torch.no_grad():
        for start in range(0, len(returns), batch_size):
            end = start + batch_size
            batch_ids = id_planes[start:end] if id_planes is not None else None
            chunks.append(policy.encode_context(
                spatial[start:end].float(), vector[start:end].float(), mask[start:end], id_planes=batch_ids,
            ))
    features = torch.cat(chunks, dim=0)
    optimizer = torch.optim.AdamW(value_params, lr=lr, weight_decay=0.0)
    losses = []
    for _ in range(updates):
        order = torch.randperm(len(returns), device=returns.device)
        for ids in order.split(batch_size):
            value = policy.value(features[ids]).squeeze(-1)
            loss = 0.5 * (returns[ids] - value).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(value_params, 0.5)
            optimizer.step()
            losses.append(float(loss.detach().item()))
    return float(np.mean(losses)) if losses else None


def scheduled_kl_coef(base_kl: float, update: int, *, floor_ratio: float = 0.1,
                      half_life: float = 40.0) -> float:
    """Smooth exponential KL decay so PPO stays anchored early then surpasses BC."""
    if base_kl <= 0.0:
        return 0.0
    if half_life <= 0.0:
        return float(base_kl)
    decay = 0.5 ** (max(0, int(update)) / float(half_life))
    return float(base_kl * (floor_ratio + (1.0 - floor_ratio) * decay))


def masked_kl(dist, prior_logits, mask, direction: str = "reverse"):
    """Exact masked KL divergence on legal actions (reverse=D_KL(PPO||BC), forward=D_KL(BC||PPO))."""
    prior_logp = F.log_softmax(prior_logits.masked_fill(~mask, -1e9), dim=-1)
    policy_logp = F.log_softmax(dist.logits.masked_fill(~mask, -1e9), dim=-1)
    if direction == "forward":
        prior_probs = prior_logp.exp()
        return (prior_probs * (prior_logp - policy_logp)).masked_fill(~mask, 0.0).sum(dim=-1).mean()
    policy_probs = policy_logp.exp()
    return (policy_probs * (policy_logp - prior_logp)).masked_fill(~mask, 0.0).sum(dim=-1).mean()


def stratified_ppo_subsample(data, max_samples: int = 8192, rng=None):
    """Cap CPU PPO backprop frames after exact full-trajectory GAE, prioritizing card plays (up to 50%)."""
    if max_samples <= 0 or len(data[3]) <= max_samples:
        return data
    if rng is None:
        rng = np.random.default_rng()
    acts_np = data[3].detach().cpu().numpy()
    play_idx = np.flatnonzero(acts_np != 0)
    wait_idx = np.flatnonzero(acts_np == 0)
    n_play = min(len(play_idx), max_samples // 2)
    n_wait = min(len(wait_idx), max_samples - n_play)
    sel_play = rng.choice(play_idx, size=n_play, replace=False) if n_play < len(play_idx) else play_idx
    sel_wait = rng.choice(wait_idx, size=n_wait, replace=False) if n_wait < len(wait_idx) else wait_idx
    keep_idx = torch.from_numpy(np.sort(np.concatenate([sel_play, sel_wait]))).to(data[3].device)
    return tuple(t[keep_idx] if t is not None else None for t in data)


def optimize_ppo(policy, reference, optimizer, data, batch_size, updates, kl_coef, *,
                 kl_direction: str = "reverse", target_kl: float | None = 0.03,
                 ent_coef: float = 0.005, wait_ent_coef: float = 0.0,
                 return_stats: bool = False, reference_runner=None):
    """Clipped PPO with masked reverse/forward KL regularization and full telemetry."""
    if len(data) == 8:
        spatial, vector, mask, actions, old_logp, id_planes, advantage, returns = data
    else:
        spatial, vector, mask, actions, old_logp, advantage, returns = data
        id_planes = None
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    # The BC reference is frozen. Reuse its predictions in float16 (2x less RAM)
    # for subsequent epochs, or precompute them on DirectML GPU when reference_runner is provided.
    reference_logits = None
    if (reference is not None and kl_coef > 0 and actions.device.type == "cpu"
            and (updates > 1 or reference_runner is not None)
            and mask.numel() * 2 <= PPO_REFERENCE_CACHE_BYTES):
        reference_logits = torch.empty(mask.shape, dtype=torch.float16, device=actions.device)
        if reference_runner is not None:
            ref_bs = 256
            n_total = len(actions)
            for s in range(0, n_total, ref_bs):
                e = min(n_total, s + ref_bs)
                ref_id_np = id_planes[s:e].numpy() if id_planes is not None else None
                ref_l = reference_runner.logits_only(
                    spatial[s:e].numpy(), vector[s:e].numpy(), mask[s:e].numpy(), ref_id_np,
                )
                reference_logits[s:e] = torch.from_numpy(ref_l).clamp(-1e4, 1e4).to(torch.float16)
    reference_precomputed = reference_logits is not None and reference_runner is not None
    divergences = []
    policy_losses, value_losses, entropies, wait_entropies = [], [], [], []
    approx_kls, clip_fractions = [], []
    initial_values = torch.empty(len(actions), dtype=torch.float32, device=actions.device)
    epochs_completed = 0
    for epoch in range(updates):
        order = torch.randperm(len(actions), device=actions.device)
        epoch_kls = []
        for batch_i, ids in enumerate(order.split(batch_size)):
            if batch_i % 8 == 0:
                check_training_stop()
            batch_ids = id_planes[ids] if id_planes is not None else None
            batch_mask = mask[ids]
            inputs = (spatial[ids].float(), vector[ids].float(), batch_mask, batch_ids)
            dist, value = policy(*inputs)
            if epoch == 0:
                initial_values[ids] = value.detach()
            log_ratio = dist.log_prob(actions[ids]) - old_logp[ids]
            ratio = log_ratio.exp()
            unclipped = ratio * advantage[ids]
            clipped = ratio.clamp(0.8, 1.2) * advantage[ids]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * (returns[ids] - value).square().mean()
            entropy = dist.entropy().mean()
            p_wait = dist.probs[:, 0].clamp(1e-6, 1.0 - 1e-6)
            wait_entropy = -(p_wait * p_wait.log() + (1.0 - p_wait) * (1.0 - p_wait).log()).mean()
            loss = policy_loss + value_loss - ent_coef * entropy
            if wait_ent_coef > 0.0:
                loss = loss - wait_ent_coef * wait_entropy
            if reference is not None and kl_coef > 0:
                with torch.no_grad():
                    if reference_logits is not None and (reference_precomputed or epoch > 0):
                        logits = reference_logits[ids].float()
                    else:
                        logits, _ = reference.logits_and_value(*inputs)
                        if reference_logits is not None:
                            reference_logits[ids] = logits.clamp(-1e4, 1e4).to(torch.float16)
                divergence = masked_kl(dist, logits, batch_mask, direction=kl_direction)
                loss = loss + kl_coef * divergence
                divergences.append(float(divergence.detach().item()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                step_kl = float(((ratio - 1.0) - log_ratio).mean().item())
                epoch_kls.append(step_kl)
                approx_kls.append(step_kl)
                clip_fractions.append(float(((ratio - 1.0).abs() > 0.2).float().mean().item()))
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))
                wait_entropies.append(float(wait_entropy.detach().item()))
        epochs_completed = epoch + 1
        if target_kl is not None and target_kl > 0 and epoch_kls and float(np.mean(epoch_kls)) > 1.5 * target_kl:
            break
    kl_mean = float(np.mean(divergences)) if divergences else None
    if not return_stats:
        return kl_mean
    ret_var = float(returns.var(unbiased=False).item())
    explained_var = (
        float(1.0 - (returns - initial_values).var(unbiased=False).item() / ret_var)
        if ret_var > 1e-8 else 0.0
    )
    stats = {
        "policy_loss": round(float(np.mean(policy_losses)), 5) if policy_losses else 0.0,
        "value_loss": round(float(np.mean(value_losses)), 5) if value_losses else 0.0,
        "entropy": round(float(np.mean(entropies)), 4) if entropies else 0.0,
        "wait_entropy": round(float(np.mean(wait_entropies)), 4) if wait_entropies else 0.0,
        "approx_kl": round(float(np.mean(approx_kls)), 5) if approx_kls else 0.0,
        "clip_fraction": round(float(np.mean(clip_fractions)), 4) if clip_fractions else 0.0,
        "explained_variance": round(explained_var, 4),
        "kl_coef": round(float(kl_coef), 5),
        "kl_direction": kl_direction,
        "epochs_completed": epochs_completed,
    }
    return kl_mean, stats


def frozen_policy(policy):
    return deepcopy(policy).eval().requires_grad_(False)
