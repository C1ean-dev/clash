"""Synthetic Clash Royale arena generator & continual Ultralytics YOLO fine-tuning pipeline.

CLI subcommands:
  - download-public     Download pre-trained public Clash Royale ONNX YOLO weights (`units_M_480x352.onnx`, `side.onnx`)
  - generate-synthetic  Synthesize YOLO-labeled Clash Royale arena frames (using real harvested backgrounds/sprites + KataCR + procedural fallback)
  - train               Merge synthetic + live auto-collected frames and fine-tune `models/vision/custom_yolo.pt`
  - cycle               Run generate-synthetic + train in one step
  - status              Show installed vision weights, dataset inventory, and latest metrics
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys

import cv2
import numpy as np
import yaml

from xbow_capture_bridge import REFERENCE_HEIGHT, REFERENCE_WIDTH
from xbow_replays import DECKS
from xbow_vision_yolo import (
    CANONICAL_UNIT_SLUGS,
    CUSTOM_YOLO_NAME,
    DEFAULT_VISION_DATASET_DIR,
    DEFAULT_VISION_MODELS_DIR,
    INSTANT_SPELL_SLUGS,
    PUBLIC_ONNX_SIDE_NAME,
    PUBLIC_ONNX_UNITS_NAME,
    YOLO_CLASS_NAMES,
    YOLO_CLASS_TO_ID,
    build_vision_detector,
    ensure_public_yolo_weights,
    normalize_unit_slug,
    tile_to_portrait_px,
    yolo_class_id,
)

BUILDING_SLUGS = frozenset({
    "barbarian-hut", "bomb-tower", "cannon", "elixir-collector", "furnace",
    "goblin-cage", "goblin-drill", "goblin-hut", "inferno-tower", "mortar",
    "tesla", "tombstone", "x-bow",
})
AIR_SLUGS = frozenset({
    "baby-dragon", "balloon", "bats", "electro-dragon", "flying-machine",
    "inferno-dragon", "lava-hound", "mega-minion", "minions", "phoenix",
    "skeleton-dragons",
})
TANK_SLUGS = frozenset({
    "electro-giant", "giant", "giant-skeleton", "golem", "lava-hound",
    "mega-knight", "pekka", "royal-giant", "sparky", "skeleton-king",
})
SWARM_COUNTS: dict[str, int] = {
    "archers": 2,
    "bats": 3,
    "barbarians": 3,
    "elite-barbarians": 2,
    "goblins": 3,
    "guards": 3,
    "minions": 3,
    "rascals": 2,
    "royal-hogs": 3,
    "royal-recruits": 3,
    "skeleton-dragons": 2,
    "skeletons": 3,
    "spear-goblins": 3,
    "wall-breakers": 2,
    "zappies": 3,
}

# High-value meta units prioritized alongside the chosen deck
META_PRIORITY_SLUGS: tuple[str, ...] = (
    "x-bow", "tesla", "knight", "archers", "skeletons", "electro-spirit",
    "ice-spirit", "ice-wizard", "hog-rider", "musketeer", "cannon", "ice-golem",
    "royal-giant", "hunter", "fisherman", "royal-ghost", "baby-dragon",
    "tombstone", "mega-minion", "valkyrie", "balloon", "inferno-tower",
    "bats", "miner", "mortar", "witch", "mini-pekka", "pekka", "mega-knight",
    "goblin-cage", "goblins", "firecracker", "dart-goblin", "princess",
    "bandit", "dark-prince", "prince", "wizard", "giant", "golem", "lumberjack",
)


def _deterministic_color(slug: str, team: str) -> tuple[int, int, int]:
    """Stable BGR accent color per unit class so procedural sprites have distinctive visual identity."""
    digest = hashlib.md5(slug.encode("utf-8")).digest()
    r = 70 + (digest[0] % 155)
    g = 70 + (digest[1] % 155)
    b = 70 + (digest[2] % 155)
    if team == "enemy":
        r = min(245, int(r * 1.15 + 25))
        b = max(35, int(b * 0.72))
    else:
        b = min(245, int(b * 1.15 + 25))
        r = max(35, int(r * 0.78))
    return (b, g, r)


def _render_procedural_arena(rng: random.Random, w: int = REFERENCE_WIDTH, h: int = REFERENCE_HEIGHT) -> np.ndarray:
    """Render a clean synthetic Clash Royale portrait arena matching REFERENCE_WIDTH x REFERENCE_HEIGHT."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    # Dark outer / HUD backdrop
    canvas[:, :] = (28, 34, 42)

    # Arena grass palette variation (classic green, royale, snowy, desert)
    palettes = [
        ((76, 152, 86), (68, 140, 78), (195, 145, 78)),
        ((70, 146, 98), (62, 134, 88), (186, 136, 70)),
        ((96, 148, 108), (88, 138, 100), (182, 140, 82)),
        ((98, 138, 152), (88, 128, 142), (175, 132, 74)),
    ]
    c1, c2, river_c = rng.choice(palettes)

    # Draw 18x32 checkerboard tiles
    tile_w = w / 18.0
    arena_top = int(round(h * 0.08))
    arena_bot = int(round(h * 0.80))
    tile_h = (arena_bot - arena_top) / 32.0
    for ty in range(32):
        y0 = int(round(arena_top + ty * tile_h))
        y1 = int(round(arena_top + (ty + 1) * tile_h))
        for tx in range(18):
            x0 = int(round(tx * tile_w))
            x1 = int(round((tx + 1) * tile_w))
            col = c1 if (tx + ty) % 2 == 0 else c2
            canvas[y0:y1, x0:x1] = col

    # River across center (tiles 15..17)
    ry0 = int(round(arena_top + 15 * tile_h))
    ry1 = int(round(arena_top + 17 * tile_h))
    canvas[ry0:ry1, :] = river_c
    # Subtle river sheen lines
    for _ in range(14):
        rx = rng.randint(10, w - 50)
        ry = rng.randint(ry0 + 3, max(ry0 + 4, ry1 - 4))
        cv2.line(canvas, (rx, ry), (rx + rng.randint(18, 46), ry), (230, 185, 115), 1)

    # Wooden bridges at x=3.5 and x=14.5
    for bx_tile in (3.5, 14.5):
        bx0 = int(round((bx_tile - 1.0) * tile_w))
        bx1 = int(round((bx_tile + 1.0) * tile_w))
        cv2.rectangle(canvas, (bx0, ry0 - 4), (bx1, ry1 + 4), (66, 102, 146), -1)
        cv2.rectangle(canvas, (bx0, ry0 - 4), (bx1, ry1 + 4), (42, 68, 102), 2)
        for plank in range(ry0, ry1, 6):
            cv2.line(canvas, (bx0 + 2, plank), (bx1 - 2, plank), (52, 82, 122), 1)

    # Stone lane paths
    for lx_tile in (3.5, 14.5):
        lx0 = int(round((lx_tile - 0.55) * tile_w))
        lx1 = int(round((lx_tile + 0.55) * tile_w))
        cv2.rectangle(canvas, (lx0, arena_top + 48), (lx1, ry0 - 4), (118, 166, 152), -1)
        cv2.rectangle(canvas, (lx0, ry1 + 4), (lx1, arena_bot - 48), (118, 166, 152), -1)

    # Enemy & Ally Princess + King towers
    def _draw_tower(cx: int, cy: int, rw: int, rh: int, team: str):
        base = (78, 72, 175) if team == "enemy" else (175, 105, 58)
        roof = (60, 56, 215) if team == "enemy" else (215, 130, 48)
        cv2.rectangle(canvas, (cx - rw, cy - rh), (cx + rw, cy + rh), (130, 134, 142), -1)
        cv2.rectangle(canvas, (cx - rw + 4, cy - rh + 4), (cx + rw - 4, cy + rh - 4), base, -1)
        cv2.rectangle(canvas, (cx - rw + 8, cy - rh - 5), (cx + rw - 8, cy - rh + 10), roof, -1)
        cv2.rectangle(canvas, (cx - rw, cy - rh), (cx + rw, cy + rh), (40, 40, 45), 2)

    _draw_tower(int(w * 0.21), int(h * 0.215), 24, 24, "enemy")
    _draw_tower(int(w * 0.79), int(h * 0.215), 24, 24, "enemy")
    _draw_tower(int(w * 0.50), int(h * 0.145), 32, 30, "enemy")
    _draw_tower(int(w * 0.21), int(h * 0.655), 24, 24, "ally")
    _draw_tower(int(w * 0.79), int(h * 0.655), 24, 24, "ally")
    _draw_tower(int(w * 0.50), int(h * 0.725), 32, 30, "ally")

    # Bottom HUD deck bar placeholder
    cv2.rectangle(canvas, (0, arena_bot), (w, h), (42, 38, 36), -1)
    cv2.rectangle(canvas, (int(w * 0.22), int(h * 0.945)), (int(w * 0.94), int(h * 0.972)), (185, 50, 195), -1)
    for s in range(4):
        sx0 = int(w * (0.21 + s * 0.19))
        cv2.rectangle(canvas, (sx0, int(h * 0.81)), (sx0 + int(w * 0.16), int(h * 0.93)), (78, 86, 98), -1)
        cv2.rectangle(canvas, (sx0, int(h * 0.81)), (sx0 + int(w * 0.16), int(h * 0.93)), (28, 32, 38), 2)
    return canvas


def _draw_hp_bar(canvas: np.ndarray, cx: int, top_y: int, bw: int, team: str, hp_frac: float) -> None:
    """Draw a realistic Clash Royale unit HP bar above a unit."""
    bar_w = max(20, min(58, int(bw * 0.85)))
    bar_h = 5
    x0 = max(2, cx - bar_w // 2)
    y0 = max(2, top_y - 9)
    x1 = min(canvas.shape[1] - 2, x0 + bar_w)
    y1 = min(canvas.shape[0] - 2, y0 + bar_h)
    if x1 <= x0 + 2 or y1 <= y0 + 1:
        return
    cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), (18, 18, 22), -1)
    fill_x = x0 + max(1, int(round((x1 - x0) * max(0.08, min(1.0, hp_frac)))))
    color = (62, 62, 232) if team == "enemy" else (238, 168, 48)
    cv2.rectangle(canvas, (x0, y0), (fill_x, y1), color, -1)


def _render_procedural_unit_sprite(slug: str, team: str, rng: random.Random) -> np.ndarray:
    """Generate an RGBA sprite patch for a unit when no real harvested crop is available yet."""
    is_building = slug in BUILDING_SLUGS
    is_tank = slug in TANK_SLUGS
    is_air = slug in AIR_SLUGS

    if is_building:
        bw, bh = rng.randint(48, 66), rng.randint(50, 68)
    elif is_tank:
        bw, bh = rng.randint(52, 74), rng.randint(58, 80)
    elif slug in SWARM_COUNTS:
        bw, bh = rng.randint(24, 34), rng.randint(28, 40)
    else:
        bw, bh = rng.randint(34, 48), rng.randint(40, 56)

    rgba = np.zeros((bh, bw, 4), dtype=np.uint8)
    accent = _deterministic_color(slug, team)
    team_col = (58, 66, 225) if team == "enemy" else (228, 148, 46)
    cx, cy = bw // 2, bh // 2

    # Ground shadow ellipse
    shadow_y = bh - max(4, bh // 8)
    cv2.ellipse(rgba, (cx, shadow_y), (max(6, bw // 3), max(3, bh // 10)), 0, 0, 360, (18, 22, 18, 130), -1)

    if is_building:
        pad = 6
        cv2.rectangle(rgba, (pad, pad + 4), (bw - pad, bh - 7), (105, 108, 114, 255), -1)
        cv2.rectangle(rgba, (pad + 4, pad + 8), (bw - pad - 4, bh - 11), (*accent, 255), -1)
        cv2.circle(rgba, (cx, cy), max(6, min(bw, bh) // 4), (*team_col, 255), -1)
        cv2.rectangle(rgba, (pad, pad + 4), (bw - pad, bh - 7), (28, 30, 34, 255), 2)
    else:
        body_y = cy - (5 if is_air else 0)
        # Torso / armor
        cv2.ellipse(rgba, (cx, body_y + 4), (max(6, bw // 3), max(7, bh // 3 - 2)), 0, 0, 360, (*accent, 255), -1)
        # Team shoulder / cloak trim
        cv2.ellipse(rgba, (cx, body_y + 6), (max(5, bw // 4), max(4, bh // 5)), 0, 0, 360, (*team_col, 255), -1)
        # Head / helmet
        head_r = max(4, min(bw, bh) // 5)
        head_y = max(head_r + 2, body_y - bh // 5)
        cv2.circle(rgba, (cx, head_y), head_r, (185, 200, 228, 255), -1)
        cv2.circle(rgba, (cx, head_y), head_r, (30, 32, 36, 255), 1)
        if is_air:
            # Wings
            cv2.ellipse(rgba, (cx - bw // 3, body_y), (max(5, bw // 5), max(3, bh // 7)), -20, 0, 360, (*accent, 230), -1)
            cv2.ellipse(rgba, (cx + bw // 3, body_y), (max(5, bw // 5), max(3, bh // 7)), 20, 0, 360, (*accent, 230), -1)
        else:
            # Weapon glint
            wx = cx + (bw // 3 if rng.random() > 0.5 else -bw // 3)
            cv2.line(rgba, (cx, body_y + 2), (wx, max(3, body_y - bh // 4)), (210, 218, 226, 255), 2)

    return rgba


def _load_sprite_bank(sprites_dir: Path, katacr_dir: Path | None = None) -> dict[str, list[Path]]:
    """Index all available sprite crops keyed by `ally_<slug>` / `enemy_<slug>` (and raw `<slug>`)."""
    bank: dict[str, list[Path]] = {}
    if sprites_dir.is_dir():
        for sub in sprites_dir.iterdir():
            if not sub.is_dir():
                continue
            files = sorted(sub.glob("*.png"))
            if files:
                bank[sub.name.lower()] = files

    # Optional KataCR Clash-Royale-Detection-Dataset segment sprites
    if katacr_dir is not None and Path(katacr_dir).is_dir():
        seg_root = Path(katacr_dir) / "images" / "segment"
        search_root = seg_root if seg_root.is_dir() else Path(katacr_dir)
        for png in search_root.rglob("*.png"):
            parent = png.parent.name.lower()
            slug = normalize_unit_slug(parent)
            if not slug or slug not in CANONICAL_UNIT_SLUGS:
                continue
            if parent.endswith("_1") or "enemy" in parent or "red" in parent:
                key = f"enemy_{slug}"
            elif parent.endswith("_0") or "ally" in parent or "blue" in parent:
                key = f"ally_{slug}"
            else:
                key = slug
            bank.setdefault(key, []).append(png)
    return bank


def _load_backgrounds(bg_dir: Path, katacr_dir: Path | None = None) -> list[Path]:
    paths: list[Path] = []
    if bg_dir.is_dir():
        paths.extend(sorted(p for p in bg_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")))
    if katacr_dir is not None and Path(katacr_dir).is_dir():
        for pat in ("*background*.jpg", "*background*.png"):
            paths.extend(sorted(Path(katacr_dir).rglob(pat)))
    return paths


def _paste_sprite(canvas: np.ndarray, sprite: np.ndarray, cx: int, cy: int) -> list[float] | None:
    """Alpha-blend an RGBA or feathered BGR sprite onto `canvas` centered at `(cx, cy)`."""
    ch, cw = canvas.shape[:2]
    sh, sw = sprite.shape[:2]
    if sh < 4 or sw < 4:
        return None
    if sprite.shape[2] == 3:
        # Create soft feathered alpha mask for harvested rectangular live crops
        alpha = np.full((sh, sw), 240, dtype=np.uint8)
        border = max(2, min(sh, sw) // 9)
        for i in range(border):
            val = int(240 * (i + 1) / (border + 1))
            alpha[i, :] = np.minimum(alpha[i, :], val)
            alpha[sh - 1 - i, :] = np.minimum(alpha[sh - 1 - i, :], val)
            alpha[:, i] = np.minimum(alpha[:, i], val)
            alpha[:, sw - 1 - i] = np.minimum(alpha[:, sw - 1 - i], val)
        bgr = sprite
    else:
        bgr = sprite[:, :, :3]
        alpha = sprite[:, :, 3]

    x0 = cx - sw // 2
    y0 = cy - sh // 2
    x1, y1 = x0 + sw, y0 + sh

    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(cw, x1), min(ch, y1)
    sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)
    if dx1 <= dx0 + 4 or dy1 <= dy0 + 4:
        return None

    a = (alpha[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0)[..., None]
    roi = canvas[dy0:dy1, dx0:dx1].astype(np.float32)
    fg = bgr[sy0:sy1, sx0:sx1].astype(np.float32)
    canvas[dy0:dy1, dx0:dx1] = np.clip(fg * a + roi * (1.0 - a), 0, 255).astype(np.uint8)
    return [float(dx0), float(dy0), float(dx1), float(dy1)]


def _apply_domain_jitter(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply subtle real-world capture jitter (brightness, contrast, slight blur, spell glow)."""
    out = img.astype(np.float32)
    alpha = rng.uniform(0.88, 1.12)
    beta = rng.uniform(-14.0, 14.0)
    out = np.clip(out * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    if rng.random() < 0.22:
        # Occasional translucent spell impact ring (Fireball / Poison / Rage)
        h, w = out.shape[:2]
        sx, sy = rng.randint(60, w - 60), rng.randint(int(h * 0.16), int(h * 0.72))
        rad = rng.randint(24, 58)
        overlay = out.copy()
        spell_col = rng.choice([(40, 130, 250), (60, 210, 90), (220, 90, 220), (240, 210, 110)])
        cv2.circle(overlay, (sx, sy), rad, spell_col, -1)
        out = cv2.addWeighted(overlay, 0.22, out, 0.78, 0)
    return out


def generate_synthetic_dataset(
    dataset_root: Path = DEFAULT_VISION_DATASET_DIR,
    *,
    count: int = 300,
    val_ratio: float = 0.15,
    deck: str = "xbow_30",
    all_classes: bool = False,
    katacr_dir: Path | None = None,
    seed: int = 42,
    clean: bool = True,
) -> dict:
    """Generate a KataCR-style synthetic YOLO dataset under `<dataset_root>/synthetic/`."""
    dataset_root = Path(dataset_root)
    syn_dir = dataset_root / "synthetic"
    if clean and syn_dir.exists():
        shutil.rmtree(syn_dir)

    for split in ("train", "val"):
        (syn_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (syn_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    sprite_bank = _load_sprite_bank(dataset_root / "sprites", katacr_dir=katacr_dir)
    backgrounds = _load_backgrounds(dataset_root / "backgrounds", katacr_dir=katacr_dir)
    bg_cache: list[np.ndarray] = []
    for p in backgrounds[:25]:
        im = cv2.imread(str(p))
        if im is not None:
            bg_cache.append(cv2.resize(im, (REFERENCE_WIDTH, REFERENCE_HEIGHT), interpolation=cv2.INTER_AREA))

    # Build weighted unit sampling pool
    deck_units = [
        normalize_unit_slug(c) for c in DECKS.get(deck, ())
        if c not in INSTANT_SPELL_SLUGS and normalize_unit_slug(c) in CANONICAL_UNIT_SLUGS
    ]
    harvested_slugs = sorted({
        normalize_unit_slug(k) for k in sprite_bank.keys()
        if normalize_unit_slug(k) in CANONICAL_UNIT_SLUGS
    })
    if all_classes:
        pool = list(CANONICAL_UNIT_SLUGS)
    else:
        pool = sorted(set(deck_units) | set(META_PRIORITY_SLUGS) | set(harvested_slugs))
        pool = [s for s in pool if s in CANONICAL_UNIT_SLUGS]

    n_val = max(1, int(round(count * max(0.05, min(0.4, val_ratio)))))
    n_train = max(1, count - n_val)
    total_boxes = 0
    real_sprite_uses = 0
    real_bg_uses = 0

    for idx in range(count):
        split = "val" if idx >= n_train else "train"
        if bg_cache and rng.random() < 0.78:
            canvas = rng.choice(bg_cache).copy()
            real_bg_uses += 1
        else:
            canvas = _render_procedural_arena(rng)

        h, w = canvas.shape[:2]
        n_groups = rng.randint(2, 8)
        labels: list[str] = []
        drawn: list[tuple[int, int, str, str, np.ndarray, bool]] = []

        for _ in range(n_groups):
            # Prioritize deck cards for ally side and diverse meta pool for enemy side
            team = "ally" if rng.random() < 0.45 else "enemy"
            if team == "ally" and deck_units and rng.random() < 0.72:
                slug = rng.choice(deck_units)
            elif harvested_slugs and rng.random() < 0.35:
                slug = rng.choice(harvested_slugs)
            else:
                slug = rng.choice(pool)

            is_building = slug in BUILDING_SLUGS
            if is_building:
                tx = rng.uniform(3.0, 15.0)
                ty = rng.uniform(5.5, 14.5) if team == "ally" else rng.uniform(17.5, 26.5)
            else:
                # Bias along the two lanes + occasional center kites
                if rng.random() < 0.75:
                    lane = 3.5 if rng.random() < 0.5 else 14.5
                    tx = max(1.2, min(16.8, rng.gauss(lane, 2.0)))
                else:
                    tx = rng.uniform(1.5, 16.5)
                ty = rng.uniform(4.5, 25.5)

            n_members = SWARM_COUNTS.get(slug, 1)
            if n_members > 1:
                n_members = rng.randint(1, n_members)

            for m in range(n_members):
                mtx = max(1.0, min(17.0, tx + (rng.uniform(-1.1, 1.1) if m > 0 else 0.0)))
                mty = max(4.0, min(27.0, ty + (rng.uniform(-1.1, 1.1) if m > 0 else 0.0)))
                px, py = tile_to_portrait_px(mtx, mty, w, h)

                # Prefer real harvested sprites when available
                key_team = f"{team}_{slug}"
                candidates = sprite_bank.get(key_team) or sprite_bank.get(slug)
                used_real = False
                if candidates and rng.random() < 0.85:
                    sp = cv2.imread(str(rng.choice(candidates)), cv2.IMREAD_UNCHANGED)
                    if sp is not None and sp.ndim == 3 and sp.shape[2] in (3, 4):
                        scale = rng.uniform(0.84, 1.18)
                        nw = max(16, int(round(sp.shape[1] * scale)))
                        nh = max(16, int(round(sp.shape[0] * scale)))
                        sp = cv2.resize(sp, (nw, nh), interpolation=cv2.INTER_AREA)
                        if rng.random() < 0.5:
                            sp = cv2.flip(sp, 1)
                        used_real = True
                    else:
                        sp = _render_procedural_unit_sprite(slug, team, rng)
                else:
                    sp = _render_procedural_unit_sprite(slug, team, rng)
                drawn.append((int(round(py)), int(round(px)), slug, team, sp, used_real))

        # Painter's algorithm: draw top-to-bottom so closer units naturally occlude farther ones
        drawn.sort(key=lambda item: item[0])
        for py, px, slug, team, sp, used_real in drawn:
            cid = yolo_class_id(team, slug)
            if cid is None:
                continue
            box = _paste_sprite(canvas, sp, px, py)
            if box is None:
                continue
            if used_real:
                real_sprite_uses += 1
            elif rng.random() < 0.85:
                _draw_hp_bar(canvas, int(round((box[0] + box[2]) * 0.5)), int(round(box[1])),
                             int(round(box[2] - box[0])), team, rng.uniform(0.18, 1.0))
            cx = ((box[0] + box[2]) * 0.5) / w
            cy = ((box[1] + box[3]) * 0.5) / h
            bw = (box[2] - box[0]) / w
            bh = (box[3] - box[1]) / h
            labels.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            total_boxes += 1

        canvas = _apply_domain_jitter(canvas, rng)
        stem = f"syn_{idx:05d}"
        cv2.imwrite(str(syn_dir / "images" / split / f"{stem}.jpg"), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        (syn_dir / "labels" / split / f"{stem}.txt").write_text(
            ("\n".join(labels) + "\n") if labels else "", encoding="utf-8"
        )

    yaml_data = {
        "path": str(syn_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(YOLO_CLASS_NAMES)},
    }
    (syn_dir / "dataset.yaml").write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": count,
        "train_images": n_train,
        "val_images": n_val,
        "total_boxes": total_boxes,
        "real_background_frames": real_bg_uses,
        "real_sprite_instances": real_sprite_uses,
        "sprite_classes_indexed": len(sprite_bank),
        "backgrounds_indexed": len(bg_cache),
        "dataset_yaml": str((syn_dir / "dataset.yaml").resolve()),
    }
    (syn_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_merged_dataset(
    dataset_root: Path = DEFAULT_VISION_DATASET_DIR,
    *,
    live_val_ratio: float = 0.15,
    live_oversample: int = 3,
    seed: int = 42,
) -> dict:
    """Merge `<dataset_root>/synthetic/` and `<dataset_root>/live/` into `<dataset_root>/merged/`."""
    dataset_root = Path(dataset_root)
    merged = dataset_root / "merged"
    if merged.exists():
        shutil.rmtree(merged)
    for split in ("train", "val"):
        (merged / "images" / split).mkdir(parents=True, exist_ok=True)
        (merged / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"syn_train": 0, "syn_val": 0, "live_train": 0, "live_val": 0}
    syn_dir = dataset_root / "synthetic"
    for split, key in (("train", "syn_train"), ("val", "syn_val")):
        img_dir = syn_dir / "images" / split
        lbl_dir = syn_dir / "labels" / split
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.glob("*.jpg")):
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.is_file():
                continue
            _link_or_copy(img, merged / "images" / split / f"s_{img.name}")
            _link_or_copy(lbl, merged / "labels" / split / f"s_{lbl.name}")
            counts[key] += 1

    live_img_dir = dataset_root / "live" / "images"
    live_lbl_dir = dataset_root / "live" / "labels"
    live_pairs: list[tuple[Path, Path]] = []
    if live_img_dir.is_dir():
        for img in sorted(live_img_dir.glob("*.jpg")):
            lbl = live_lbl_dir / f"{img.stem}.txt"
            if lbl.is_file() and lbl.stat().st_size > 0:
                live_pairs.append((img, lbl))

    rng = random.Random(seed)
    rng.shuffle(live_pairs)
    n_live_val = int(round(len(live_pairs) * live_val_ratio)) if len(live_pairs) >= 6 else 0
    live_val = live_pairs[:n_live_val]
    live_train = live_pairs[n_live_val:]

    for img, lbl in live_val:
        _link_or_copy(img, merged / "images" / "val" / f"l_{img.name}")
        _link_or_copy(lbl, merged / "labels" / "val" / f"l_{lbl.name}")
        counts["live_val"] += 1

    reps = max(1, int(live_oversample))
    for img, lbl in live_train:
        # Verified own-play frames get an extra boost over pseudo-labeled live frames
        is_verified = img.stem.startswith("verified_")
        mult = reps + (1 if is_verified else 0)
        for r in range(mult):
            suffix = f"_r{r}" if r > 0 else ""
            _link_or_copy(img, merged / "images" / "train" / f"l_{img.stem}{suffix}{img.suffix}")
            _link_or_copy(lbl, merged / "labels" / "train" / f"l_{lbl.stem}{suffix}{lbl.suffix}")
            counts["live_train"] += 1

    yaml_data = {
        "path": str(merged.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(YOLO_CLASS_NAMES)},
    }
    yaml_path = merged / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")
    return {
        "dataset_yaml": str(yaml_path.resolve()),
        "counts": counts,
        "total_train": counts["syn_train"] + counts["live_train"],
        "total_val": counts["syn_val"] + counts["live_val"],
    }


def train_vision_yolo(
    dataset_root: Path = DEFAULT_VISION_DATASET_DIR,
    models_dir: Path = DEFAULT_VISION_MODELS_DIR,
    *,
    base_model: str = "yolov8n.pt",
    from_scratch: bool = False,
    epochs: int = 12,
    imgsz: int = 480,
    batch: int = 8,
    device: str = "auto",
    live_oversample: int = 3,
    min_promote_map50: float = 0.45,
    force_promote: bool = False,
) -> dict:
    """Continually fine-tune Ultralytics YOLO on merged synthetic + live data and promote `custom_yolo.pt`."""
    import torch
    from ultralytics import YOLO

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    merged_info = build_merged_dataset(dataset_root, live_oversample=live_oversample)
    if merged_info["total_train"] < 4 or merged_info["total_val"] < 1:
        raise RuntimeError(
            "Dataset insuficiente para treinar YOLO "
            f"(train={merged_info['total_train']}, val={merged_info['total_val']}). "
            "Execute `generate-synthetic` ou colete quadros ao vivo primeiro."
        )

    custom_pt = models_dir / CUSTOM_YOLO_NAME
    metrics_path = models_dir / "metrics.json"
    if not from_scratch and custom_pt.is_file():
        init_weights = str(custom_pt)
    else:
        base_path = Path(base_model)
        if not base_path.is_absolute() and len(base_path.parts) == 1:
            init_weights = str(models_dir / base_path.name)
        else:
            init_weights = str(base_path)
    resolved_device = ("0" if torch.cuda.is_available() else "cpu") if device == "auto" else device

    model = YOLO(init_weights)
    run_name = f"yolo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results = model.train(
        data=merged_info["dataset_yaml"],
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        device=resolved_device,
        project=str((Path(dataset_root) / "train_runs").resolve()),
        name=run_name,
        exist_ok=True,
        workers=0,
        pretrained=True,
        mosaic=0.35,
        fliplr=0.5,
        flipud=0.0,  # Never flip ally/enemy vertically!
        degrees=0.0,
        verbose=False,
    )

    save_dir = Path(getattr(results, "save_dir", Path(dataset_root) / "train_runs" / run_name))
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.is_file():
        best_pt = save_dir / "weights" / "last.pt"
    if not best_pt.is_file():
        raise RuntimeError(f"Treino finalizado sem weights em {save_dir}")

    results_dict = getattr(results, "results_dict", {}) or {}
    map50 = float(results_dict.get("metrics/mAP50(B)", 0.0) or 0.0)
    map50_95 = float(results_dict.get("metrics/mAP50-95(B)", 0.0) or 0.0)
    precision = float(results_dict.get("metrics/precision(B)", 0.0) or 0.0)
    recall = float(results_dict.get("metrics/recall(B)", 0.0) or 0.0)

    prev_map50 = -1.0
    if metrics_path.is_file():
        try:
            prev_map50 = float(json.loads(metrics_path.read_text(encoding="utf-8")).get("map50", -1.0))
        except Exception:
            prev_map50 = -1.0

    meets_floor = map50 >= float(min_promote_map50)
    beats_prev = (not custom_pt.is_file()) or (map50 >= prev_map50 - 0.015)
    promoted = force_promote or (meets_floor and beats_prev)
    if promoted:
        if custom_pt.is_file():
            shutil.copy2(custom_pt, models_dir / "custom_yolo.bak.pt")
        tmp_dst = custom_pt.with_suffix(".pt.tmp")
        shutil.copy2(best_pt, tmp_dst)
        tmp_dst.replace(custom_pt)

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "init_weights": init_weights,
        "best_weights": str(best_pt),
        "promoted": promoted,
        "active_custom_model": str(custom_pt) if custom_pt.is_file() else None,
        "device": resolved_device,
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "previous_map50": round(prev_map50, 4) if prev_map50 >= 0 else None,
        "dataset": merged_info["counts"],
    }
    if promoted or not metrics_path.is_file():
        metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def vision_status(
    dataset_root: Path = DEFAULT_VISION_DATASET_DIR,
    models_dir: Path = DEFAULT_VISION_MODELS_DIR,
) -> dict:
    dataset_root = Path(dataset_root)
    models_dir = Path(models_dir)
    pub_units = models_dir / PUBLIC_ONNX_UNITS_NAME
    pub_side = models_dir / PUBLIC_ONNX_SIDE_NAME
    custom_pt = models_dir / CUSTOM_YOLO_NAME
    metrics_path = models_dir / "metrics.json"
    stats_path = dataset_root / "stats.json"

    live_imgs = len(list((dataset_root / "live" / "images").glob("*.jpg"))) if (dataset_root / "live" / "images").is_dir() else 0
    syn_train = len(list((dataset_root / "synthetic" / "images" / "train").glob("*.jpg"))) if (dataset_root / "synthetic" / "images" / "train").is_dir() else 0
    syn_val = len(list((dataset_root / "synthetic" / "images" / "val").glob("*.jpg"))) if (dataset_root / "synthetic" / "images" / "val").is_dir() else 0
    sprites = sum(1 for _ in (dataset_root / "sprites").rglob("*.png")) if (dataset_root / "sprites").is_dir() else 0
    backgrounds = len(list((dataset_root / "backgrounds").glob("*.jpg"))) if (dataset_root / "backgrounds").is_dir() else 0
    review = len(list((dataset_root / "review" / "images").glob("*.jpg"))) if (dataset_root / "review" / "images").is_dir() else 0

    return {
        "models": {
            "public_onnx_units": str(pub_units) if pub_units.is_file() else None,
            "public_onnx_side": str(pub_side) if pub_side.is_file() else None,
            "custom_yolo_pt": str(custom_pt) if custom_pt.is_file() else None,
            "active_auto": str(custom_pt if custom_pt.is_file() else (pub_units if pub_units.is_file() else "hp_bar_only")),
        },
        "dataset": {
            "live_frames": live_imgs,
            "synthetic_train": syn_train,
            "synthetic_val": syn_val,
            "harvested_sprites": sprites,
            "harvested_backgrounds": backgrounds,
            "review_queue": review,
        },
        "collector_stats": json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else None,
        "latest_train_metrics": json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_VISION_DATASET_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_VISION_MODELS_DIR)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download-public", help="Baixar pesos YOLO ONNX públicos de Clash Royale")
    p_dl.add_argument("--force", action="store_true")
    p_dl.add_argument("--test-image", type=Path, default=Path("calibration/preview.png"))

    p_gen = sub.add_parser("generate-synthetic", help="Gerar dataset sintético de arena no formato YOLO")
    p_gen.add_argument("--count", type=int, default=300)
    p_gen.add_argument("--val-ratio", type=float, default=0.15)
    p_gen.add_argument("--deck", choices=DECKS, default="xbow_30")
    p_gen.add_argument("--all-classes", action="store_true")
    p_gen.add_argument("--katacr-dir", type=Path, default=None)
    p_gen.add_argument("--seed", type=int, default=42)

    p_tr = sub.add_parser("train", help="Mesclar sintético + ao vivo e treinar/atualizar custom_yolo.pt")
    p_tr.add_argument("--base-model", default="yolov8n.pt")
    p_tr.add_argument("--from-scratch", action="store_true")
    p_tr.add_argument("--epochs", type=int, default=12)
    p_tr.add_argument("--imgsz", type=int, default=480)
    p_tr.add_argument("--batch", type=int, default=8)
    p_tr.add_argument("--device", default="auto")
    p_tr.add_argument("--live-oversample", type=int, default=3)
    p_tr.add_argument("--force-promote", action="store_true")

    p_cyc = sub.add_parser("cycle", help="Gerar sintético + treinar + promover custom_yolo.pt")
    p_cyc.add_argument("--synthetic-count", type=int, default=300)
    p_cyc.add_argument("--deck", choices=DECKS, default="xbow_30")
    p_cyc.add_argument("--katacr-dir", type=Path, default=None)
    p_cyc.add_argument("--epochs", type=int, default=12)
    p_cyc.add_argument("--imgsz", type=int, default=480)
    p_cyc.add_argument("--batch", type=int, default=8)
    p_cyc.add_argument("--device", default="auto")
    p_cyc.add_argument("--force-promote", action="store_true")

    sub.add_parser("status", help="Exibir estado dos pesos e do dataset de visão")

    args = parser.parse_args()
    if args.cmd == "download-public":
        paths = ensure_public_yolo_weights(args.models_dir, force=args.force)
        result = {"downloaded": {k: str(v) for k, v in paths.items()}}
        if args.test_image and args.test_image.is_file():
            img = cv2.imread(str(args.test_image))
            if img is not None:
                portrait = cv2.resize(img, (REFERENCE_WIDTH, REFERENCE_HEIGHT), interpolation=cv2.INTER_AREA)
                det = build_vision_detector("public", deck="xbow_30", models_dir=args.models_dir)
                if det is not None:
                    preds = det.detect(portrait)
                    result["smoke_test"] = {
                        "image": str(args.test_image),
                        "backend": det.backend_name,
                        "detections": len(preds),
                        "sample": preds[:5],
                    }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.cmd == "generate-synthetic":
        info = generate_synthetic_dataset(
            args.dataset_dir,
            count=args.count,
            val_ratio=args.val_ratio,
            deck=args.deck,
            all_classes=args.all_classes,
            katacr_dir=args.katacr_dir,
            seed=args.seed,
        )
        print(json.dumps(info, indent=2, ensure_ascii=False))
    elif args.cmd == "train":
        report = train_vision_yolo(
            args.dataset_dir,
            args.models_dir,
            base_model=args.base_model,
            from_scratch=args.from_scratch,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            live_oversample=args.live_oversample,
            force_promote=args.force_promote,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.cmd == "cycle":
        gen = generate_synthetic_dataset(
            args.dataset_dir,
            count=args.synthetic_count,
            deck=args.deck,
            katacr_dir=args.katacr_dir,
        )
        tr = train_vision_yolo(
            args.dataset_dir,
            args.models_dir,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            force_promote=args.force_promote,
        )
        print(json.dumps({"synthetic": gen, "train": tr}, indent=2, ensure_ascii=False))
    elif args.cmd == "status":
        print(json.dumps(vision_status(args.dataset_dir, args.models_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
