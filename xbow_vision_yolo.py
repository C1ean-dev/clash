"""Public Clash Royale YOLO detector (ONNX / Ultralytics), hybrid HP-bar fusion, and live auto-collector.

Supports three seamless vision backends:
  1. Custom fine-tuned Ultralytics YOLO weights (`models/vision/custom_yolo.pt`)
  2. Public pre-trained Clash Royale ONNX YOLO weights (`models/vision/units_M_480x352.onnx` + `side.onnx`,
     auto-downloaded on demand from Pbatch/ClashRoyaleBuildABot)
  3. Graceful fallback to OpenCV HP-bar unit tracking when no weights are installed

Also provides `VisionAutoCollector`, which runs during live matches to automatically:
  - Build ground-truth YOLO labels from confirmed own card plays (`record_own_play`)
  - Collect high-confidence pseudo-labeled frames from the active YOLO detector
  - Harvest clean isolated unit sprites and empty arena backgrounds for synthetic YOLO training
  - Queue uncertain / HP-bar-missed frames into `review/` for active improvement
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time
from typing import Callable
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image

from xbow_capture_bridge import GRID, REFERENCE_HEIGHT, REFERENCE_WIDTH
from xbow_replays import DECKS

DEFAULT_VISION_MODELS_DIR = Path("models/vision")
DEFAULT_VISION_DATASET_DIR = Path("runs/vision_dataset")

PUBLIC_ONNX_UNITS_NAME = "units_M_480x352.onnx"
PUBLIC_ONNX_SIDE_NAME = "side.onnx"
CUSTOM_YOLO_NAME = "custom_yolo.pt"

PUBLIC_MODEL_URLS = {
    PUBLIC_ONNX_UNITS_NAME: (
        "https://raw.githubusercontent.com/Pbatch/ClashRoyaleBuildABot/"
        "main/clashroyalebuildabot/models/units_M_480x352.onnx"
    ),
    PUBLIC_ONNX_SIDE_NAME: (
        "https://raw.githubusercontent.com/Pbatch/ClashRoyaleBuildABot/"
        "main/clashroyalebuildabot/models/side.onnx"
    ),
}

# Exact 96-class index order used by Pbatch/ClashRoyaleBuildABot units_M_480x352.onnx
CRBAB_DETECTOR_UNITS: tuple[str, ...] = (
    "archer",
    "archer_queen",
    "balloon",
    "bandit",
    "barbarian",
    "barbarian_hut",
    "bat",
    "battle_healer",
    "battle_ram",
    "bomb_tower",
    "bomber",
    "bowler",
    "brawler",
    "cannon",
    "cannon_cart",
    "dark_prince",
    "dart_goblin",
    "electro_dragon",
    "electro_giant",
    "electro_spirit",
    "electro_wizard",
    "elite_barbarian",
    "elixir_collector",
    "elixir_golem_large",
    "elixir_golem_medium",
    "elixir_golem_small",
    "executioner",
    "fire_spirit",
    "fire_cracker",
    "fisherman",
    "flying_machine",
    "furnace",
    "giant",
    "giant_skeleton",
    "giant_snowball",
    "goblin",
    "goblin_cage",
    "goblin_drill",
    "goblin_hut",
    "golden_knight",
    "golem",
    "golemite",
    "guard",
    "heal_spirit",
    "hog",
    "hog_rider",
    "baby_dragon",
    "hunter",
    "ice_golem",
    "ice_spirit",
    "ice_wizard",
    "inferno_dragon",
    "inferno_tower",
    "knight",
    "lava_hound",
    "lava_pup",
    "little_prince",
    "lumberjack",
    "magic_archer",
    "mega_knight",
    "mega_minion",
    "mighty_miner",
    "miner",
    "minion",
    "minipekka",
    "monk",
    "mortar",
    "mother_witch",
    "musketeer",
    "night_witch",
    "pekka",
    "phoenix_egg",
    "phoenix_large",
    "phoenix_small",
    "prince",
    "princess",
    "ram_rider",
    "rascal_boy",
    "rascal_girl",
    "royal_ghost",
    "royal_giant",
    "royal_guardian",
    "royal_hog",
    "royal_recruit",
    "skeleton",
    "skeleton_dragon",
    "skeleton_king",
    "sparky",
    "spear_goblin",
    "tesla",
    "tombstone",
    "valkyrie",
    "wall_breaker",
    "witch",
    "wizard",
    "x_bow",
    "zappy",
)

# Normalize detector unit names (CRBAB / KataCR / Roboflow) to RoyaleSim CARD_NAMES slugs.
UNIT_TO_CARD_SLUG: dict[str, str] = {
    "archer": "archers",
    "archers": "archers",
    "barbarian": "barbarians",
    "barbarians": "barbarians",
    "bat": "bats",
    "bats": "bats",
    "brawler": "goblin-cage",
    "elite_barbarian": "elite-barbarians",
    "elite-barbarian": "elite-barbarians",
    "elixir_golem_large": "elixir-golem",
    "elixir_golem_medium": "elixir-golem",
    "elixir_golem_small": "elixir-golem",
    "elixir-golem-large": "elixir-golem",
    "elixir-golem-medium": "elixir-golem",
    "elixir-golem-small": "elixir-golem",
    "fire_cracker": "firecracker",
    "fire-cracker": "firecracker",
    "goblin": "goblins",
    "goblins": "goblins",
    "golemite": "golem",
    "guard": "guards",
    "guards": "guards",
    "hog": "royal-hogs",
    "lava_pup": "lava-hound",
    "lava-pup": "lava-hound",
    "log": "the-log",
    "minion": "minions",
    "minions": "minions",
    "minipekka": "mini-pekka",
    "mini_pekka": "mini-pekka",
    "phoenix_egg": "phoenix",
    "phoenix_large": "phoenix",
    "phoenix_small": "phoenix",
    "phoenix-egg": "phoenix",
    "phoenix-large": "phoenix",
    "phoenix-small": "phoenix",
    "rascal_boy": "rascals",
    "rascal_girl": "rascals",
    "rascal-boy": "rascals",
    "rascal-girl": "rascals",
    "royal_guardian": "little-prince",
    "royal-guardian": "little-prince",
    "royal_hog": "royal-hogs",
    "royal-hog": "royal-hogs",
    "royal_recruit": "royal-recruits",
    "royal-recruit": "royal-recruits",
    "skeleton": "skeletons",
    "skeletons": "skeletons",
    "skeleton_dragon": "skeleton-dragons",
    "skeleton-dragon": "skeleton-dragons",
    "spear_goblin": "spear-goblins",
    "spear-goblin": "spear-goblins",
    "wall_breaker": "wall-breakers",
    "wall-breaker": "wall-breakers",
    "xbow": "x-bow",
    "x_bow": "x-bow",
    "zappy": "zappies",
    "zappies": "zappies",
}

INSTANT_SPELL_SLUGS: frozenset[str] = frozenset({
    "arrows", "clone", "earthquake", "fireball", "freeze", "giant-snowball",
    "lightning", "mirror", "poison", "rage", "rocket", "the-log", "tornado",
    "vines", "zap", "goblin-curse", "barbarian-barrel", "royal-delivery",
})

# Cards that spawn different unit slugs than their own card slug
DECK_CARD_SPAWN_UNITS: dict[str, tuple[str, ...]] = {
    "barbarian-barrel": ("barbarians",),
    "goblin-barrel": ("goblins",),
    "goblin-gang": ("goblins", "spear-goblins"),
    "graveyard": ("skeletons",),
    "minion-horde": ("minions",),
    "royal-delivery": ("royal-recruits",),
    "skeleton-army": ("skeletons",),
    "skeleton-barrel": ("skeletons",),
    "three-musketeers": ("musketeer",),
    "tombstone": ("tombstone", "skeletons"),
    "witch": ("witch", "skeletons"),
    "night-witch": ("night-witch", "bats"),
    "barbarian-hut": ("barbarian-hut", "barbarians"),
    "goblin-hut": ("goblin-hut", "spear-goblins"),
    "goblin-cage": ("goblin-cage",),
    "goblin-drill": ("goblin-drill", "goblins"),
    "furnace": ("furnace", "fire-spirit"),
    "battle-ram": ("battle-ram", "barbarians"),
}


def normalize_unit_slug(raw_label: str) -> str:
    """Normalize any detector label (e.g. 'ally_x_bow', 'MiniPekka_1') to a canonical RoyaleSim slug."""
    if not raw_label:
        return ""
    s = str(raw_label).strip().lower()
    s = re.sub(r"^(ally|enemy|blue|red)[_\-\s]+", "", s)
    s = re.sub(r"[_\-\s]+(0|1|ally|enemy|blue|red)$", "", s)
    s = re.sub(r"[\s_]+", "-", s).strip("-")
    if s in UNIT_TO_CARD_SLUG:
        return UNIT_TO_CARD_SLUG[s]
    underscored = s.replace("-", "_")
    if underscored in UNIT_TO_CARD_SLUG:
        return UNIT_TO_CARD_SLUG[underscored]
    return s


def _build_canonical_unit_slugs() -> tuple[str, ...]:
    slugs = {normalize_unit_slug(u) for u in CRBAB_DETECTOR_UNITS}
    for cards in DECKS.values():
        for c in cards:
            if c not in INSTANT_SPELL_SLUGS:
                slugs.add(normalize_unit_slug(c))
            for spawned in DECK_CARD_SPAWN_UNITS.get(c, ()):
                slugs.add(normalize_unit_slug(spawned))
    slugs.discard("")
    return tuple(sorted(slugs))


CANONICAL_UNIT_SLUGS: tuple[str, ...] = _build_canonical_unit_slugs()
YOLO_CLASS_NAMES: tuple[str, ...] = tuple(
    f"{team}_{slug}" for slug in CANONICAL_UNIT_SLUGS for team in ("ally", "enemy")
)
YOLO_CLASS_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(YOLO_CLASS_NAMES)}


def yolo_class_id(team: str, slug: str) -> int | None:
    norm_team = "enemy" if str(team).lower() in ("enemy", "opponent", "red", "1") else "ally"
    norm_slug = normalize_unit_slug(slug)
    return YOLO_CLASS_TO_ID.get(f"{norm_team}_{norm_slug}")


def possible_ally_slugs_for_deck(deck: str | None) -> frozenset[str] | None:
    """Return all canonical unit slugs that our own deck can legally spawn on the arena."""
    if not deck or deck not in DECKS:
        return None
    allowed: set[str] = set()
    for card in DECKS[deck]:
        if card not in INSTANT_SPELL_SLUGS:
            allowed.add(normalize_unit_slug(card))
        for spawned in DECK_CARD_SPAWN_UNITS.get(card, ()):
            allowed.add(normalize_unit_slug(spawned))
    return frozenset(allowed)


def tile_to_portrait_px(tile_x: float, tile_y: float, width: int = REFERENCE_WIDTH,
                        height: int = REFERENCE_HEIGHT) -> tuple[float, float]:
    """Project arena tile (0..18, 0..32) to portrait pixel coordinates."""
    x0, y0, x1, y1 = GRID
    ref_x = x0 + (float(tile_x) / 18.0) * (x1 - x0)
    ref_y = y1 - (float(tile_y) / 32.0) * (y1 - y0)
    return ref_x * (width / REFERENCE_WIDTH), ref_y * (height / REFERENCE_HEIGHT)


def ensure_public_yolo_weights(
    models_dir: Path = DEFAULT_VISION_MODELS_DIR,
    *,
    force: bool = False,
    timeout_s: float = 60.0,
) -> dict[str, Path]:
    """Download public pre-trained Clash Royale ONNX YOLO weights (`units_M_480x352.onnx` + `side.onnx`)."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for filename, url in PUBLIC_MODEL_URLS.items():
        dst = models_dir / filename
        if not force and dst.is_file() and dst.stat().st_size > 10_000:
            paths[filename] = dst
            continue
        tmp = dst.with_suffix(dst.suffix + ".part")
        req = Request(url, headers={"User-Agent": "xbow-vision-yolo/1.0"})
        with urlopen(req, timeout=timeout_s) as resp:
            data = resp.read()
        if len(data) < 10_000:
            raise RuntimeError(f"Download inválido para {filename}: apenas {len(data)} bytes.")
        tmp.write_bytes(data)
        tmp.replace(dst)
        paths[filename] = dst
    return paths


def resolve_vision_model_path(
    spec: str | Path | None,
    models_dir: Path = DEFAULT_VISION_MODELS_DIR,
    *,
    auto_download: bool = False,
) -> Path | None:
    """Resolve 'auto', 'public', 'custom', 'none', or an explicit file path to a detector weight file."""
    if spec is None:
        return None
    text = str(spec).strip()
    if not text or text.lower() in ("none", "off", "false", "hp", "hp_only"):
        return None
    models_dir = Path(models_dir)
    custom_pt = models_dir / CUSTOM_YOLO_NAME
    public_onnx = models_dir / PUBLIC_ONNX_UNITS_NAME

    low = text.lower()
    if low == "custom":
        return custom_pt if custom_pt.is_file() else None
    if low == "public":
        if public_onnx.is_file():
            return public_onnx
        if auto_download:
            return ensure_public_yolo_weights(models_dir)[PUBLIC_ONNX_UNITS_NAME]
        return None
    if low == "auto":
        if custom_pt.is_file():
            return custom_pt
        if public_onnx.is_file():
            return public_onnx
        if auto_download:
            try:
                return ensure_public_yolo_weights(models_dir)[PUBLIC_ONNX_UNITS_NAME]
            except Exception:
                return None
        return None

    path = Path(text)
    return path if path.is_file() else None


class OnnxSideClassifier:
    """Lightweight 16x16 ally/enemy crop classifier (`side.onnx`)."""

    SIDE_SIZE = 16

    def __init__(self, model_path: Path):
        import onnxruntime as ort

        providers = [
            p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()
        ] or ["CPUExecutionProvider"]
        self.model_path = Path(model_path)
        self.sess = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def classify_rgb(self, rgb_image: Image.Image, bbox: tuple[int, int, int, int]) -> tuple[str, float]:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(rgb_image.width - 1, int(x1)))
        y1 = max(0, min(rgb_image.height - 1, int(y1)))
        x2 = max(x1 + 1, min(rgb_image.width, int(x2)))
        y2 = max(y1 + 1, min(rgb_image.height, int(y2)))
        crop = rgb_image.crop((x1, y1, x2, y2)).resize(
            (self.SIDE_SIZE, self.SIDE_SIZE), Image.Resampling.BICUBIC
        )
        arr = (np.asarray(crop, dtype=np.float32) / 255.0)[None, ...]
        logits = self.sess.run([self.output_name], {self.input_name: arr})[0][0]
        # Softmax confidence for side score
        shifted = logits - float(np.max(logits))
        probs = np.exp(shifted)
        probs = probs / max(1e-8, float(np.sum(probs)))
        idx = int(np.argmax(probs))
        return ("ally", "enemy")[idx], float(probs[idx])


class PublicOnnxYoloDetector:
    """Pre-trained Clash Royale ONNX YOLO detector (`units_M_480x352.onnx` + optional `side.onnx`)."""

    UNIT_Y_START = 0.05
    UNIT_Y_END = 0.80

    def __init__(
        self,
        weights_path: Path,
        *,
        side_weights_path: Path | None = None,
        deck: str | None = None,
        conf: float = 0.38,
    ):
        import onnxruntime as ort

        self.weights_path = Path(weights_path)
        providers = [
            p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()
        ] or ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(self.weights_path), providers=providers)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.input_dtype = np.float16 if "16" in str(inp.type).lower() else np.float32
        self.model_height = int(inp.shape[2])
        self.model_width = int(inp.shape[3])
        self.output_name = self.sess.get_outputs()[0].name
        self.conf = float(conf)
        self.possible_ally_slugs = possible_ally_slugs_for_deck(deck)

        if side_weights_path is None:
            candidate = self.weights_path.parent / PUBLIC_ONNX_SIDE_NAME
            if candidate.is_file():
                side_weights_path = candidate
        self.side_classifier = (
            OnnxSideClassifier(side_weights_path)
            if side_weights_path is not None and Path(side_weights_path).is_file()
            else None
        )

    def _resize_pad(self, rgb_crop: Image.Image) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        ratio = rgb_crop.height / max(1, rgb_crop.width)
        if ratio > self.model_height / self.model_width:
            new_h = self.model_height
            new_w = max(1, int(self.model_height / ratio))
        else:
            new_w = self.model_width
            new_h = max(1, int(self.model_width * ratio))
        resized = rgb_crop.resize((new_w, new_h), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=self.input_dtype)
        dx = self.model_width - new_w
        dy = self.model_height - new_h
        pad_right = dx // 2
        pad_left = dx - pad_right
        pad_bottom = dy // 2
        pad_top = dy - pad_bottom
        padded = np.pad(
            arr,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=114,
        )
        tensor = padded.transpose(2, 0, 1) / self.input_dtype(255.0)
        return tensor[None, ...], (pad_left, pad_right, pad_top, pad_bottom)

    def detect(self, portrait_bgr: np.ndarray) -> list[dict]:
        h, w = portrait_bgr.shape[:2]
        rgb_full = Image.fromarray(cv2.cvtColor(portrait_bgr, cv2.COLOR_BGR2RGB))
        y_top = int(round(self.UNIT_Y_START * h))
        y_bot = int(round(self.UNIT_Y_END * h))
        crop_rgb = rgb_full.crop((0, y_top, w, y_bot))

        tensor, (pad_l, pad_r, pad_t, pad_b) = self._resize_pad(crop_rgb)
        raw = self.sess.run([self.output_name], {self.input_name: tensor})[0]
        pred = raw[0] if raw.ndim == 3 else raw
        if pred.size == 0 or pred.shape[-1] < 6:
            return []

        pred = np.asarray(pred, dtype=np.float32)
        pred = pred[pred[:, 4] >= self.conf]
        if pred.size == 0:
            return []

        eff_w = max(1.0, float(self.model_width - pad_l - pad_r))
        eff_h = max(1.0, float(self.model_height - pad_t - pad_b))
        crop_h = float(y_bot - y_top)

        pred[:, [0, 2]] = (pred[:, [0, 2]] - pad_l) * (w / eff_w)
        pred[:, [1, 3]] = (pred[:, [1, 3]] - pad_t) * (crop_h / eff_h) + y_top

        detections: list[dict] = []
        for row in pred:
            x1, y1, x2, y2, conf, cls_f = [float(v) for v in row[:6]]
            cls_idx = int(round(cls_f))
            if not (0 <= cls_idx < len(CRBAB_DETECTOR_UNITS)):
                continue
            raw_name = CRBAB_DETECTOR_UNITS[cls_idx]
            slug = normalize_unit_slug(raw_name)
            x1 = max(0.0, min(float(w - 1), x1))
            y1 = max(0.0, min(float(h - 1), y1))
            x2 = max(x1 + 1.0, min(float(w), x2))
            y2 = max(y1 + 1.0, min(float(h), y2))

            side_conf = 0.85
            if self.possible_ally_slugs is not None and slug not in self.possible_ally_slugs:
                team = "enemy"
                side_conf = 0.99
            elif self.side_classifier is not None:
                team, side_conf = self.side_classifier.classify_rgb(
                    rgb_full, (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
                )
            else:
                team = "enemy" if (y1 + y2) * 0.5 < h * 0.44 else "ally"
                side_conf = 0.6

            # Ground anchor sits near lower center of unit bounding box
            cx = (x1 + x2) * 0.5
            cy = y1 + (y2 - y1) * 0.72
            detections.append({
                "class_name": slug,
                "raw_class": raw_name,
                "team": team,
                "center_px": [round(cx, 1), round(cy, 1)],
                "bbox_px": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "conf": round(conf, 4),
                "side_conf": round(side_conf, 4),
                "estimated_hp": 1.0,
                "source": "public_onnx_yolo",
            })
        return detections


class UltralyticsYoloDetector:
    """Ultralytics YOLOv8/YOLO11 detector (`.pt` or `.onnx` custom/KataCR/Roboflow weights)."""

    def __init__(
        self,
        weights_path: Path,
        *,
        deck: str | None = None,
        conf: float = 0.32,
        side_weights_path: Path | None = None,
    ):
        from ultralytics import YOLO

        self.weights_path = Path(weights_path)
        self.model = YOLO(str(self.weights_path))
        self.conf = float(conf)
        self.possible_ally_slugs = possible_ally_slugs_for_deck(deck)
        if side_weights_path is None:
            candidate = self.weights_path.parent / PUBLIC_ONNX_SIDE_NAME
            if candidate.is_file():
                side_weights_path = candidate
        self.side_classifier = (
            OnnxSideClassifier(side_weights_path)
            if side_weights_path is not None and Path(side_weights_path).is_file()
            else None
        )

    @staticmethod
    def _parse_team_from_label(label: str) -> str | None:
        low = label.lower().strip()
        if re.search(r"(^|[_\-\s])(enemy|red|opp|opponent)($|[_\-\s])", low) or low.endswith("_1"):
            return "enemy"
        if re.search(r"(^|[_\-\s])(ally|blue|own|self)($|[_\-\s])", low) or low.endswith("_0"):
            return "ally"
        return None

    def detect(self, portrait_bgr: np.ndarray) -> list[dict]:
        results = self.model.predict(portrait_bgr, conf=self.conf, verbose=False)
        if not results:
            return []
        r0 = results[0]
        names = r0.names or {}
        rgb_full = None
        h, w = portrait_bgr.shape[:2]
        detections: list[dict] = []
        for box in r0.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = str(names.get(cls_id, ""))
            slug = normalize_unit_slug(label)
            if not slug:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            x1 = max(0.0, min(float(w - 1), x1))
            y1 = max(0.0, min(float(h - 1), y1))
            x2 = max(x1 + 1.0, min(float(w), x2))
            y2 = max(y1 + 1.0, min(float(h), y2))

            team = self._parse_team_from_label(label)
            if self.possible_ally_slugs is not None and slug not in self.possible_ally_slugs:
                team = "enemy"
            elif team is None:
                if self.side_classifier is not None:
                    if rgb_full is None:
                        rgb_full = Image.fromarray(cv2.cvtColor(portrait_bgr, cv2.COLOR_BGR2RGB))
                    team, _ = self.side_classifier.classify_rgb(
                        rgb_full, (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
                    )
                else:
                    team = "enemy" if (y1 + y2) * 0.5 < h * 0.44 else "ally"

            cx = (x1 + x2) * 0.5
            cy = y1 + (y2 - y1) * 0.72
            detections.append({
                "class_name": slug,
                "raw_class": label,
                "team": team,
                "center_px": [round(cx, 1), round(cy, 1)],
                "bbox_px": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "conf": round(conf, 4),
                "estimated_hp": 1.0,
                "source": "custom_yolo",
            })
        return detections


class HybridVisionDetector:
    """Combines a YOLO detector (public ONNX or custom Ultralytics) with OpenCV HP-bar fusion.

    Benefits:
      - YOLO gives exact unit classes and tight bounding boxes
      - HP-bar fusion attaches real `estimated_hp` and cross-checks ally/enemy team colors
      - Unmatched HP bars remain visible as fallback AND signal missed detections for active learning
      - Supports hot-reloading whenever `models/vision/custom_yolo.pt` is updated on disk
    """

    def __init__(
        self,
        weights_path: Path,
        *,
        deck: str | None = None,
        conf: float = 0.30,
        hp_bar_fn: Callable[[np.ndarray], list[dict]] | None = None,
        watch_custom_path: Path | None = None,
    ):
        self.weights_path = Path(weights_path)
        self.deck = deck
        self.conf = float(conf)
        self.hp_bar_fn = hp_bar_fn
        self.watch_custom_path = Path(watch_custom_path) if watch_custom_path is not None else None
        self._loaded_mtime: float = 0.0
        self.backend_name: str = "none"
        self.primary = self._load_backend(self.weights_path)
        self.last_hp_bars: list[dict] = []
        self.last_unmatched_hp_bars: list[dict] = []

    def _load_backend(self, path: Path):
        self.weights_path = Path(path)
        self._loaded_mtime = self.weights_path.stat().st_mtime if self.weights_path.is_file() else 0.0
        if self.weights_path.suffix.lower() == ".onnx" and "units_" in self.weights_path.name.lower():
            self.backend_name = f"public_onnx:{self.weights_path.name}"
            return PublicOnnxYoloDetector(self.weights_path, deck=self.deck, conf=self.conf)
        self.backend_name = f"ultralytics:{self.weights_path.name}"
        return UltralyticsYoloDetector(self.weights_path, deck=self.deck, conf=self.conf)

    def reload_if_newer(self) -> bool:
        """Hot-reload if a newer fine-tuned `custom_yolo.pt` appeared or the active weight changed."""
        target = self.watch_custom_path if (self.watch_custom_path and self.watch_custom_path.is_file()) else self.weights_path
        if not target.is_file():
            return False
        mtime = target.stat().st_mtime
        if target != self.weights_path or mtime > self._loaded_mtime + 1e-3:
            try:
                self.primary = self._load_backend(target)
                return True
            except Exception:
                return False
        return False

    def detect(self, portrait_bgr: np.ndarray) -> list[dict]:
        yolo_dets = self.primary.detect(portrait_bgr)
        hp_bars = self.hp_bar_fn(portrait_bgr) if self.hp_bar_fn is not None else []
        self.last_hp_bars = hp_bars
        used_hp: set[int] = set()

        # Fuse HP bars onto matching YOLO boxes for exact HP fraction & team verification
        for det in yolo_dets:
            bx1, by1, bx2, by2 = det["bbox_px"]
            bcx = (bx1 + bx2) * 0.5
            best_idx = -1
            best_dist = 1e9
            for idx, hp in enumerate(hp_bars):
                if idx in used_hp:
                    continue
                hx, hy = hp["center_px"]
                dx = abs(hx - bcx)
                dy = hy - by1
                if dx <= max(32.0, (bx2 - bx1) * 0.75) and -38.0 <= dy <= max(36.0, (by2 - by1) * 0.75):
                    dist = math.hypot(dx, dy)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx
            if best_idx >= 0:
                used_hp.add(best_idx)
                hp_match = hp_bars[best_idx]
                det["estimated_hp"] = float(hp_match.get("estimated_hp", 1.0))
                det["hp_verified"] = True
                # Trust HP bar team color when allowed by deck constraints
                hp_team = hp_match.get("team")
                if hp_team in ("ally", "enemy"):
                    allowed = getattr(self.primary, "possible_ally_slugs", None)
                    if hp_team == "enemy" or allowed is None or det["class_name"] in allowed:
                        det["team"] = hp_team

        unmatched = [hp for idx, hp in enumerate(hp_bars) if idx not in used_hp]
        self.last_unmatched_hp_bars = unmatched

        # Keep unmatched HP bars as fallback so units not yet in YOLO are still visible to the policy
        fused = list(yolo_dets)
        for hp in unmatched:
            row = dict(hp)
            row["source"] = "hp_bar_fallback"
            fused.append(row)
        return fused


def build_vision_detector(
    spec: str | Path | None,
    *,
    deck: str | None = None,
    conf: float = 0.38,
    models_dir: Path = DEFAULT_VISION_MODELS_DIR,
    hp_bar_fn: Callable[[np.ndarray], list[dict]] | None = None,
    auto_download: bool = False,
) -> HybridVisionDetector | None:
    """Factory that builds a HybridVisionDetector from 'auto', 'public', 'custom', or a file path."""
    resolved = resolve_vision_model_path(spec, models_dir=models_dir, auto_download=auto_download)
    if resolved is None:
        return None
    watch_custom = (Path(models_dir) / CUSTOM_YOLO_NAME) if str(spec).strip().lower() in ("auto", "custom") else None
    return HybridVisionDetector(
        resolved,
        deck=deck,
        conf=conf,
        hp_bar_fn=hp_bar_fn,
        watch_custom_path=watch_custom,
    )


def _box_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1e-6, area_a + area_b - inter)


@dataclass
class _PendingOwnPlay:
    card: str
    tile_x: float
    tile_y: float
    px_x: float
    px_y: float
    ts: float
    matched: bool = False


class VisionAutoCollector:
    """Automatic live-match dataset, sprite, background, and hard-example collector for YOLO fine-tuning."""

    def __init__(
        self,
        root_dir: Path = DEFAULT_VISION_DATASET_DIR,
        *,
        min_frame_interval_s: float = 1.6,
        pseudo_conf_threshold: float = 0.52,
        uncertain_conf_low: float = 0.22,
        max_live_frames: int = 2500,
        max_sprites_per_class: int = 80,
        max_backgrounds: int = 40,
        max_review_frames: int = 600,
    ):
        self.root_dir = Path(root_dir)
        self.live_img_dir = self.root_dir / "live" / "images"
        self.live_lbl_dir = self.root_dir / "live" / "labels"
        self.review_img_dir = self.root_dir / "review" / "images"
        self.review_meta_dir = self.root_dir / "review" / "meta"
        self.sprites_dir = self.root_dir / "sprites"
        self.backgrounds_dir = self.root_dir / "backgrounds"
        self.stats_path = self.root_dir / "stats.json"

        for d in (
            self.live_img_dir, self.live_lbl_dir,
            self.review_img_dir, self.review_meta_dir,
            self.sprites_dir, self.backgrounds_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        self.min_frame_interval_s = float(min_frame_interval_s)
        self.pseudo_conf_threshold = float(pseudo_conf_threshold)
        self.uncertain_conf_low = float(uncertain_conf_low)
        self.max_live_frames = int(max_live_frames)
        self.max_sprites_per_class = int(max_sprites_per_class)
        self.max_backgrounds = int(max_backgrounds)
        self.max_review_frames = int(max_review_frames)

        self._last_saved_ts: float = -1e9
        self._last_review_ts: float = -1e9
        self._last_bg_ts: float = -1e9
        self._seq: int = 0
        self._pending_plays: deque[_PendingOwnPlay] = deque(maxlen=16)
        self.stats = self._load_stats()

    def _load_stats(self) -> dict:
        base = {
            "frames_saved": len(list(self.live_img_dir.glob("*.jpg"))),
            "verified_play_frames": 0,
            "verified_play_labels": 0,
            "pseudo_labels": 0,
            "hard_examples": len(list(self.review_img_dir.glob("*.jpg"))),
            "sprites_harvested": sum(1 for _ in self.sprites_dir.rglob("*.png")),
            "backgrounds_harvested": len(list(self.backgrounds_dir.glob("*.jpg"))),
            "updated_at": None,
        }
        if self.stats_path.is_file():
            try:
                saved = json.loads(self.stats_path.read_text(encoding="utf-8"))
                base.update(saved)
            except Exception:
                pass
        return base

    def _save_stats(self) -> None:
        self.stats["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp = self.stats_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.stats, indent=2), encoding="utf-8")
        tmp.replace(self.stats_path)

    def reset_match(self) -> None:
        self._pending_plays.clear()

    def record_own_play(self, card: str, tile_x: float, tile_y: float, timestamp_s: float | None = None) -> None:
        """Register an own card play so the next matching spawn becomes a verified ground-truth label."""
        slug = normalize_unit_slug(card)
        if not slug or slug in INSTANT_SPELL_SLUGS:
            return
        px_x, px_y = tile_to_portrait_px(tile_x, tile_y)
        self._pending_plays.append(_PendingOwnPlay(
            card=slug,
            tile_x=float(tile_x),
            tile_y=float(tile_y),
            px_x=px_x,
            px_y=px_y,
            ts=time.time() if timestamp_s is None else float(timestamp_s),
        ))

    def _next_stem(self, prefix: str) -> str:
        self._seq += 1
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{stamp}_{self._seq:05d}"

    def _write_yolo_sample(self, portrait_bgr: np.ndarray, boxes: list[tuple[int, list[float]]],
                           prefix: str) -> bool:
        if not boxes or self.stats["frames_saved"] >= self.max_live_frames:
            return False
        h, w = portrait_bgr.shape[:2]
        lines = []
        for cls_id, (x1, y1, x2, y2) in boxes:
            cx = max(0.0, min(1.0, ((x1 + x2) * 0.5) / w))
            cy = max(0.0, min(1.0, ((y1 + y2) * 0.5) / h))
            bw = max(1e-4, min(1.0, (x2 - x1) / w))
            bh = max(1e-4, min(1.0, (y2 - y1) / h))
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        stem = self._next_stem(prefix)
        img_path = self.live_img_dir / f"{stem}.jpg"
        lbl_path = self.live_lbl_dir / f"{stem}.txt"
        if not cv2.imwrite(str(img_path), portrait_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
            return False
        lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.stats["frames_saved"] += 1
        return True

    def _harvest_sprites(self, portrait_bgr: np.ndarray, dets: list[dict]) -> int:
        h, w = portrait_bgr.shape[:2]
        harvested = 0
        all_boxes = [d["bbox_px"] for d in dets if "bbox_px" in d]
        for idx, d in enumerate(dets):
            if "bbox_px" not in d:
                continue
            conf = float(d.get("conf", 0.0))
            verified = bool(d.get("verified_play", False))
            if not verified and conf < max(0.65, self.pseudo_conf_threshold):
                continue
            box = d["bbox_px"]
            bw, bh = box[2] - box[0], box[3] - box[1]
            if bw < 16 or bh < 16 or bw > w * 0.45 or bh > h * 0.40:
                continue
            # Only harvest clean isolated sprites (no heavy overlap with neighboring units)
            if any(j != idx and _box_iou(box, other) > 0.08 for j, other in enumerate(all_boxes)):
                continue
            cls_name = f"{d['team']}_{normalize_unit_slug(d['class_name'])}"
            if cls_name not in YOLO_CLASS_TO_ID:
                continue
            cls_dir = self.sprites_dir / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            existing = len(list(cls_dir.glob("*.png")))
            if existing >= self.max_sprites_per_class:
                continue
            x1 = max(0, int(math.floor(box[0])))
            y1 = max(0, int(math.floor(box[1])))
            x2 = min(w, int(math.ceil(box[2])))
            y2 = min(h, int(math.ceil(box[3])))
            crop = portrait_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            out = cls_dir / f"{self._next_stem('spr')}.png"
            if cv2.imwrite(str(out), crop):
                harvested += 1
                self.stats["sprites_harvested"] += 1
        return harvested

    def observe_frame(
        self,
        portrait_bgr: np.ndarray,
        detections: list[dict],
        *,
        unmatched_hp_bars: list[dict] | None = None,
        elapsed_s: float = 0.0,
        timestamp_s: float | None = None,
    ) -> dict:
        """Process one in-match portrait frame for auto-labeling, sprite harvesting, and review queuing."""
        now = time.time() if timestamp_s is None else float(timestamp_s)
        h, w = portrait_bgr.shape[:2]
        report = {"saved_live": False, "saved_review": False, "saved_bg": False, "sprites": 0}

        # 1. Harvest clean arena background during empty match opening
        if (
            elapsed_s <= 3.0
            and not detections
            and not (unmatched_hp_bars or ())
            and self.stats["backgrounds_harvested"] < self.max_backgrounds
            and (now - self._last_bg_ts) >= 8.0
        ):
            bg_path = self.backgrounds_dir / f"{self._next_stem('bg')}.jpg"
            if cv2.imwrite(str(bg_path), portrait_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                self._last_bg_ts = now
                self.stats["backgrounds_harvested"] += 1
                report["saved_bg"] = True

        # 2. Match pending verified own plays against newly appeared ally boxes / HP bars
        verified_boxes: list[tuple[int, list[float]]] = []
        verified_dets: list[dict] = []
        active_pending: deque[_PendingOwnPlay] = deque(maxlen=16)
        for play in self._pending_plays:
            age = now - play.ts
            if age > 2.6 or play.matched:
                continue
            if age >= 0.25:
                # Look for nearest ally detection (YOLO or HP-bar) near played tile
                best_det = None
                best_dist = 78.0
                for d in detections:
                    if d.get("team") != "ally":
                        continue
                    cx, cy = d["center_px"]
                    dist = math.hypot(cx - play.px_x, cy - play.px_y)
                    if dist < best_dist:
                        best_dist = dist
                        best_det = d
                if best_det is not None:
                    box = best_det.get("bbox_px")
                    if box is None:
                        # Synthesize a reasonable troop bbox around an HP-bar anchor
                        cx, cy = best_det["center_px"]
                        box = [max(0.0, cx - 24.0), max(0.0, cy - 10.0),
                               min(float(w), cx + 24.0), min(float(h), cy + 44.0)]
                    cid = yolo_class_id("ally", play.card)
                    if cid is not None:
                        verified_boxes.append((cid, box))
                        verified_dets.append({
                            "class_name": play.card,
                            "team": "ally",
                            "bbox_px": box,
                            "conf": 1.0,
                            "verified_play": True,
                        })
                        play.matched = True
                        self.stats["verified_play_labels"] += 1
                        continue
            active_pending.append(play)
        self._pending_plays = active_pending

        # 3. Collect high-confidence YOLO detections on this frame
        yolo_dets = [d for d in detections if "bbox_px" in d and d.get("source") != "hp_bar_fallback"]
        high_conf_boxes: list[tuple[int, list[float]]] = []
        for d in yolo_dets:
            if float(d.get("conf", 0.0)) < self.pseudo_conf_threshold:
                continue
            cid = yolo_class_id(d["team"], d["class_name"])
            if cid is not None:
                high_conf_boxes.append((cid, d["bbox_px"]))

        # Save verified own-play frame immediately (merged with high-conf context boxes on same frame)
        if verified_boxes:
            merged = list(verified_boxes)
            for cid, box in high_conf_boxes:
                if not any(_box_iou(box, vbox) > 0.35 for _, vbox in verified_boxes):
                    merged.append((cid, box))
            if self._write_yolo_sample(portrait_bgr, merged, "verified"):
                self._last_saved_ts = now
                self.stats["verified_play_frames"] += 1
                self.stats["pseudo_labels"] += max(0, len(merged) - len(verified_boxes))
                report["saved_live"] = True
                extra_yolo = [
                    d for d in yolo_dets
                    if not any(_box_iou(d["bbox_px"], vd["bbox_px"]) > 0.35 for vd in verified_dets)
                ]
                report["sprites"] += self._harvest_sprites(portrait_bgr, verified_dets + extra_yolo)
        elif (
            high_conf_boxes
            and not (unmatched_hp_bars or ())
            and (now - self._last_saved_ts) >= self.min_frame_interval_s
        ):
            if self._write_yolo_sample(portrait_bgr, high_conf_boxes, "live"):
                self._last_saved_ts = now
                self.stats["pseudo_labels"] += len(high_conf_boxes)
                report["saved_live"] = True
                report["sprites"] += self._harvest_sprites(portrait_bgr, yolo_dets)

        # 4. Queue hard examples (missed HP-bar units or uncertain YOLO detections not already verified)
        uncertain = [
            d for d in yolo_dets
            if self.uncertain_conf_low <= float(d.get("conf", 0.0)) < self.pseudo_conf_threshold
            and not any(_box_iou(d["bbox_px"], vd["bbox_px"]) > 0.35 for vd in verified_dets)
        ]
        missed = list(unmatched_hp_bars or ())
        if (
            (missed or uncertain)
            and self.stats["hard_examples"] < self.max_review_frames
            and (now - self._last_review_ts) >= max(2.5, self.min_frame_interval_s * 1.8)
        ):
            stem = self._next_stem("review")
            img_path = self.review_img_dir / f"{stem}.jpg"
            meta_path = self.review_meta_dir / f"{stem}.json"
            if cv2.imwrite(str(img_path), portrait_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
                meta_path.write_text(json.dumps({
                    "timestamp_s": now,
                    "elapsed_s": round(elapsed_s, 2),
                    "missed_hp_bars": missed,
                    "uncertain_detections": uncertain,
                    "all_detections": detections,
                }, indent=2), encoding="utf-8")
                self._last_review_ts = now
                self.stats["hard_examples"] += 1
                report["saved_review"] = True

        if any(report.values()):
            self._save_stats()
        return report
