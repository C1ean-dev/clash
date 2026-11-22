"""Unit & end-to-end tests for public YOLO detection, live auto-collection, and synthetic YOLO training."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from royaleimitate.replays import CARD_NAMES
from xbow_capture_bridge import CaptureBridge, card_key
from xbow_screen_vision import ScreenVisionAnalyzer
from xbow_vision_train import (
    build_merged_dataset,
    generate_synthetic_dataset,
    train_vision_yolo,
    vision_status,
)
from xbow_vision_yolo import (
    CRBAB_DETECTOR_UNITS,
    DEFAULT_VISION_MODELS_DIR,
    PUBLIC_ONNX_UNITS_NAME,
    YOLO_CLASS_NAMES,
    VisionAutoCollector,
    build_vision_detector,
    normalize_unit_slug,
    tile_to_portrait_px,
    yolo_class_id,
)


class VisionYoloPipelineTests(unittest.TestCase):
    def test_all_public_onnx_units_map_to_simulator_cards(self):
        self.assertEqual(len(CRBAB_DETECTOR_UNITS), 97)
        missing = []
        for raw in CRBAB_DETECTOR_UNITS:
            slug = normalize_unit_slug(raw)
            if card_key(slug) not in CARD_NAMES:
                missing.append((raw, slug))
        self.assertEqual(missing, [], f"Unmapped public ONNX units: {missing}")

    def test_public_onnx_detector_and_bridge_on_preview(self):
        weights = DEFAULT_VISION_MODELS_DIR / PUBLIC_ONNX_UNITS_NAME
        if not weights.is_file():
            self.skipTest("Public ONNX weights not present.")
        det = build_vision_detector("public", deck="xbow_30", models_dir=DEFAULT_VISION_MODELS_DIR)
        self.assertIsNotNone(det)
        self.assertIn("public_onnx", det.backend_name)

        # Verify detector runs cleanly on a portrait arena image and maps outputs into CaptureBridge
        preview = Path("calibration/preview.png")
        img = cv2.imread(str(preview)) if preview.is_file() else None
        if img is None:
            img = np.full((896, 568, 3), 80, dtype=np.uint8)
        portrait = cv2.resize(img, (568, 896), interpolation=cv2.INTER_AREA)
        preds = det.detect(portrait)
        self.assertIsInstance(preds, list)
        for d in preds:
            self.assertIn(d["team"], ("ally", "enemy"))
            self.assertIn(d["class_name"], CARD_NAMES)
            if d.get("source") == "public_onnx_yolo":
                self.assertIn("bbox_px", d)
                self.assertEqual(len(d["bbox_px"]), 4)

        # Verify CaptureBridge accepts synthetic YOLO detections for all unit types
        bridge = CaptureBridge("xbow_30", allow_mid_match=True)
        captured = {
            "in_game": True,
            "timestamp_s": 10.0,
            "detections": [
                {"class_name": "electro-spirit", "team": "ally", "center_px": [300.0, 620.0],
                 "bbox_px": [280.0, 600.0, 320.0, 640.0], "conf": 0.87, "estimated_hp": 1.0,
                 "source": "public_onnx_yolo"},
                {"class_name": "royal-recruits", "team": "enemy", "center_px": [420.0, 520.0],
                 "bbox_px": [390.0, 490.0, 450.0, 550.0], "conf": 0.72, "estimated_hp": 0.8,
                 "source": "public_onnx_yolo"},
            ],
            "own_actions": [],
            "enemy_plays": [{"card": "royal-recruits", "frame_confirmed": True}],
            "record": {"visual_state": {
                "time_left_s": 170.0, "overtime": False,
                "hand": ["x-bow", "tesla", "knight", "archers"], "next_card": "skeletons",
                "hand_confidence": 0.9, "elixir": 6.5,
                "tower_hp_self": [3052, 4824, 3052], "tower_hp_enemy": [3052, 4824, 3052],
                "arena_px": [0.0, 0.0, 568.0, 896.0],
            }},
        }
        pub = bridge.push(captured)
        self.assertIsNotNone(pub)
        cards = {(u["side"], u["card"]) for u in pub["units"]}
        self.assertIn(("own", "electro-spirit"), cards)
        self.assertIn(("opponent", "royal-recruits"), cards)

    def test_auto_collector_verified_play_pseudo_bg_and_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "vision_ds"
            collector = VisionAutoCollector(root, min_frame_interval_s=0.1, pseudo_conf_threshold=0.50)
            portrait = np.full((896, 568, 3), 90, dtype=np.uint8)

            # 1. Empty opening frame -> background harvested
            r_bg = collector.observe_frame(portrait, [], unmatched_hp_bars=[], elapsed_s=0.5, timestamp_s=100.0)
            self.assertTrue(r_bg["saved_bg"])
            self.assertEqual(collector.stats["backgrounds_harvested"], 1)

            # 2. Verified own play -> ground-truth label + sprite crop
            collector.record_own_play("x-bow", 12.0, 12.0, timestamp_s=101.0)
            px, py = tile_to_portrait_px(12.0, 12.0)
            dets = [{
                "class_name": "tesla",  # Intentionally wrong detector guess; verified play overrides to x-bow!
                "team": "ally",
                "center_px": [px + 4.0, py - 3.0],
                "bbox_px": [px - 22.0, py - 24.0, px + 26.0, py + 24.0],
                "conf": 0.40,
                "source": "public_onnx_yolo",
            }]
            r_ver = collector.observe_frame(portrait, dets, unmatched_hp_bars=[], elapsed_s=5.0, timestamp_s=101.6)
            self.assertTrue(r_ver["saved_live"])
            self.assertEqual(collector.stats["verified_play_frames"], 1)
            self.assertEqual(collector.stats["verified_play_labels"], 1)
            lbl_files = list((root / "live" / "labels").glob("verified_*.txt"))
            self.assertEqual(len(lbl_files), 1)
            first_cls = int(lbl_files[0].read_text(encoding="utf-8").split()[0])
            self.assertEqual(first_cls, yolo_class_id("ally", "x-bow"))
            self.assertGreaterEqual(len(list((root / "sprites" / "ally_x-bow").glob("*.png"))), 1)

            # 3. Missed HP bar -> queued into review/ (NOT pseudo-labeled)
            unmatched = [{"class_name": "knight", "team": "enemy", "center_px": [200.0, 300.0], "estimated_hp": 0.8}]
            r_rev = collector.observe_frame(portrait, [], unmatched_hp_bars=unmatched, elapsed_s=12.0, timestamp_s=110.0)
            self.assertFalse(r_rev["saved_live"])
            self.assertTrue(r_rev["saved_review"])
            self.assertEqual(collector.stats["hard_examples"], 1)

    def test_synthetic_generation_merge_and_end_to_end_yolo_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = Path(tmp) / "vision_ds"
            models_dir = Path(tmp) / "models"

            # Seed one synthetic live sample to verify merge + oversampling
            collector = VisionAutoCollector(ds_dir, min_frame_interval_s=0.0)
            portrait = np.full((896, 568, 3), 110, dtype=np.uint8)
            collector.observe_frame(portrait, [], elapsed_s=0.2, timestamp_s=1.0)
            collector.record_own_play("knight", 9.0, 10.0, timestamp_s=2.0)
            px, py = tile_to_portrait_px(9.0, 10.0)
            collector.observe_frame(
                portrait,
                [{
                    "class_name": "knight",
                    "team": "ally",
                    "center_px": [px, py],
                    "bbox_px": [px - 18.0, py - 22.0, px + 18.0, py + 22.0],
                    "conf": 0.85,
                    "source": "public_onnx_yolo",
                }],
                elapsed_s=4.0,
                timestamp_s=2.6,
            )

            gen = generate_synthetic_dataset(ds_dir, count=12, val_ratio=0.25, deck="xbow_30", seed=7)
            self.assertEqual(gen["train_images"], 9)
            self.assertEqual(gen["val_images"], 3)
            self.assertGreater(gen["total_boxes"], 12)
            self.assertGreater(gen["backgrounds_indexed"], 0)
            self.assertGreater(gen["sprite_classes_indexed"], 0)

            merged = build_merged_dataset(ds_dir, live_oversample=2)
            self.assertEqual(merged["counts"]["syn_train"], 9)
            self.assertEqual(merged["counts"]["syn_val"], 3)
            self.assertGreaterEqual(merged["counts"]["live_train"], 3)  # verified gets +1 boost

            # Run a real 1-epoch Ultralytics fine-tune smoke test (CPU, small imgsz for speed)
            report = train_vision_yolo(
                ds_dir,
                models_dir,
                base_model="yolov8n.pt",
                epochs=1,
                imgsz=160,
                batch=4,
                device="cpu",
                force_promote=True,
            )
            self.assertTrue(report["promoted"])
            custom_pt = models_dir / "custom_yolo.pt"
            self.assertTrue(custom_pt.is_file())

            # Load the newly promoted custom_yolo.pt through our unified detector and run inference
            det = build_vision_detector("custom", deck="xbow_30", models_dir=models_dir, conf=0.05)
            self.assertIsNotNone(det)
            self.assertIn("ultralytics:custom_yolo.pt", det.backend_name)
            preds = det.detect(portrait)
            self.assertIsInstance(preds, list)

            st = vision_status(ds_dir, models_dir)
            self.assertIsNotNone(st["models"]["custom_yolo_pt"])
            self.assertEqual(st["dataset"]["synthetic_train"], 9)


if __name__ == "__main__":
    unittest.main()
