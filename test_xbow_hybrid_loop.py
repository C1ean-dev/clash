"""Unit & end-to-end tests for the Hybrid Live <-> Synthetic Policy Loop."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import torch

from xbow_environment import battle_env
from xbow_hybrid_loop import (
    ARCHETYPE_RIVALS,
    analyze_live_meta,
    classify_enemy_archetype,
    hybrid_status,
    run_hybrid_cycle,
)
from xbow_selfplay import PFSPPool, TACTICAL_RIVALS
from xbow_training import XBowPolicy, save_checkpoint


def _write_dummy_live_match(path: Path, deck: str, result: str, enemy_cards: list[str]) -> None:
    rows = [
        {"kind": "header", "deck": deck, "checkpoint": "dummy.pt"},
        {
            "kind": "step",
            "frame": {
                "tick": 0, "time_left_s": 180.0, "overtime": False,
                "hand": ["x-bow", "tesla", "knight", "archers"], "next_card": "skeletons",
                "own_elixir_milli": 7000,
                "own_towers": {"hp": [4824, 3052, 3052], "max_hp": [4824, 3052, 3052], "crowns": 0},
                "opponent_towers": {"hp": [4824, 3052, 3052], "max_hp": [4824, 3052, 3052], "crowns": 0},
                "units": [
                    {"side": "opponent", "card": c, "position": [4.0, 22.0], "hp": 1000, "max_hp": 1000,
                     "kind": "ground", "radius": 0.5, "form": "basic"}
                    for c in enemy_cards
                ],
                "spells": [],
                "events": [],
            },
            "suggestion": {"kind": "wait", "action": 0, "played_action": 0},
            "played_action": 0,
        },
        {
            "kind": "step",
            "frame": {
                "tick": 20, "time_left_s": 179.0, "overtime": False,
                "hand": ["x-bow", "tesla", "knight", "archers"], "next_card": "skeletons",
                "own_elixir_milli": 7500,
                "own_towers": {"hp": [4824, 3052, 3052], "max_hp": [4824, 3052, 3052], "crowns": 1 if result == "win" else 0},
                "opponent_towers": {"hp": [4824, 3052, 3052], "max_hp": [4824, 3052, 3052], "crowns": 1 if result == "loss" else 0},
                "units": [],
                "spells": [],
                "events": [],
            },
            "suggestion": {"kind": "wait", "action": 0, "played_action": 0},
            "played_action": 0,
        },
        {"kind": "footer", "result": result, "steps": 2},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


class HybridLoopTests(unittest.TestCase):
    def test_all_six_archetype_rivals_registered_in_tactical_rivals(self):
        for rival_name, _deck_key in ARCHETYPE_RIVALS:
            self.assertIn(rival_name, TACTICAL_RIVALS)

    def test_archetype_classification_ignores_pre_yolo_knight_only_matches(self):
        self.assertIsNone(classify_enemy_archetype(Counter({"knight": 500})))
        self.assertEqual(
            classify_enemy_archetype(Counter({"knight": 200, "hog-rider": 12, "musketeer": 8})),
            "hog_push",
        )
        self.assertEqual(
            classify_enemy_archetype(Counter({"golem": 5, "night-witch": 7, "lumberjack": 4})),
            "golem_patient",
        )
        self.assertEqual(
            classify_enemy_archetype(Counter({"pekka": 6, "bandit": 10, "battle-ram": 9})),
            "pekka_push",
        )
        self.assertEqual(
            classify_enemy_archetype(Counter({"princess": 15, "goblin-barrel": 8, "inferno-tower": 6})),
            "log_bait_push",
        )

    def test_analyze_live_meta_seeds_pfsp_toward_losing_archetypes(self):
        with tempfile.TemporaryDirectory() as tmp:
            matches_dir = Path(tmp) / "live_matches"
            matches_dir.mkdir(parents=True)
            # Pre-YOLO knight-only match (should be scanned overall, not classified into an archetype)
            _write_dummy_live_match(matches_dir / "m01.jsonl", "xbow_30", "loss", ["knight"])
            # 3 losses vs Golem beatdown -> lowest PFSP seed -> highest sampling weight
            for i in range(3):
                _write_dummy_live_match(matches_dir / f"m_golem_{i}.jsonl", "xbow_30", "loss", ["golem", "night-witch"])
            # 3 wins vs Hog 2.6 -> higher PFSP seed -> lower sampling weight
            for i in range(3):
                _write_dummy_live_match(matches_dir / f"m_hog_{i}.jsonl", "xbow_30", "win", ["hog-rider", "musketeer"])

            meta = analyze_live_meta(matches_dir, deck="xbow_30")
            self.assertEqual(meta["matches_scanned"], 7)
            self.assertEqual(meta["classified_matches"], 6)
            self.assertLess(meta["pfsp_seed"]["golem_patient"], meta["pfsp_seed"]["hog_push"])

            pool = PFSPPool(initial_wr=meta["pfsp_seed"])
            names = [r for r, _ in ARCHETYPE_RIVALS]
            weights = dict(zip(names, pool.weights(names)))
            self.assertGreater(weights["golem_patient"], weights["hog_push"])

    def test_run_hybrid_cycle_end_to_end_and_hot_reload_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matches_dir = root / "live_matches"
            matches_dir.mkdir(parents=True)
            _write_dummy_live_match(matches_dir / "m1.jsonl", "xbow_30", "loss", ["pekka", "bandit"])
            _write_dummy_live_match(matches_dir / "m2.jsonl", "xbow_30", "win", ["hog-rider", "cannon"])

            env = battle_env("xbow_30", "hog_26", obs_profile="v3")
            try:
                obs, _ = env.reset(seed=1)
                sample_obs = obs["blue"]
                policy = XBowPolicy(
                    sample_obs["spatial"].shape[0],
                    sample_obs["vector"].shape[0],
                    len(sample_obs["action_mask"]),
                    architecture=3,
                    obs_profile="v3",
                )
            finally:
                env.close()

            bc_path = root / "bc.pt"
            live_path = root / "ppo_live.pt"
            save_checkpoint(bc_path, policy, "xbow_30", "bc", {"epoch": 1})
            save_checkpoint(live_path, policy, "xbow_30", "ppo", {"update": 1})
            # Ensure initial mtime is slightly in the past so promotion mtime is strictly newer
            old_t = time.time() - 5.0
            os.utime(live_path, (old_t, old_t))

            report = run_hybrid_cycle(
                deck="xbow_30",
                checkpoint=live_path,
                bc_checkpoint=bc_path,
                output=live_path,
                matches_dir=matches_dir,
                skip_awr=True,
                ppo_updates=1,
                rollout_games=2,
                batch_size=32,
                eval_games_per_rival=1,
                eval_rival_limit=2,
                parallel_games=2,
                threads=4,
                seed=123,
                force_promote=True,
            )

            self.assertTrue(report["promoted"])
            self.assertEqual(len(report["synthetic_updates"]), 1)
            self.assertIn("ab_eval", report)
            self.assertEqual(report["ab_eval"]["games_evaluated"], 3)  # incumbent + top 2 hardest meta archetypes
            self.assertTrue(live_path.is_file())
            self.assertTrue(live_path.with_suffix(".bak.pt").is_file())
            self.assertTrue((root / "hybrid_loop.json").is_file())
            self.assertGreater(live_path.stat().st_mtime, old_t)
            self.assertGreaterEqual(len(list((root / "league").glob("*.pt"))), 1)

            st = hybrid_status(live_path, matches_dir, "xbow_30")
            self.assertEqual(st["cycles_completed"], 1)
            self.assertEqual(st["promotions"], 1)


class DeckDiscoveryAndAllCardsTests(unittest.TestCase):
    def test_all_119_cards_can_be_simulated_in_battle_env(self):
        from xbow_deck_discovery import ALL_SIMULATOR_CARDS
        from xbow_environment import battle_env

        self.assertEqual(len(ALL_SIMULATOR_CARDS), 119)
        # Partition all 119 cards into valid 8-card rival decks and verify every single card resets cleanly!
        cards = list(ALL_SIMULATOR_CARDS)
        for i in range(0, len(cards), 8):
            chunk = cards[i:i + 8]
            if len(chunk) < 8:
                pad = [c for c in cards if c not in chunk][:8 - len(chunk)]
                chunk = chunk + pad
            env = battle_env("xbow_30", tuple(chunk), obs_profile="v3")
            try:
                obs, _ = env.reset(seed=i + 1)
                self.assertIn("blue", obs)
                self.assertIn("red", obs)
            finally:
                env.close()

    def test_discovered_deck_store_completion_dedup_pfsp_and_scripted_move(self):
        import numpy as np
        from xbow_deck_discovery import DiscoveredDeckStore, complete_partial_deck
        from xbow_environment import battle_env
        from xbow_selfplay import scripted_move

        # Pre-YOLO knight-only match is rejected
        self.assertIsNone(complete_partial_deck(Counter({"knight": 300})))
        # Partial 5-card match is completed cleanly to 8 distinct simulator cards
        built = complete_partial_deck(Counter({
            "royal-recruits": 40, "royal-hogs": 35, "zappies": 25, "flying-machine": 20, "barbarian-barrel": 15,
        }))
        self.assertIsNotNone(built)
        cards, status = built
        self.assertEqual(len(cards), 8)
        self.assertEqual(len(set(cards)), 8)
        self.assertEqual(status, "completed")

        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "discovered_decks.json"
            store = DiscoveredDeckStore(store_path)
            e1 = store.record_live_match(Counter({
                "royal-recruits": 30, "royal-hogs": 25, "valkyrie": 20, "bats": 15,
                "fireball": 10, "arrows": 10, "goblin-cage": 10, "flying-machine": 10,
            }), "loss")
            self.assertIsNotNone(e1)
            self.assertEqual(e1["status"], "verified")
            self.assertEqual(e1["losses"], 1)

            # Duplicate match increments counts without duplicating deck entry
            e2 = store.record_live_match(Counter({
                "royal-recruits": 20, "royal-hogs": 15, "valkyrie": 10, "bats": 10,
                "fireball": 5, "arrows": 5, "goblin-cage": 5, "flying-machine": 5,
            }), "loss")
            self.assertEqual(e1["id"], e2["id"])
            self.assertEqual(len(store.data["decks"]), 1)
            self.assertEqual(store.data["decks"][e1["id"]]["losses"], 2)

            # Verify dynamic scripted_move works with dyn_* rival style on that exact discovered deck
            env = battle_env("xbow_30", tuple(e1["cards"]), obs_profile="v3")
            try:
                obs, _ = env.reset(seed=9)
                act = scripted_move(env, 1, obs["red"], np.random.default_rng(9), f"dyn_{e1['play_style']}_{e1['id']}")
                self.assertTrue(obs["red"]["action_mask"][act])
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
