import unittest

import numpy as np
import torch
from royalegym.protocol import (
    DeployStatus, EMPTY_CARD, EntityKind, MatchSetup, ShuffleMode,
)
from royaleimitate.replays import CARD_NAMES

from xbow_environment import battle_env
from xbow_capture_bridge import CaptureBridge
from xbow_public_feed import PublicFeed
from xbow_replays import DECKS
from xbow_training import clone_loss, kata_weights


class PublicObservationTests(unittest.TestCase):
    def setUp(self):
        self.env = battle_env("xbow_30")
        self.addCleanup(self.env.close)
        ids = {card.name: card.card_id for card in self.env.engine.cards()}
        self.deck = [ids[CARD_NAMES[key]] for key in DECKS["xbow_30"]]

    def opening(self, red_deck, red_elixir):
        observations, _ = self.env.reset(seed=1, options={"setup": MatchSetup(
            decks=[self.deck, red_deck], shuffle=int(ShuffleMode.NONE),
            elixir_milli=[10000, red_elixir],
        )})
        for _ in range(12):
            observations, *_ = self.env.step({"blue": 0, "red": 0})
        return observations

    def test_hidden_enemy_state_has_no_effect_until_public_play(self):
        baseline = self.opening(self.deck, 10000)["blue"]
        changed = self.opening(list(reversed(self.deck)), 7000)
        for key in ("vector", "spatial", "action_mask"):
            np.testing.assert_array_equal(baseline[key], changed["blue"][key])
        self.assertEqual(self.env.obs_builder.memory[0].foe_recent, [])
        red_action = self.env.action_parser.encode(0, 3, 10)
        self.assertTrue(changed["red"]["action_mask"][red_action])
        self.env.step({"blue": 0, "red": red_action})
        self.assertEqual(self.env.obs_builder.memory[0].foe_recent, [self.deck[-1]])

    def test_xbow_lock_pays_once_for_active_target(self):
        observations = self.opening(self.deck, 10000)
        action = self.env.action_parser.encode(0, 3, 13)
        self.assertTrue(observations["blue"]["action_mask"][action])
        self.env.step({"blue": action, "red": 0})
        for _ in range(24):
            self.env.step({"blue": 0, "red": 0})
        self.assertAlmostEqual(self.env.reward_terms(0)["XBowLockReward"], 0.04)
        self.assertLess(min(self.env.battle_state.players[1].tower_hp), 3052)

    def test_cycle_tracks_the_last_four_public_plays(self):
        observations = self.opening(self.deck, 6000)
        played = []
        for _ in range(120):
            legal = np.flatnonzero(observations["red"]["action_mask"][1:1 + 4 * 32 * 18])
            action = int(legal[0] + 1) if legal.size else 0
            observations, *_ = self.env.step({"blue": 0, "red": action})
            played.extend(result.card_id for result in self.env.last_results
                          if result.team == 1 and result.status == DeployStatus.OK)
            if len(played) == 5:
                break
        self.assertEqual(len(played), 5)
        self.assertEqual(self.env.obs_builder.memory[0].enemy_queue(), played[-4:])


class ResamplingTests(unittest.TestCase):
    def test_kata_weights_decay_after_each_play(self):
        weights = kata_weights(np.array([0, 1, 0, 0, 0, 1, 0]), 0.25)
        np.testing.assert_allclose(weights, [4 / 3, 4, 2, 4 / 3, 4 / 3, 4, 2])

    def test_importance_correction_preserves_timing_and_resamples_placements(self):
        probabilities = torch.tensor([[0.8, 0.15, 0.05]]).expand(4, -1)
        distribution = torch.distributions.Categorical(probs=probabilities)
        expected_timing = -(torch.log(torch.tensor(0.8)) + torch.log(torch.tensor(0.2))) / 2
        weighted_choice = -0.75 * torch.log(torch.tensor(0.75))
        loss = clone_loss(distribution, torch.tensor([0, 1, 1, 1]),
                          torch.tensor([2, 2 / 3, 2 / 3, 2 / 3]))
        torch.testing.assert_close(loss, expected_timing + weighted_choice)


class PublicFeedTests(unittest.TestCase):
    def test_feed_reproduces_policy_inputs_with_public_fields_only(self):
        for deck in DECKS:
            with self.subTest(deck=deck):
                self.compare_feed(deck)

    def compare_feed(self, deck):
        env = battle_env(deck)
        feed = PublicFeed(deck)
        try:
            obs, _ = env.reset(seed=43)
            cards = {card_id: key for key, card_id in feed.keys.items()}
            cards[EMPTY_CARD] = None
            events = []
            for step in range(36):
                state = env.battle_state
                own = state.players[0]
                towers = [
                    {"hp": p.tower_hp, "max_hp": p.tower_max_hp,
                     "crowns": p.crowns, "king_active": p.king_active}
                    for p in state.players
                ]
                packet = {
                    "tick": state.tick, "hand": [cards[c] for c in own.hand],
                    "next_card": cards[own.next_card], "own_elixir_milli": own.elixir_milli,
                    "own_towers": towers[0], "opponent_towers": towers[1],
                    "events": events,
                    "units": [{
                        "side": ("own", "opponent")[unit.team], "card": cards[unit.card_id],
                        "position": [unit.x / feed.scale, unit.y / feed.scale],
                        "hp": unit.hp, "max_hp": unit.max_hp, "radius": unit.radius / feed.scale,
                        "kind": "building" if unit.kind == EntityKind.BUILDING
                        else "air" if unit.flying else "ground",
                        "deploying": unit.deploy_ticks > 0, "stunned": unit.stun_ticks > 0,
                        "footprint": ([
                            [unit.footprint[0] / feed.scale, unit.footprint[1] / feed.scale],
                            [unit.footprint[2] / feed.scale, unit.footprint[3] / feed.scale],
                        ] if unit.footprint is not None else None),
                    } for unit in state.entities if unit.tower_slot < 0 and unit.hp > 0],
                    "spells": [{
                        "side": ("own", "opponent")[spell.team], "card": cards.get(spell.card_id),
                        "position": [spell.x / feed.scale, spell.y / feed.scale],
                        "own_aim": ([spell.aim_x / feed.scale, spell.aim_y / feed.scale]
                                    if spell.team == 0 else None),
                    } for spell in state.spells],
                }
                projected = feed.update(packet)
                for key in ("spatial", "vector"):
                    np.testing.assert_array_equal(projected[key], obs["blue"][key],
                                                  err_msg=f"{deck}, tick {state.tick}, {key}")
                self.assertTrue(np.all(projected["action_mask"] <= obs["blue"]["action_mask"]))
                self.assertGreaterEqual(projected["action_mask"].sum(),
                                        0.6 * obs["blue"]["action_mask"].sum())
                actions = dict.fromkeys(("blue", "red"), 0)
                if step in (12, 19, 27):
                    for agent in actions:
                        legal = np.flatnonzero(obs[agent]["action_mask"][1:])
                        actions[agent] = int(legal[0] + 1) if legal.size else 0
                obs, *_ = env.step(actions)
                events = [
                    {"tick": r.tick, "side": ("own", "opponent")[r.team], "card": cards[r.card_id]}
                    for r in env.last_results if r.status == DeployStatus.OK
                ]
            with self.assertRaises(ValueError):
                feed.update({**packet, "opponent_elixir_milli": 9000})
            with self.assertRaisesRegex(ValueError, "deve avançar"):
                feed.update(packet)
        finally:
            feed.close()
            env.close()

    def test_capture_bridge_consumes_confirmed_plays_and_ignores_enemy_estimates(self):
        bridge = CaptureBridge("xbow_30")
        feed = PublicFeed("xbow_30")
        self.addCleanup(feed.close)
        frame = {
            "in_game": True, "timestamp_s": 10.0,
            "record": {"visual_state": {
                "time_left_s": 180, "overtime": False,
                "hand": ["xbow", "tesla", "archers", "knight"], "next_card": "skeletons",
                "elixir": 6, "enemy_elixir_est": 0.1, "seen_enemy_cards": ["rocket"],
                "tower_hp_self": [3012, 4824, 3052], "tower_hp_enemy": [3000, 4824, 3030],
                "arena_px": [0, 0, 568, 896],
            }},
        }
        start = bridge.push(frame)
        self.assertEqual(start["opponent_towers"]["hp"], (4824, 3030, 3000))
        feed.update(start)
        self.assertEqual(feed.env.obs_builder.memory[0].enemy_queue(), [EMPTY_CARD] * 4)
        early = {**frame, "timestamp_s": 10.25, "enemy_plays": [
            {"card": "rocket", "clock_confirmed": False},
            {"card": "hog-rider", "frame_confirmed": True},
        ]}
        self.assertIsNone(bridge.push(early))
        second = bridge.push({**frame, "timestamp_s": 15.0, "detections": [{
            "class_name": "hog-rider", "team": "enemy", "center_px": [284, 450],
            "estimated_hp": 0.8,
        }]})
        self.assertEqual(second["tick"], 100)
        self.assertEqual(second["events"], [{"card": "hog-rider", "side": "opponent", "tick": 5}])
        self.assertEqual(second["units"][0]["kind"], "ground")
        current = feed.update(second)
        self.assertEqual(feed.env.obs_builder.memory[0].foe_recent, [feed.card("hog-rider")])
        self.assertGreater(np.count_nonzero(current["action_mask"]), 1)


if __name__ == "__main__":
    unittest.main()
