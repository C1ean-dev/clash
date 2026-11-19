import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import torch
from royalegym.protocol import (
    DeployStatus, EMPTY_CARD, EntityKind, MatchSetup, ShuffleMode,
)
from royaleimitate.replays import CARD_NAMES

from xbow_environment import PPO_GAMMA, battle_env
from xbow_capture_bridge import CaptureBridge
from xbow_dashboard import DashboardHandler
from xbow_play_session import PlaySession, PLAY_STEP_MS
from xbow_public_feed import PublicFeed
from xbow_replays import DECKS
from xbow_selfplay import Match, Rival, play_wave
from xbow_training import clone_loss, kata_weights


class DashboardResponseTests(unittest.TestCase):
    def test_client_abort_closes_response_during_headers_or_body(self):
        for previous_writes in (0, 1):
            with self.subTest(previous_writes=previous_writes):
                handler = DashboardHandler.__new__(DashboardHandler)
                handler.request_version = "HTTP/1.1"
                handler.requestline = "GET /api/videos HTTP/1.1"
                handler.close_connection = False
                error = ConnectionAbortedError(10053, "Cliente desconectado")
                handler.wfile = Mock()
                handler.wfile.write.side_effect = [None] * previous_writes + [error]

                handler.reply_json({"ok": True}, 200)

                self.assertTrue(handler.close_connection)
                self.assertEqual(handler.wfile.write.call_count, previous_writes + 1)


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


class JointTrainingTests(unittest.TestCase):
    def test_potential_is_zero_sum_and_leaves_the_discounted_outcome_objective(self):
        env = battle_env("xbow_30", "hog_26", reward_profile="potential")
        self.addCleanup(env.close)
        env.reset(seed=42, options={"setup": MatchSetup(
            decks=env.state_mutator.decks, elixir_milli=[10000, 6000],
        )})
        initial = env.battle_state
        initial_potential = sum(float(term.potential(initial, 0)) * weight
                                for term, weight in env.reward_fn.terms[1:])
        self.assertGreater(initial_potential, 0)
        discount = 1.0
        shaping_return = 0.0
        while env.agents:
            _, reward, *_ = env.step({"blue": 0, "red": 0})
            self.assertEqual(reward["blue"], -reward["red"])
            terms = env.reward_fn.terms_for(0)
            self.assertNotIn("XBowLockReward", terms)
            shaping_return += discount * sum(value for name, value in terms.items() if name != "terminal")
            discount *= PPO_GAMMA
        self.assertTrue(env.battle_state.game_over)
        self.assertAlmostEqual(shaping_return, -initial_potential)

    def test_same_match_records_actions_and_rewards_from_each_side(self):
        env = battle_env("hog_26", "xbow_30")
        self.addCleanup(env.close)
        reset, step = env.reset, env.step
        xbow, free = Mock(), Mock()
        xbow.decide.return_value = (0, -0.25, 0.0)
        free.decide.return_value = (0, -0.75, 0.0)
        match = Match(Rival("free", "free", free), seat=1, seed=42, number=1)

        def short_match(*, seed, options):
            return reset(seed=seed, options={**options, "max_ticks": 60})

        def reward_by_side(actions):
            obs, _, terminated, truncated, infos = step(actions)
            return obs, {"blue": -1.0, "red": 1.0}, terminated, truncated, infos

        with TemporaryDirectory() as folder, patch.object(env, "reset", side_effect=short_match), \
                patch.object(env, "step", side_effect=reward_by_side):
            [(xbow_run, free_run)] = play_wave(Path(folder), "ppo", xbow, [match], [env], "cpu",
                                                collect=True, collect_both=True)
        xbow_stats, xbow_rows = xbow_run
        free_stats, free_rows = free_run
        self.assertEqual(xbow_stats["steps"], free_stats["steps"])
        self.assertGreater(xbow_stats["reward"], 0)
        self.assertEqual(xbow_stats["reward"], -free_stats["reward"])
        np.testing.assert_array_equal(xbow_rows[4], -0.25)
        np.testing.assert_array_equal(free_rows[4], -0.75)
        self.assertTrue(np.all(xbow_rows[-2] > 0))
        self.assertTrue(np.all(free_rows[-2] < 0))


class PlaySessionTests(unittest.TestCase):
    def test_places_the_current_card_on_the_next_simulation_step(self):
        session = PlaySession()
        self.addCleanup(session.stop)
        opening = session.snapshot()["opening_ms"]
        self.assertGreater(opening, 0)
        while session.snapshot()["opening_ms"]:
            session._step()

        before = session.snapshot()
        card = next(card for card in before["human"]["hand"] if card["legal_tiles"])
        slot = card["slot"]
        x, y = card["legal_tiles"][0]
        self.assertFalse(session.queue_human_action(slot, x, y, card["id"] + 1)["ok"])
        queued = session.queue_human_action(slot, x, y, card["id"])
        self.assertTrue(queued["ok"])
        self.assertFalse(session.queue_human_action(slot, x, y, card["id"])["ok"])
        session._step()

        after = session.snapshot()
        self.assertEqual(after["ms"] - before["ms"], PLAY_STEP_MS)
        self.assertEqual(after["command"], {"id": queued["command_id"], "status": "played"})
        self.assertNotEqual(after["human"]["hand"][slot]["id"], card["id"])
        self.assertNotIn("hand", after["ai"])


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
