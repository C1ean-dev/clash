from contextlib import ExitStack
import os
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

from xbow_arena import LIVE_FRAMES, live_data
from xbow_environment import PPO_GAMMA, RIVAL_DECKS, battle_env, pack_id_planes
from xbow_capture_bridge import CaptureBridge
from xbow_dashboard import DashboardHandler
from xbow_metrics import (
    TrainingStopped, check_training_stop, defer_training_stop, read_run_journal, training_journal,
)
from xbow_live_learn import finetune_from_live, load_live_matches
from xbow_live_overlay import LiveMatchRecorder, infer_match_result, tile_to_window_px
from xbow_play_session import PlaySession, PLAY_STEP_MS
from xbow_public_feed import PublicFeed
from xbow_replays import DECKS, MatchSamples, board_value, eligible_replay
from xbow_rollout_pool import RolloutWorkerPool
from xbow_selfplay import Match, PFSPPool, Rival, decide_batch, play_wave, rollout_tensors
from xbow_training import (
    DiskMatchSamples, MirrorSpec, RawDiskMatchSamples, ReplayFrames, XBowPolicy, bc_lr_scheduler,
    bc_optimizer, clone_loss, evaluate_clone, frozen_policy, kata_weights, load_checkpoint,
    masked_kl, mirror_sample, optimize_ppo, save_checkpoint, scheduled_kl_coef, warmup_value_head,
)


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
        for key in ("vector", "spatial", "action_mask", "card_ids", "spell_ids", "unit_ids"):
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
    def test_continuous_stop_waits_for_checkpoint_boundary(self):
        with TemporaryDirectory() as directory:
            signal = Path(directory) / "stop"
            with patch.dict(os.environ, {"XBOW_STOP_FILE": str(signal)}):
                with defer_training_stop(True):
                    signal.touch()
                    check_training_stop()
                    with self.assertRaises(TrainingStopped):
                        check_training_stop(at_boundary=True)
                with self.assertRaises(TrainingStopped):
                    check_training_stop()

    def test_continuous_journal_bounds_its_events_and_keeps_regular_history(self):
        with TemporaryDirectory() as directory, patch("xbow_metrics.CONTINUOUS_EVENTS", 12):
            output = Path(directory)
            with training_journal(output, "xbow_30") as report:
                report("ppo_update", done=1)
            regular = (output / "progress.jsonl").read_bytes()
            with training_journal(output, "xbow_30", continuous=True) as report:
                for update in range(2, 270):
                    report("ppo_update", done=update)
            self.assertEqual((output / "progress.jsonl").read_bytes(), regular)
            updates = [event["done"] for event in read_run_journal(output)
                       if event["event"] == "ppo_update"]
            self.assertEqual(updates[0], 1)
            self.assertEqual(updates[-1], 269)
            self.assertLess(len(updates), 12 + 128)

    def test_live_arena_limits_frames_and_discards_finished_boards(self):
        with ExitStack() as stack:
            output = Path(stack.enter_context(TemporaryDirectory()))
            folder = output / "arena-live"
            stack.enter_context(patch.dict(os.environ, {"XBOW_ARENA": "1", "XBOW_ARENA_DIR": str(folder)}))
            stack.enter_context(patch("xbow_arena.PUBLISH_INTERVAL", 0))
            environments = [battle_env("xbow_30") for _ in range(4)]
            for env in environments:
                stack.callback(env.close)
                reset = env.reset

                def short_match(*, seed, options, reset=reset):
                    return reset(seed=seed, options={**options, "max_ticks": 120})

                stack.enter_context(patch.object(env, "reset", side_effect=short_match))
            policy = Mock()
            matches = [Match(Rival("free", "xbow_30", policy), n % 2, n, n) for n in range(1, 5)]
            frame_counts = []

            def decide(_, inputs, device):
                boards = live_data(folder, "ppo")
                self.assertEqual([board["header"]["number"] for board in boards], [1, 2, 3, 4])
                frame_counts.extend(len(board["frames"]) for board in boards)
                self.assertEqual(len(list(folder.iterdir())), 4)
                return [(0, 0.0, 0.0)] * len(inputs)

            stack.enter_context(patch("xbow_selfplay.decide_batch", side_effect=decide))
            play_wave(output, "ppo", policy, matches, environments, "cpu")
            self.assertEqual(min(frame_counts), 1)
            self.assertEqual(max(frame_counts), LIVE_FRAMES)
            self.assertEqual(live_data(folder, "ppo"), [])
            self.assertFalse(list(folder.iterdir()))

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
        env = battle_env(deck, obs_profile="v2")
        feed = PublicFeed(deck, obs_profile="v2")
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

    def test_feed_v3_builds_rich_observations_for_v3_policy(self):
        bridge = CaptureBridge("xbow_30")
        feed = PublicFeed("xbow_30", obs_profile="v3")
        self.addCleanup(feed.close)
        frame = {
            "in_game": True, "timestamp_s": 10.0,
            "record": {"visual_state": {
                "time_left_s": 180, "overtime": False,
                "hand": ["xbow", "tesla", "archers", "knight"], "next_card": "skeletons",
                "elixir": 6, "tower_hp_self": [3012, 4824, 3052], "tower_hp_enemy": [3000, 4824, 3030],
                "arena_px": [0, 0, 568, 896],
            }},
        }
        first = feed.update(bridge.push(frame))
        policy = XBowPolicy.from_observation(first, env=feed.env).eval()
        self.assertEqual(policy.architecture, 3)
        self.assertEqual(policy.obs_profile, "v3")
        second = feed.update(bridge.push({**frame, "timestamp_s": 15.0, "detections": [{
            "class_name": "hog-rider", "team": "enemy", "center_px": [284, 450], "estimated_hp": 0.8,
        }]}))
        self.assertEqual(pack_id_planes(second).shape, (8, 32, 18))
        action, logp, value = policy.decide(second, "cpu", deterministic=True)
        self.assertTrue(second["action_mask"][action])
        self.assertTrue(np.isfinite(logp) and np.isfinite(value))


class ArchitectureV3Tests(unittest.TestCase):
    def test_v3_policy_forward_gradients_masking_and_checkpoint_roundtrip(self):
        env = battle_env("xbow_30", "hog_26", obs_profile="v3")
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=7)
        blue_lockout = obs["blue"]
        # Advance past deploy lockout so cards become legal.
        for _ in range(12):
            obs, *_ = env.step({"blue": 0, "red": 0})
        blue_ready = obs["blue"]
        self.assertEqual(blue_ready["spatial"].shape, (66, 32, 18))
        self.assertEqual(pack_id_planes(blue_ready).shape, (8, 32, 18))

        policy = XBowPolicy.from_observation(blue_ready, env=env).to(memory_format=torch.channels_last)
        self.assertEqual(policy.architecture, 3)

        # Verify both all-wait lockout frame and ready frame produce exact normalized log-probs.
        spatial = torch.as_tensor(np.stack([blue_lockout["spatial"], blue_ready["spatial"]]))
        vector = torch.as_tensor(np.stack([blue_lockout["vector"], blue_ready["vector"]]))
        mask = torch.as_tensor(np.stack([blue_lockout["action_mask"], blue_ready["action_mask"]]), dtype=torch.bool)
        id_planes = torch.as_tensor(np.stack([pack_id_planes(blue_lockout), pack_id_planes(blue_ready)]))

        logits, values = policy.logits_and_value(spatial, vector, mask, id_planes=id_planes)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(values).all())
        torch.testing.assert_close(logits.logsumexp(dim=-1), torch.zeros(2), atol=1e-4, rtol=0)
        # Lockout row must put all probability mass on wait (action 0).
        self.assertAlmostEqual(float(torch.softmax(logits[0], dim=-1)[0].item()), 1.0, places=5)

        # Verify gradients flow through embeddings, ResNet, factored head, and value head.
        dist, vals = policy(spatial, vector, mask, id_planes=id_planes)
        loss = -dist.log_prob(torch.tensor([0, 0])).mean() + vals.square().mean()
        loss.backward()
        self.assertIsNotNone(policy.card_embed.weight.grad)
        self.assertIsNotNone(policy.unit_embed.weight.grad)
        self.assertIsNotNone(policy.coarse_blocks[0].conv1.weight.grad)
        self.assertIsNotNone(policy.value[-1].weight.grad)

        # Save & reload checkpoint roundtrip.
        policy.eval()
        with TemporaryDirectory() as directory:
            ckpt = Path(directory) / "v3.pt"
            save_checkpoint(ckpt, policy, "xbow_30", "bc", {"nll": 0.5})
            reloaded, saved = load_checkpoint(ckpt, "xbow_30", "cpu", memory_format=torch.channels_last)
            self.assertEqual(saved["architecture"], 3)
            self.assertEqual(reloaded.obs_profile, "v3")
            with torch.no_grad():
                l1, v1 = policy.logits_and_value(spatial, vector, mask, id_planes=id_planes)
                l2, v2 = reloaded.logits_and_value(spatial, vector, mask, id_planes=id_planes)
            torch.testing.assert_close(l1, l2, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(v1, v2, atol=1e-5, rtol=1e-5)

    def test_v3_disk_frames_rollout_and_ppo_end_to_end(self):
        env = battle_env("xbow_30", "hog_26", reward_profile="potential", obs_profile="v3")
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=9)
        policy = XBowPolicy.from_observation(obs["blue"], env=env).to(memory_format=torch.channels_last)
        reference = frozen_policy(policy)

        # 1. DiskMatchSamples + ReplayFrames + evaluate_clone
        frames_sp, frames_vec, frames_mask, frames_ids, frames_act = [], [], [], [], []
        for _ in range(4):
            b = obs["blue"]
            frames_sp.append(b["spatial"].astype(np.float16))
            frames_vec.append(b["vector"].astype(np.float16))
            frames_mask.append(np.packbits(b["action_mask"]))
            frames_ids.append(pack_id_planes(b))
            frames_act.append(0)
            obs, *_ = env.step({"blue": 0, "red": 0})
        sample = MatchSamples(
            "smoke", np.stack(frames_sp), np.stack(frames_vec), np.stack(frames_mask),
            np.array(frames_act), np.stack(frames_ids),
        )
        with TemporaryDirectory() as directory:
            with (Path(directory) / "frames.bin").open("w+b") as store:
                disk_sample = DiskMatchSamples(sample, store)
                loader = torch.utils.data.DataLoader(ReplayFrames([disk_sample], policy.spec[-1]), batch_size=2)
                metrics = evaluate_clone(policy.eval(), loader, "cpu")
                self.assertTrue(np.isfinite(metrics["nll"]))

        # 2. Batched rollout + PPO update with v3 id_planes
        reset = env.reset
        with TemporaryDirectory() as folder, patch.object(
            env, "reset", side_effect=lambda *, seed, options: reset(seed=seed, options={**options, "max_ticks": 80})
        ):
            match = Match(Rival("bc", "hog_26", reference), seat=0, seed=42, number=1)
            [(xbow_run, free_run)] = play_wave(
                Path(folder), "ppo", policy.eval(), [match], [env], "cpu",
                collect=True, collect_both=True, capture_arena=False,
            )
        _, xbow_rows = xbow_run
        self.assertEqual(len(xbow_rows), 8)
        self.assertEqual(xbow_rows[5].shape[1:], (8, 32, 18))
        data = rollout_tensors([xbow_rows], "cpu")
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
        kl = optimize_ppo(policy.train(), reference, optimizer, data, batch_size=4, updates=2, kl_coef=0.1)
        self.assertIsNotNone(kl)
        self.assertTrue(np.isfinite(kl))

    def test_legacy_v2_policy_remains_backwards_compatible(self):
        env = battle_env("xbow_30", "hog_26", obs_profile="v2")
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=11)
        policy = XBowPolicy.from_observation(obs["blue"], env=env)
        self.assertEqual(policy.architecture, 2)
        self.assertEqual(policy.obs_profile, "v2")
        pred = decide_batch(policy.eval(), [(obs["blue"], None), (obs["red"], None)], "cpu")
        self.assertEqual(len(pred), 2)
        with TemporaryDirectory() as directory:
            ckpt = Path(directory) / "v2.pt"
            save_checkpoint(ckpt, policy, "xbow_30", "bc", {"nll": 0.9})
            loaded, saved = load_checkpoint(ckpt, "xbow_30", "cpu")
            self.assertEqual(saved["architecture"], 2)
            self.assertEqual(loaded.obs_profile, "v2")


class Phase2BCTests(unittest.TestCase):
    def test_mirror_sample_matches_real_mirrored_engine_state(self):
        env_a = battle_env("xbow_30", "hog_26", obs_profile="v3")
        env_b = battle_env("xbow_30", "hog_26", obs_profile="v3")
        self.addCleanup(env_a.close)
        self.addCleanup(env_b.close)
        setup = MatchSetup(
            decks=env_a.state_mutator.decks, shuffle=int(ShuffleMode.NONE),
            elixir_milli=[10000, 10000],
        )
        obs_a, _ = env_a.reset(seed=1, options={"setup": setup})
        obs_b, _ = env_b.reset(seed=1, options={"setup": setup})
        for _ in range(12):
            obs_a, *_ = env_a.step({"blue": 0, "red": 0})
            obs_b, *_ = env_b.step({"blue": 0, "red": 0})

        # Play mirrored cards: blue X-Bow at x=3 vs x=14; red Hog at x=3 vs x=14 (in red's frame).
        act_a_blue = env_a.action_parser.encode(0, 3, 13)
        act_b_blue = env_b.action_parser.encode(0, 14, 13)
        act_a_red = env_a.action_parser.encode(0, 3, 14)
        act_b_red = env_b.action_parser.encode(0, 14, 14)
        obs_a, *_ = env_a.step({"blue": act_a_blue, "red": act_a_red})
        obs_b, *_ = env_b.step({"blue": act_b_blue, "red": act_b_red})
        # Step enough ticks for troops to move (dx velocity nonzero) and damage asymmetric towers.
        for _ in range(14):
            obs_a, *_ = env_a.step({"blue": 0, "red": 0})
            obs_b, *_ = env_b.step({"blue": 0, "red": 0})

        spec = MirrorSpec.from_env(env_a)
        self.assertEqual(len(spec.spatial_swaps), 2)
        self.assertEqual(len(spec.vector_swaps), 2)

        a = obs_a["blue"]
        b = obs_b["blue"]
        sp_m, vec_m, mask_m, ids_m, target_m = mirror_sample(
            a["spatial"], a["vector"], a["action_mask"].astype(bool),
            pack_id_planes(a), act_a_blue, spec,
        )
        self.assertEqual(target_m, act_b_blue)
        np.testing.assert_allclose(vec_m, b["vector"], atol=1e-5)
        # Every non-king-center cell (king sits on centerline x=9.0 -> floor=9, mirrored to 8)
        # matches the real RustEngine mirrored battle bit-for-bit, including dx velocities!
        king_mask = np.ones((32, 18), dtype=bool)
        king_mask[3, 8:10] = False
        king_mask[29, 8:10] = False
        np.testing.assert_array_equal(ids_m[:, king_mask], pack_id_planes(b)[:, king_mask])
        names = env_a.obs_builder.channel_names()
        for ch, name in enumerate(names):
            if name == "enemy_troop_zone":
                continue  # Static king-tower footprint border has 1-tile floor parity.
            np.testing.assert_allclose(
                sp_m[ch][king_mask], b["spatial"][ch][king_mask], atol=1e-5,
                err_msg=f"channel {name} mismatch",
            )
        # Action mask matches across the entire active arena outside static King Tower footprint rows (1..4, 27..29).
        mask_m_grid = mask_m[1:1 + 4 * 32 * 18].reshape(4, 32, 18)
        mask_b_grid = b["action_mask"][1:1 + 4 * 32 * 18].astype(bool).reshape(4, 32, 18)
        np.testing.assert_array_equal(mask_m_grid[:, 5:27, :], mask_b_grid[:, 5:27, :])
        self.assertEqual(mask_m[0], bool(b["action_mask"][0]))

        # Double-mirroring must be the exact identity.
        sp_2, vec_2, mask_2, ids_2, target_2 = mirror_sample(sp_m, vec_m, mask_m, ids_m, target_m, spec)
        self.assertEqual(target_2, act_a_blue)
        np.testing.assert_array_equal(sp_2, a["spatial"])
        np.testing.assert_array_equal(vec_2, a["vector"])
        np.testing.assert_array_equal(mask_2, a["action_mask"].astype(bool))
        np.testing.assert_array_equal(ids_2, pack_id_planes(a))
        # Board value is also mirror-invariant.
        self.assertAlmostEqual(board_value(env_a.battle_state, 0), board_value(env_b.battle_state, 0), places=5)

    def test_competitive_filter_excludes_casual_but_keeps_validation_tags(self):
        import json
        from collections import Counter

        deck_cards = [{"card_key": k} for k in DECKS["xbow_30"]]
        deck_map = {frozenset(DECKS["xbow_30"]): "xbow_30"}

        def make_payload(battle_type):
            battle = {
                "team": {"players": [{"deck": deck_cards}]},
                "opponent": {"players": [{"deck": deck_cards}]},
            }
            if battle_type is not None:
                battle["battle_type"] = battle_type
            return json.dumps({"battle": battle, "events": []})

        counts = Counter()
        self.assertIsNotNone(eligible_replay("t1", "victory", make_payload("pathOfLegend"), deck_map, counts, competitive_only=True))
        self.assertIsNotNone(eligible_replay("t2", "victory", make_payload(None), deck_map, counts, competitive_only=True))
        self.assertIsNone(eligible_replay("t3", "victory", make_payload("friendly"), deck_map, counts, competitive_only=True))
        self.assertIsNone(eligible_replay("t4", "victory", make_payload("clanMate"), deck_map, counts, competitive_only=True))
        self.assertEqual(counts["casual_filtered"], 2)
        # Reserved validation tags are always kept even if casual.
        self.assertIsNotNone(eligible_replay("t3", "victory", make_payload("friendly"), deck_map, counts, competitive_only=True, keep_tags={"t3"}))

    def test_raw_disk_match_samples_roundtrip_and_values(self):
        env = battle_env("xbow_30", "hog_26", obs_profile="v3")
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=5)
        b = obs["blue"]
        sp = np.stack([b["spatial"].astype(np.float16)] * 3)
        vec = np.stack([b["vector"].astype(np.float16)] * 3)
        msk = np.stack([np.packbits(b["action_mask"])] * 3)
        ids = np.stack([pack_id_planes(b)] * 3)
        acts = np.array([0, 1, 0])
        vals = np.array([0.0, 0.25, -0.5], dtype=np.float16)
        sample = MatchSamples("raw-test", sp, vec, msk, acts, ids, vals)
        with TemporaryDirectory() as directory:
            with (Path(directory) / "raw.bin").open("w+b") as store:
                raw = RawDiskMatchSamples(sample, store)
                for idx in range(3):
                    s, v, m, i = raw.frame(idx)
                    np.testing.assert_array_equal(s, sp[idx])
                    np.testing.assert_array_equal(v, vec[idx])
                    np.testing.assert_array_equal(m, msk[idx])
                    np.testing.assert_array_equal(i, ids[idx])
                np.testing.assert_array_equal(raw.values, vals)
                ds = ReplayFrames([raw], b["action_mask"].size, include_values=True)
                self.assertAlmostEqual(float(ds[1][-1]), 0.25, places=3)

    def test_bc_optimizer_cosine_schedule_and_value_warmup_only_updates_critic(self):
        env = battle_env("xbow_30", "hog_26", obs_profile="v3")
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=3)
        policy = XBowPolicy.from_observation(obs["blue"], env=env)
        opt = bc_optimizer(policy, lr=3e-4, weight_decay=2e-2)
        self.assertEqual(len(opt.param_groups), 2)
        self.assertGreater(opt.param_groups[0]["weight_decay"], 0.0)
        self.assertEqual(opt.param_groups[1]["weight_decay"], 0.0)

        sched = bc_lr_scheduler(opt, total_steps=40, min_lr_ratio=0.05)
        lrs = []
        for _ in range(40):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        self.assertLess(lrs[0], lrs[2])  # Warmup rises
        self.assertAlmostEqual(lrs[-1], 3e-4 * 0.05, delta=1e-5)  # Cosine finishes near min_lr

        # Value warmup must change ONLY policy.value weights and leave trunk/actor untouched.
        b = obs["blue"]
        spatial = torch.as_tensor(np.stack([b["spatial"]] * 4))
        vector = torch.as_tensor(np.stack([b["vector"]] * 4))
        mask = torch.as_tensor(np.stack([b["action_mask"]] * 4), dtype=torch.bool)
        ids = torch.as_tensor(np.stack([pack_id_planes(b)] * 4))
        actions = torch.zeros(4, dtype=torch.long)
        logp = torch.zeros(4, dtype=torch.float32)
        adv = torch.zeros(4, dtype=torch.float32)
        returns = torch.tensor([1.0, -1.0, 0.5, -0.5], dtype=torch.float32)

        before_trunk = policy.stem[0].weight.detach().clone()
        before_actor = policy.wait_head.weight.detach().clone()
        before_value = policy.value[-1].weight.detach().clone()
        warmup_loss = warmup_value_head(
            policy, (spatial, vector, mask, actions, logp, ids, adv, returns), batch_size=2, updates=2,
        )
        self.assertTrue(np.isfinite(warmup_loss))
        torch.testing.assert_close(policy.stem[0].weight, before_trunk)
        torch.testing.assert_close(policy.wait_head.weight, before_actor)
        self.assertFalse(torch.equal(policy.value[-1].weight, before_value))


class Phase3PPOLeagueTests(unittest.TestCase):
    def test_reverse_kl_annealing_and_ppo_telemetry(self):
        # KL decay schedule
        self.assertAlmostEqual(scheduled_kl_coef(0.12, 0), 0.12)
        self.assertAlmostEqual(scheduled_kl_coef(0.12, 40, floor_ratio=0.1, half_life=40.0), 0.12 * 0.55)
        self.assertLess(scheduled_kl_coef(0.12, 400), 0.015)
        self.assertGreaterEqual(scheduled_kl_coef(0.12, 10000, floor_ratio=0.1), 0.012 - 1e-9)

        # Masked Reverse & Forward KL
        mask = torch.tensor([[True, True, True, False], [True, False, False, False]])
        p_logits = torch.tensor([[0.0, 1.0, -1.0, -1e9], [0.0, -1e9, -1e9, -1e9]])
        q_logits = torch.tensor([[0.5, -0.5, 0.2, -1e9], [0.0, -1e9, -1e9, -1e9]])
        dist_p = torch.distributions.Categorical(logits=p_logits, validate_args=False)
        self.assertAlmostEqual(float(masked_kl(dist_p, p_logits, mask, "reverse").item()), 0.0, places=5)
        self.assertAlmostEqual(float(masked_kl(dist_p, p_logits, mask, "forward").item()), 0.0, places=5)
        rev = float(masked_kl(dist_p, q_logits, mask, "reverse").item())
        fwd = float(masked_kl(dist_p, q_logits, mask, "forward").item())
        self.assertGreater(rev, 0.0)
        self.assertGreater(fwd, 0.0)
        self.assertNotAlmostEqual(rev, fwd, places=4)

        # Full PPO telemetry + temperature inference
        env = battle_env("xbow_30", "hog_26", reward_profile="potential", obs_profile="v3")
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=13)
        for _ in range(12):
            obs, *_ = env.step({"blue": 0, "red": 0})
        b = obs["blue"]
        policy = XBowPolicy.from_observation(b, env=env)
        reference = frozen_policy(policy)
        action, logp, val = policy.eval().decide(b, "cpu", temperature=0.45)
        self.assertTrue(b["action_mask"][action])
        self.assertTrue(np.isfinite(logp) and np.isfinite(val))

        spatial = torch.as_tensor(np.stack([b["spatial"]] * 4))
        vector = torch.as_tensor(np.stack([b["vector"]] * 4))
        tmask = torch.as_tensor(np.stack([b["action_mask"]] * 4), dtype=torch.bool)
        ids = torch.as_tensor(np.stack([pack_id_planes(b)] * 4))
        actions = torch.zeros(4, dtype=torch.long)
        old_logp = torch.full((4,), logp, dtype=torch.float32)
        adv = torch.tensor([1.0, -1.0, 0.5, -0.5], dtype=torch.float32)
        returns = torch.tensor([0.8, -0.8, 0.4, -0.4], dtype=torch.float32)
        opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)
        kl, stats = optimize_ppo(
            policy.train(), reference, opt,
            (spatial, vector, tmask, actions, old_logp, ids, adv, returns),
            batch_size=2, updates=2, kl_coef=0.1, kl_direction="reverse",
            ent_coef=0.012, wait_ent_coef=0.025, return_stats=True,
        )
        self.assertTrue(np.isfinite(kl))
        for key in ("policy_loss", "value_loss", "entropy", "wait_entropy", "approx_kl",
                    "clip_fraction", "explained_variance", "kl_coef", "epochs_completed"):
            self.assertIn(key, stats)
            self.assertTrue(np.isfinite(stats[key]))

    def test_conditional_temperature_preserves_wait_probability(self):
        env = battle_env("xbow_30", "hog_26", reward_profile="potential", obs_profile="v3")
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=21)
        for _ in range(14):
            obs, *_ = env.step({"blue": 0, "red": 0})
        b = obs["blue"]
        self.assertTrue(b["action_mask"][1:].any())
        policy = XBowPolicy.from_observation(b, env=env).eval()
        # Force a realistic Clash Royale distribution: P(wait) ~ 0.85, plays spread over many tiles.
        with torch.no_grad():
            policy.wait_head.bias.fill_(6.0)
        # Measure empirical wait rate at T=1.0 vs T=0.45 across 400 draws: must stay close!
        gen1 = torch.Generator().manual_seed(101)
        gen2 = torch.Generator().manual_seed(101)
        waits_t1 = sum(policy.decide(b, "cpu", generator=gen1, temperature=1.0)[0] == 0 for _ in range(400))
        waits_t45 = sum(policy.decide(b, "cpu", generator=gen2, temperature=0.45)[0] == 0 for _ in range(400))
        self.assertLess(abs(waits_t1 - waits_t45) / 400.0, 0.08)

    def test_pfsp_prioritizes_hardest_opponents(self):
        pfsp = PFSPPool()
        for _ in range(10):
            pfsp.record("easy", "win")
            pfsp.record("hard", "loss")
        weights = pfsp.weights(["easy", "hard", "unseen"])
        self.assertGreater(weights[1], weights[2])
        self.assertGreater(weights[2], weights[0])
        self.assertGreater(weights[0], 0.05)  # Exploration floor kept alive
        # With 6 meta decks, a single 0% WR deck must be capped at max_share (0.28)
        pfsp6 = PFSPPool({
            "pekka_bs": 0.0, "hog_26": 1.0, "splashyard": 1.0,
            "royal_giant": 0.9, "log_bait": 0.85, "golem_beatdown": 0.9,
        })
        w6 = pfsp6.weights(["pekka_bs", "hog_26", "splashyard", "royal_giant", "log_bait", "golem_beatdown"])
        self.assertAlmostEqual(float(w6.sum()), 1.0, places=6)
        self.assertLessEqual(float(w6.max()), 0.28 + 1e-6)
        self.assertEqual(int(np.argmax(w6)), 0)  # pekka_bs is still #1 priority

    def test_all_six_rival_decks_instantiate_cleanly(self):
        self.assertGreaterEqual(len(RIVAL_DECKS), 6)
        for rival_deck in RIVAL_DECKS:
            with self.subTest(rival_deck=rival_deck):
                env = battle_env("xbow_30", rival_deck, reward_profile="potential", obs_profile="v3")
                try:
                    obs, _ = env.reset(seed=21)
                    obs, rewards, *_ = env.step({"blue": 0, "red": 0})
                    self.assertEqual(obs["blue"]["spatial"].shape, (66, 32, 18))
                    self.assertEqual(rewards["blue"], -rewards["red"])
                finally:
                    env.close()

    def test_multiprocess_rollout_worker_pool_runs_matches_and_collects_trajectories(self):
        env = battle_env("xbow_30", "hog_26", reward_profile="potential", obs_profile="v3")
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=31)
        policy = XBowPolicy.from_observation(obs["blue"], env=env).eval()
        free = XBowPolicy.from_observation(obs["red"], env=env).eval()
        matches = [
            Match(Rival("hog_26", "hog_26", free), seat=0, seed=101, number=1),
            Match(Rival("log_bait", "log_bait", free), seat=1, seed=102, number=2),
        ]
        with TemporaryDirectory() as directory:
            with RolloutWorkerPool(2, "v3", "potential", Path(directory)) as pool:
                pool.publish_model("free", free, force=True)
                collected = list(pool.run_matches(
                    "xbow_30", "ppo", policy, matches,
                    collect=True, collect_both=True, capture_arena=False,
                    policy_key="xbow", force_sync_policy=True,
                    rival_keys={id(free): "free"},
                ))
        self.assertEqual(len(collected), 2)
        for _, ((x_stats, x_rows), (f_stats, f_rows)) in collected:
            self.assertGreater(x_stats["steps"], 10)
            self.assertEqual(x_stats["steps"], f_stats["steps"])
            self.assertEqual(len(x_rows), 8)
            self.assertEqual(x_rows[5].shape[1:], (8, 32, 18))
            self.assertEqual(x_rows[2].dtype, bool)


class LiveOverlayAndContinualLearningTests(unittest.TestCase):
    def test_live_obs_profile_and_persistent_unit_velocity_tracker(self):
        env = battle_env("xbow_30", "hog_26", obs_profile="live")
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=5)
        b = obs["blue"]
        self.assertEqual(b["spatial"].shape, (34, 32, 18))
        self.assertEqual(pack_id_planes(b).shape, (8, 32, 18))
        policy = XBowPolicy.from_observation(b, env=env)
        self.assertEqual(policy.obs_profile, "live")
        with TemporaryDirectory() as directory:
            ckpt = Path(directory) / "live.pt"
            save_checkpoint(ckpt, policy, "xbow_30", "bc", {"nll": 0.7})
            reloaded, _ = load_checkpoint(ckpt, "xbow_30", "cpu")
            self.assertEqual(reloaded.obs_profile, "live")

        # Verify UnitIdTracker preserves UIDs across frames so unit_velocity activates!
        feed = PublicFeed("xbow_30", obs_profile="live")
        self.addCleanup(feed.close)
        base_towers = {"hp": (4824, 3052, 3052), "max_hp": (4824, 3052, 3052), "crowns": 0, "king_active": False}
        hand = ("x-bow", "tesla", "archers", "knight")
        feed.update({
            "tick": 0, "reset": True, "hand": hand, "next_card": "skeletons",
            "own_elixir_milli": 6000, "own_towers": base_towers, "opponent_towers": base_towers,
            "units": [], "spells": [], "events": [],
        })
        feed.update({
            "tick": 20, "hand": hand, "next_card": "skeletons",
            "own_elixir_milli": 7000, "own_towers": base_towers, "opponent_towers": base_towers,
            "units": [{"side": "opponent", "card": "hog-rider", "position": [3.5, 20.0],
                       "hp": 1408, "max_hp": 1408, "kind": "ground"}],
            "spells": [], "events": [{"tick": 10, "side": "opponent", "card": "hog-rider"}],
        })
        moved = feed.update({
            "tick": 30, "hand": hand, "next_card": "skeletons",
            "own_elixir_milli": 7500, "own_towers": base_towers, "opponent_towers": base_towers,
            "units": [{"side": "opponent", "card": "hog-rider", "position": [3.5, 18.0],
                       "hp": 1408, "max_hp": 1408, "kind": "ground"}],
            "spells": [], "events": [],
        })
        names = list(feed.env.obs_builder.channel_names())
        dy_minus_idx = names.index("enemy_dy_minus")
        self.assertGreater(float(moved["spatial"][dy_minus_idx].max()), 0.0)
        summary = feed.public_summary()
        self.assertIn("HogRider", summary["enemy_queue"])
        self.assertGreaterEqual(summary["enemy_elixir"], 0.0)

    def test_live_recorder_overlay_projection_and_offline_awr_finetune(self):
        from xbow_capture_bridge import arena_position

        # Tile -> Window pixel -> Tile projection roundtrip
        px, py = tile_to_window_px(3, 13, 568, 896)
        tx, ty = arena_position((px, py), (0, 0, 568, 896))
        self.assertAlmostEqual(tx, 3.5, places=2)
        self.assertAlmostEqual(ty, 13.5, places=2)

        hand = ["x-bow", "tesla", "archers", "knight"]
        win_own = {"hp": [4824, 3052, 3052], "max_hp": [4824, 3052, 3052], "crowns": 1, "king_active": False}
        win_opp = {"hp": [4824, 0, 3052], "max_hp": [4824, 3052, 3052], "crowns": 0, "king_active": True}
        self.assertEqual(infer_match_result({"own_towers": win_own, "opponent_towers": win_opp}), "win")
        self.assertEqual(infer_match_result({"own_towers": win_opp, "opponent_towers": win_own}), "loss")

        with TemporaryDirectory() as directory:
            rec_dir = Path(directory)
            rec = LiveMatchRecorder(rec_dir, "xbow_30", "dummy.pt")
            base_t = {"hp": [4824, 3052, 3052], "max_hp": [4824, 3052, 3052], "crowns": 0, "king_active": False}
            # Record 1 winning match + 1 losing match, each with a real confirmed X-Bow play.
            for match_idx, (final_own, final_opp) in enumerate(((win_own, win_opp), (win_opp, win_own))):
                frames = [
                    {"tick": 0, "reset": True, "hand": hand, "next_card": "skeletons",
                     "own_elixir_milli": 6000, "own_towers": base_t, "opponent_towers": base_t,
                     "units": [], "spells": [], "events": []},
                    {"tick": 120, "hand": hand, "next_card": "skeletons",
                     "own_elixir_milli": 9000, "own_towers": base_t, "opponent_towers": base_t,
                     "units": [], "spells": [], "events": []},
                    {"tick": 130, "hand": ["skeletons", "tesla", "archers", "knight"], "next_card": "fireball",
                     "own_elixir_milli": 3000, "own_towers": final_own, "opponent_towers": final_opp,
                     "units": [{"side": "own", "card": "x-bow", "position": [3.5, 13.5],
                                "hp": 1330, "max_hp": 1330, "kind": "building"}],
                     "spells": [], "events": [{"tick": 122, "side": "own", "card": "x-bow"}]},
                ]
                for f in frames:
                    rec.record(f, {"kind": "wait", "action": 0, "probability": 0.9}, {"enemy_elixir": 6.0})
                rec.finish()

            paths = sorted(rec_dir.glob("*.jsonl"))
            self.assertEqual(len(paths), 2)
            samples, info = load_live_matches(paths, "xbow_30", "live")
            self.assertEqual(info["matches_used"], 2)
            self.assertEqual(info["wins"], 1)
            self.assertEqual(info["losses"], 1)
            self.assertEqual(info["plays"], 2)

            feed = PublicFeed("xbow_30", obs_profile="live")
            self.addCleanup(feed.close)
            first_obs, _ = feed.env.reset(seed=1)
            policy = XBowPolicy.from_observation(first_obs["blue"], env=feed.env)
            before = policy.stem[0].weight.detach().clone()
            metrics = finetune_from_live(
                policy, samples, mirror_spec=MirrorSpec.from_env(feed.env),
                epochs=2, batch_size=4, lr=1e-4, kl_coef=0.15,
            )
            self.assertTrue(np.isfinite(metrics["policy_loss"]))
            self.assertTrue(np.isfinite(metrics["value_loss"]))
            self.assertTrue(np.isfinite(metrics["kl_to_previous"]))
            self.assertFalse(torch.equal(policy.stem[0].weight, before))

    def test_dashboard_model_info_and_live_matches_summary(self):
        import json
        from xbow_dashboard import live_matches_summary, read_run_model_info

        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "exp1" / "xbow_30"
            run_dir.mkdir(parents=True)
            env = battle_env("xbow_30", "hog_26", obs_profile="live")
            self.addCleanup(env.close)
            obs, _ = env.reset(seed=11)
            policy = XBowPolicy.from_observation(obs["blue"], env=env)
            save_checkpoint(
                run_dir / "ppo.pt", policy, "xbow_30", "ppo",
                {"update": 3, "reward_profile": "potential", "mirror_augmentation": True},
                {"pfsp_wr": {"hog_26": 0.42, "log_bait": 0.65}},
            )
            info = read_run_model_info(run_dir)
            self.assertEqual(info["architecture"], 3)
            self.assertEqual(info["obs_profile"], "live")
            self.assertEqual(info["channels"], 34)
            self.assertEqual(info["reward_profile"], "potential")
            self.assertTrue(info["mirror_augmentation"])
            self.assertAlmostEqual(info["pfsp_wr"]["hog_26"], 0.42, places=3)

            live_dir = root / "live_matches"
            live_dir.mkdir()
            (live_dir / "match_001.jsonl").write_text(
                json.dumps({"kind": "header", "deck": "xbow_30", "started": "2026-01-01T12:00:00Z"}) + "\n"
                + json.dumps({"kind": "frame", "tick": 10}) + "\n"
                + json.dumps({"kind": "footer", "result": "win", "frames": 42}) + "\n",
                encoding="utf-8",
            )
            (live_dir / "match_002.jsonl").write_text(
                json.dumps({"kind": "header", "deck": "xbow_30", "started": "2026-01-01T12:10:00Z"}) + "\n"
                + json.dumps({"kind": "footer", "result": "loss", "frames": 18}) + "\n",
                encoding="utf-8",
            )
            summary = live_matches_summary(root)
            self.assertEqual(summary["count"], 2)
            self.assertEqual(summary["wins"], 1)
            self.assertEqual(summary["losses"], 1)
            self.assertEqual(summary["total_frames"], 60)
            self.assertEqual(len(summary["matches"]), 2)

    def test_dashboard_jobs_validates_obs_profile_batch_size_and_workers(self):
        import json
        from unittest.mock import patch
        from xbow_dashboard_jobs import DashboardJobs

        class DummyProc:
            pid = 999999
            def poll(self):
                return None

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "replays").mkdir()
            jobs = DashboardJobs(root)
            jobs.inventory.snapshot = lambda: {"status": "ready", "counts": {"xbow_30": 100, "icebow": 10}, "parts": 1}

            with patch("xbow_dashboard_jobs.subprocess.Popen", return_value=DummyProc()):
                job = jobs.launch({
                    "deck": "xbow_30", "mode": "new", "matches": 20, "epochs": 4,
                    "updates": 2, "games": 4, "evaluate": 0, "threads": 2,
                    "obs_profile": "live", "batch_size": 128, "rollout_workers": 2,
                })
            saved_job = json.loads((jobs.store / job["id"] / "job.json").read_text(encoding="utf-8"))
            cmd = saved_job["command"]
            self.assertIn("--obs-profile", cmd)
            self.assertEqual(cmd[cmd.index("--obs-profile") + 1], "live")
            self.assertEqual(cmd[cmd.index("--batch-size") + 1], "128")
            self.assertEqual(cmd[cmd.index("--rollout-workers") + 1], "2")
            self.assertEqual(job["parameters"]["obs_profile"], "live")
            self.assertEqual(job["parameters"]["batch_size"], 128)
            self.assertEqual(job["parameters"]["rollout_workers"], 2)

            jobs.child = None
            with patch("xbow_dashboard_jobs.process_alive", return_value=False):
                with self.assertRaises(ValueError):
                    jobs.launch({
                        "deck": "xbow_30", "mode": "new", "matches": 20, "epochs": 4,
                        "updates": 2, "games": 4, "evaluate": 0, "threads": 2,
                        "obs_profile": "invalid", "batch_size": 128, "rollout_workers": 1,
                    })
                with self.assertRaises(ValueError):
                    jobs.launch({
                        "deck": "xbow_30", "mode": "new", "matches": 20, "epochs": 4,
                        "updates": 2, "games": 4, "evaluate": 0, "threads": 2,
                        "obs_profile": "v2", "batch_size": 128, "rollout_workers": 1,
                    })
                with self.assertRaises(ValueError):
                    jobs.launch({
                        "deck": "xbow_30", "mode": "new", "matches": 20, "epochs": 4,
                        "updates": 2, "games": 4, "evaluate": 0, "threads": 2,
                        "obs_profile": "v3", "batch_size": 999, "rollout_workers": 1,
                    })

    def test_screen_vision_analyzer_connects_cleanly_to_capture_bridge_and_live_policy(self):
        import cv2
        from xbow_capture_bridge import CaptureBridge, REFERENCE_HEIGHT, REFERENCE_WIDTH
        from xbow_screen_vision import (
            ROI_ELIXIR_BAR, ROI_HAND_SLOTS, ROI_NEXT_CARD, ScreenVisionAnalyzer,
            fit_portrait_rect, load_atlas_templates,
        )

        # Widescreen 1920x1080 (Google Play Games fullscreen) -> centered 9:16 portrait area.
        left, top, w, h = fit_portrait_rect((0, 0, 1920, 1080))
        self.assertEqual((top, h), (0, 1080))
        self.assertAlmostEqual(w / h, REFERENCE_WIDTH / REFERENCE_HEIGHT, places=2)
        self.assertEqual(left, (1920 - w) // 2)

        templates = load_atlas_templates("xbow_30")
        self.assertEqual(len(templates), 8)

        # Synthesize a realistic 568x896 portrait frame with 7.0 elixir bar, 4 deck cards, next card & enemy HP bar.
        frame = np.full((REFERENCE_HEIGHT, REFERENCE_WIDTH, 3), (95, 75, 55), dtype=np.uint8)
        ex0, ey0, ex1, ey1 = [int(round(v * d)) for v, d in zip(
            ROI_ELIXIR_BAR, (REFERENCE_WIDTH, REFERENCE_HEIGHT, REFERENCE_WIDTH, REFERENCE_HEIGHT),
        )]
        frame[ey0:ey1, ex0:ex1] = (35, 30, 35)
        fill_end = ex0 + int(round((ex1 - ex0) * 0.7))
        frame[ey0:ey1, ex0:fill_end] = (230, 50, 220)  # Bright magenta BGR

        chosen = ["x-bow", "tesla", "knight", "archers", "skeletons"]
        for roi, card in zip((*ROI_HAND_SLOTS, ROI_NEXT_CARD), chosen):
            rx0, ry0, rx1, ry1 = [int(round(v * d)) for v, d in zip(
                roi, (REFERENCE_WIDTH, REFERENCE_HEIGHT, REFERENCE_WIDTH, REFERENCE_HEIGHT),
            )]
            ph, pw = ry1 - ry0, rx1 - rx0
            iy0, iy1 = ry0 + int(ph * 0.08), ry0 + int(ph * 0.78)
            ix0, ix1 = rx0 + int(pw * 0.10), rx0 + int(pw * 0.90)
            frame[iy0:iy1, ix0:ix1] = cv2.resize(templates[card], (ix1 - ix0, iy1 - iy0))

        # Paint a bright red enemy unit HP bar in the arena (x~200, y~320).
        frame[320:326, 185:220] = (40, 40, 240)

        analyzer = ScreenVisionAnalyzer("xbow_30")
        captured = analyzer.analyze(frame, timestamp_s=100.0)
        self.assertTrue(captured["in_game"])
        vs = captured["record"]["visual_state"]
        self.assertAlmostEqual(vs["elixir"], 7.0, delta=0.35)
        self.assertEqual(vs["hand"], chosen[:4])
        self.assertEqual(vs["next_card"], chosen[4])
        self.assertGreaterEqual(len(captured["detections"]), 1)

        # Verify full chain: ScreenVisionAnalyzer -> CaptureBridge -> PublicFeed(live) -> XBowPolicy.decide
        bridge = CaptureBridge("xbow_30")
        pub = bridge.push(captured)
        self.assertIsNotNone(pub)
        feed = PublicFeed("xbow_30", obs_profile="live")
        self.addCleanup(feed.close)
        obs = feed.update(pub)
        policy = XBowPolicy.from_observation(obs, env=feed.env).eval()
        action, logp, val = policy.decide(obs, "cpu", temperature=0.45)
        self.assertTrue(obs["action_mask"][action])
        self.assertTrue(np.isfinite(logp) and np.isfinite(val))

        # Verify 4x2 deck screen validator (with realistic blue UI + 58px dark Google Play Games left sidebar).
        from xbow_screen_vision import ROI_DECK_SCREEN_SLOTS, deck_copy_url, verify_deck_screen
        all_templates = load_atlas_templates(None)
        deck_frame = np.full((REFERENCE_HEIGHT, REFERENCE_WIDTH, 3), (110, 58, 28), dtype=np.uint8)
        expected_cards = list(DECKS["xbow_30"])
        for roi, card in zip(ROI_DECK_SCREEN_SLOTS, expected_cards, strict=True):
            rx0, ry0, rx1, ry1 = [int(round(v * d)) for v, d in zip(
                roi, (REFERENCE_WIDTH, REFERENCE_HEIGHT, REFERENCE_WIDTH, REFERENCE_HEIGHT),
            )]
            ph, pw = ry1 - ry0, rx1 - rx0
            iy0, iy1 = ry0 + int(ph * 0.12), ry0 + int(ph * 0.82)
            ix0, ix1 = rx0 + int(pw * 0.10), rx0 + int(pw * 0.90)
            deck_frame[iy0:iy1, ix0:ix1] = cv2.resize(all_templates[card], (ix1 - ix0, iy1 - iy0))
        # Prepend a 58px dark toolbar on the left, matching Google Play Games PC window layout.
        sidebar = np.full((REFERENCE_HEIGHT, 58, 3), 22, dtype=np.uint8)
        with_sidebar = np.concatenate([sidebar, deck_frame], axis=1)
        ok_check = verify_deck_screen(with_sidebar, "xbow_30")
        self.assertTrue(ok_check["valid"])
        self.assertEqual(ok_check["missing"], [])
        self.assertIn("27000008", deck_copy_url("xbow_30"))

        # Swap slot 1 (x-bow) with hog-rider -> validator must flag missing x-bow and wrong slot #1!
        rx0, ry0, rx1, ry1 = [int(round(v * d)) for v, d in zip(
            ROI_DECK_SCREEN_SLOTS[0], (REFERENCE_WIDTH, REFERENCE_HEIGHT, REFERENCE_WIDTH, REFERENCE_HEIGHT),
        )]
        ph, pw = ry1 - ry0, rx1 - rx0
        iy0, iy1 = ry0 + int(ph * 0.12), ry0 + int(ph * 0.82)
        ix0, ix1 = rx0 + int(pw * 0.10), rx0 + int(pw * 0.90)
        deck_frame[iy0:iy1, ix0:ix1] = cv2.resize(all_templates["hog-rider"], (ix1 - ix0, iy1 - iy0))
        bad_check = verify_deck_screen(np.concatenate([sidebar, deck_frame], axis=1), "xbow_30")
        self.assertFalse(bad_check["valid"])
        self.assertIn("x-bow", bad_check["missing"])
        self.assertEqual(bad_check["wrong_slots"][0]["detected"], "hog-rider")

        # Verify save_deck_templates + ensure_deck_by_clicks (switches from wrong deck to slot 2, never clicks in-game).
        from xbow_screen_vision import ensure_deck_by_clicks, norm_to_screen_px, save_deck_templates
        sx, sy = norm_to_screen_px((100, 50, 626, 896), 58, 568, 0.5, 0.5)
        self.assertEqual((sx, sy), (100 + 58 + 284, 50 + 448))
        with TemporaryDirectory() as tmpl_tmp:
            saved_paths = save_deck_templates(with_sidebar, "xbow_30", Path(tmpl_tmp))
            self.assertEqual(len(saved_paths), 8)
            bad_img = np.concatenate([sidebar, deck_frame], axis=1)
            clicks = []

            class FakeVision:
                def __init__(self, seq):
                    self.seq = list(seq)
                    self.idx = 0
                def grab_once(self):
                    out = self.seq[min(self.idx, len(self.seq) - 1)]
                    self.idx += 1
                    return out, (100, 50, out.shape[1], out.shape[0])

            # Sequence: initial lobby check (bad) -> after opening Decks (bad) -> after clicking slot 1 (good!)
            res = ensure_deck_by_clicks(
                FakeVision([bad_img, bad_img, with_sidebar]), "xbow_30",
                templates_dir=Path(tmpl_tmp), delay=0.0,
                click_fn=lambda x, y: clicks.append((x, y)),
            )
            self.assertTrue(res["ok"])
            self.assertTrue(res["switched"])
            self.assertEqual(res["slot_used"], 1)
            self.assertGreaterEqual(len(clicks), 4)  # Collection, Decks, Slot 1, Battle

            # Safety guard: refuses to click if currently inside an active match (elixir bar visible).
            in_game_clicks = []
            res_busy = ensure_deck_by_clicks(
                FakeVision([frame]), "xbow_30", delay=0.0,
                click_fn=lambda x, y: in_game_clicks.append((x, y)),
            )
            self.assertFalse(res_busy["ok"])
            self.assertEqual(in_game_clicks, [])

            # Verify LiveGameController web backend status & validation
            from xbow_live_controller import LiveGameController
            ctrl = LiveGameController(Path(tmpl_tmp))
            st = ctrl.status("xbow_30")
            self.assertEqual(st["templates_total"], 8)
            self.assertFalse(st["overlay"]["running"])
            self.assertTrue(ctrl.action("stop_overlay", {"deck": "xbow_30"})["ok"])
            self.assertTrue(ctrl.action("stop_auto_train", {"deck": "xbow_30"})["ok"])
            self.assertIn("auto_trainer", ctrl.status("xbow_30"))
            with self.assertRaises(ValueError):
                ctrl.action("start_overlay", {"deck": "xbow_30", "run_id": "../escape"})

    def test_live_auto_trainer_state_machine_anti_trophy_guard_and_awr_hot_reload(self):
        import cv2
        from xbow_live_trainer import LiveAutoTrainer
        from xbow_screen_vision import (
            REFERENCE_HEIGHT, REFERENCE_WIDTH, ROI_ELIXIR_BAR, ROI_HAND_SLOTS, ROI_NEXT_CARD,
            ScreenVisionAnalyzer, detect_screen_state, load_atlas_templates,
        )

        # Build synthetic frames for: unsafe lobby (trophy road), classic 1v1 lobby, mode picker, in-match, post-match.
        templates_dir = Path("calibration/templates")
        shield = cv2.imread(str(templates_dir / "ui" / "classic_1v1_shield.png"))
        card_banner = cv2.imread(str(templates_dir / "ui" / "classic_1v1_card.png"))
        self.assertIsNotNone(shield)
        self.assertIsNotNone(card_banner)

        def make_lobby(with_classic_shield: bool):
            f = np.full((REFERENCE_HEIGHT, REFERENCE_WIDTH, 3), (90, 50, 20), dtype=np.uint8)
            # Yellow/orange Battle button in (0.18..0.68, 0.72..0.85)
            f[int(0.73 * REFERENCE_HEIGHT):int(0.84 * REFERENCE_HEIGHT),
              int(0.20 * REFERENCE_WIDTH):int(0.66 * REFERENCE_WIDTH)] = (25, 195, 245)
            if with_classic_shield:
                sh, sw = shield.shape[:2]
                y0, x0 = int(0.73 * REFERENCE_HEIGHT), int(0.74 * REFERENCE_WIDTH)
                f[y0:y0 + sh, x0:x0 + sw] = shield
            return f

        unsafe_lobby = make_lobby(False)
        safe_lobby = make_lobby(True)

        mode_picker = np.full((REFERENCE_HEIGHT, REFERENCE_WIDTH, 3), (80, 40, 30), dtype=np.uint8)
        ch, cw = card_banner.shape[:2]
        mode_picker[240:240 + ch, 40:40 + cw] = card_banner

        post_match = np.full((REFERENCE_HEIGHT, REFERENCE_WIDTH, 3), (40, 35, 35), dtype=np.uint8)
        # Blue OK button at (0.36..0.64, 0.84..0.93) with white 'OK' text
        post_match[int(0.85 * REFERENCE_HEIGHT):int(0.92 * REFERENCE_HEIGHT),
                   int(0.38 * REFERENCE_WIDTH):int(0.62 * REFERENCE_WIDTH)] = (235, 145, 35)
        cv2.putText(post_match, "OK", (int(0.45 * REFERENCE_WIDTH), int(0.90 * REFERENCE_HEIGHT)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

        self.assertEqual(detect_screen_state(unsafe_lobby)["state"], "lobby")
        self.assertFalse(detect_screen_state(unsafe_lobby)["classic_1v1_selected"])
        self.assertTrue(detect_screen_state(safe_lobby)["classic_1v1_selected"])
        self.assertEqual(detect_screen_state(mode_picker)["state"], "mode_picker")
        self.assertEqual(detect_screen_state(post_match)["state"], "post_match")

        # Synthesize a valid in-match frame
        atlas = load_atlas_templates("xbow_30")
        match_f = np.full((REFERENCE_HEIGHT, REFERENCE_WIDTH, 3), (95, 75, 55), dtype=np.uint8)
        ex0, ey0, ex1, ey1 = [int(round(v * d)) for v, d in zip(
            ROI_ELIXIR_BAR, (REFERENCE_WIDTH, REFERENCE_HEIGHT, REFERENCE_WIDTH, REFERENCE_HEIGHT),
        )]
        match_f[ey0:ey1, ex0:ex1] = (35, 30, 35)
        match_f[ey0:ey1, ex0:ex0 + int((ex1 - ex0) * 0.8)] = (230, 50, 220)
        for roi, c_name in zip((*ROI_HAND_SLOTS, ROI_NEXT_CARD), ["x-bow", "tesla", "knight", "archers", "skeletons"]):
            rx0, ry0, rx1, ry1 = [int(round(v * d)) for v, d in zip(
                roi, (REFERENCE_WIDTH, REFERENCE_HEIGHT, REFERENCE_WIDTH, REFERENCE_HEIGHT),
            )]
            ph, pw = ry1 - ry0, rx1 - rx0
            iy0, iy1 = ry0 + int(ph * 0.08), ry0 + int(ph * 0.78)
            ix0, ix1 = rx0 + int(pw * 0.10), rx0 + int(pw * 0.90)
            match_f[iy0:iy1, ix0:ix1] = cv2.resize(atlas[c_name], (ix1 - ix0, iy1 - iy0))

        with TemporaryDirectory() as directory:
            tmp = Path(directory)
            env = battle_env("xbow_30", "hog_26", obs_profile="live")
            self.addCleanup(env.close)
            obs, _ = env.reset(seed=7)
            pol = XBowPolicy.from_observation(obs["blue"], env=env)
            ckpt = tmp / "ppo.pt"
            bc_ckpt = tmp / "bc.pt"
            out_ckpt = tmp / "ppo_live.pt"
            save_checkpoint(bc_ckpt, pol, "xbow_30", "bc", {"nll": 0.8})
            save_checkpoint(ckpt, pol, "xbow_30", "ppo", {"update": 1})

            class SeqVision:
                def __init__(self, frames):
                    self.frames = list(frames)
                    self.i = 0
                    self.current_hwnd = 12345
                    self.analyzer = ScreenVisionAnalyzer("xbow_30", templates_dir=None)
                def grab_once(self):
                    out = self.frames[min(self.i, len(self.frames) - 1)]
                    self.i += 1
                    return out, (0, 0, out.shape[1], out.shape[0])

            # 1) Hard anti-trophy guard: stuck on unsafe Trophy Road lobby -> pauses with paused_unsafe_mode!
            unsafe_clicks = []
            t_guard = LiveAutoTrainer(
                checkpoint=ckpt, reference=bc_ckpt, output=out_ckpt, deck="xbow_30",
                mode="classic_1v1", max_games=2, learn_every=1, ensure_deck=False,
                record_dir=tmp / "live1", state_file=tmp / "st1.json",
                loop_sleep_s=0.0, vision=SeqVision([unsafe_lobby] * 10),
                click_fn=lambda x, y: unsafe_clicks.append((x, y)),
            )
            rc = t_guard.run()
            self.assertEqual(rc, 0)
            self.assertEqual(t_guard.status, "paused_unsafe_mode")
            self.assertEqual(t_guard.games_played, 0)

            # 2) Background click no-op detection -> stops cleanly with needs_cursor_permission (exit code 3)!
            noop_trainer = LiveAutoTrainer(
                checkpoint=ckpt, reference=bc_ckpt, output=out_ckpt, deck="xbow_30",
                mode="classic_1v1", max_games=2, learn_every=1, ensure_deck=False,
                click_mode="background", record_dir=tmp / "live2", state_file=tmp / "st2.json",
                loop_sleep_s=0.001, vision=SeqVision([safe_lobby] * 20),
                click_fn=lambda x, y: None,
            )
            rc_noop = noop_trainer.run()
            self.assertEqual(rc_noop, 3)
            self.assertTrue(noop_trainer.needs_cursor_permission)
            self.assertEqual(noop_trainer.status, "needs_cursor_permission")

            # 3) Full cycle: mode_picker -> safe_lobby -> match (x3 frames) -> post_match -> finished!
            seq_clicks = []
            full_trainer = LiveAutoTrainer(
                checkpoint=ckpt, reference=bc_ckpt, output=out_ckpt, deck="xbow_30",
                mode="classic_1v1", max_games=1, learn_every=1, ensure_deck=False,
                record_dir=tmp / "live3", state_file=tmp / "st3.json",
                loop_sleep_s=0.0,
                vision=SeqVision([mode_picker, safe_lobby, match_f, match_f, match_f, post_match]),
                click_fn=lambda x, y: seq_clicks.append((x, y)),
            )
            rc_ok = full_trainer.run()
            self.assertEqual(rc_ok, 0)
            self.assertEqual(full_trainer.status, "finished")
            self.assertEqual(full_trainer.games_played, 1)
            self.assertGreaterEqual(len(seq_clicks), 2)
            self.assertEqual(len(list((tmp / "live3").glob("*.jsonl"))), 1)


if __name__ == "__main__":
    unittest.main()
