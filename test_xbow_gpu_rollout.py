"""Unit & integration tests for the AMD RX 6600 DirectML GPU + Xeon Rollout Engine."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from xbow_environment import battle_env, pack_id_planes
from xbow_gpu_rollout import (
    DmlPolicyRunner,
    GpuBatchedRolloutPool,
    create_rollout_pool,
    directml_available,
)
from xbow_rollout_pool import RolloutWorkerPool
from xbow_selfplay import Match, Rival
from xbow_training import XBowPolicy, frozen_policy


def _tiny_v3_policy() -> XBowPolicy:
    env = battle_env("xbow_30", "hog_26", obs_profile="v3")
    try:
        obs, _ = env.reset(seed=11)
        b = obs["blue"]
        pol = XBowPolicy(
            b["spatial"].shape[0],
            b["vector"].shape[0],
            len(b["action_mask"]),
            architecture=3,
            obs_profile="v3",
        ).eval()
        return pol
    finally:
        env.close()


class GpuRolloutTests(unittest.TestCase):
    def test_dml_policy_runner_matches_pytorch_logits_and_values(self):
        policy = _tiny_v3_policy()
        env = battle_env("xbow_30", "hog_26", obs_profile="v3")
        try:
            obs, _ = env.reset(seed=23)
            b = obs["blue"]
        finally:
            env.close()

        sp = torch.as_tensor(b["spatial"], dtype=torch.float32).unsqueeze(0)
        vec = torch.as_tensor(b["vector"], dtype=torch.float32).unsqueeze(0)
        msk = torch.as_tensor(b["action_mask"], dtype=torch.bool).unsqueeze(0)
        idp = torch.as_tensor(pack_id_planes(b), dtype=torch.int64).unsqueeze(0)
        with torch.no_grad():
            ref_l, ref_v = policy.logits_and_value(sp, vec, msk, id_planes=idp)

        with tempfile.TemporaryDirectory() as tmp:
            runner = DmlPolicyRunner(policy, Path(tmp), use_dml=directml_available(), use_fp16=True)
            try:
                ol, ov = runner.logits_and_value(
                    sp.numpy(), vec.numpy(), msk.numpy(), idp.numpy(),
                )
                finite = np.isfinite(ref_l.numpy())
                max_ldiff = float(np.max(np.abs(ref_l.numpy()[finite] - ol[finite])))
                max_vdiff = float(np.max(np.abs(ref_v.numpy() - ov)))
                self.assertLess(max_ldiff, 5e-2)
                self.assertLess(max_vdiff, 2e-2)

                # Verify batched sampling returns legal actions and finite logprobs/values
                rng = np.random.default_rng(7)
                acts, logps, vals = runner.decide_batch_arrays(
                    np.repeat(sp.numpy(), 4, axis=0),
                    np.repeat(vec.numpy(), 4, axis=0),
                    np.repeat(msk.numpy(), 4, axis=0),
                    np.repeat(idp.numpy(), 4, axis=0),
                    rng,
                )
                self.assertEqual(acts.shape, (4,))
                self.assertEqual(logps.shape, (4,))
                self.assertEqual(vals.shape, (4,))
                for a in acts:
                    self.assertTrue(bool(msk[0, int(a)]))
                self.assertTrue(np.all(np.isfinite(logps)))
                self.assertTrue(np.all(np.isfinite(vals)))
            finally:
                runner.close()

    def test_gpu_batched_rollout_pool_collect_eval_and_collect_both(self):
        policy = _tiny_v3_policy()
        rival_pol = frozen_policy(policy)
        matches_scripted = [
            Match(Rival("hog_push", "hog_26"), i % 2, 300 + i, i + 1)
            for i in range(4)
        ]
        matches_neural = [
            Match(Rival("self_rival", "xbow_30", rival_pol), i % 2, 500 + i, i + 1)
            for i in range(2)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            arena_dir = Path(tmp) / "missing_arena_dir" / "arena-live"
            prev_arena = os.environ.get("XBOW_ARENA")
            prev_arena_dir = os.environ.get("XBOW_ARENA_DIR")
            os.environ["XBOW_ARENA"] = "1"
            os.environ["XBOW_ARENA_DIR"] = str(arena_dir)
            try:
                with GpuBatchedRolloutPool(
                    2, "v3", "potential", Path(tmp),
                    envs_per_worker=2, use_dml=directml_available(),
                ) as pool:
                    # 1. collect=True against scripted rival
                    collected = list(pool.run_matches(
                        "xbow_30", "test_collect", policy, matches_scripted,
                        collect=True, capture_arena=False,
                    ))
                    self.assertEqual(len(collected), 4)
                    for _m, (stats, rows) in collected:
                        self.assertIn("result", stats)
                        self.assertGreater(stats["steps"], 0)
                        self.assertEqual(len(rows), 8)
                        self.assertEqual(len(rows[0]), stats["steps"])

                    # 2. collect=False evaluation (exercises WAIT-only skipping)
                    evaluated = list(pool.run_matches(
                        "xbow_30", "test_eval", policy, matches_scripted,
                        collect=False, capture_arena=False, force_sync_policy=False,
                    ))
                    self.assertEqual(len(evaluated), 4)
                    for _m, stats in evaluated:
                        self.assertIn("result", stats)
                        self.assertGreater(stats["steps"], 0)

                    # 3. collect_both=True neural vs neural WITH live arena capture enabled (dashboard mode)
                    both = list(pool.run_matches(
                        "xbow_30", "test_both", policy, matches_neural,
                        collect=True, collect_both=True, capture_arena=True,
                        force_sync_policy=False,
                    ))
                    self.assertEqual(len(both), 2)
                    self.assertTrue(arena_dir.is_dir())
                    for _m, ((s0, r0), (s1, r1)) in both:
                        self.assertGreater(len(r0), 0)
                        self.assertGreater(len(r1), 0)
                        self.assertIn("result", s0)
                        self.assertIn("result", s1)
                    self.assertGreater(pool.gpu_batches, 0)
                    self.assertGreater(pool.gpu_decisions, 0)
            finally:
                if prev_arena is None:
                    os.environ.pop("XBOW_ARENA", None)
                else:
                    os.environ["XBOW_ARENA"] = prev_arena
                if prev_arena_dir is None:
                    os.environ.pop("XBOW_ARENA_DIR", None)
                else:
                    os.environ["XBOW_ARENA_DIR"] = prev_arena_dir

    def test_create_rollout_pool_respects_disable_dml_env(self):
        prev = os.environ.get("XBOW_DISABLE_DML")
        try:
            os.environ["XBOW_DISABLE_DML"] = "1"
            self.assertFalse(directml_available())
            with tempfile.TemporaryDirectory() as tmp:
                pool = create_rollout_pool(2, "v3", "potential", Path(tmp))
                self.assertIsInstance(pool, RolloutWorkerPool)
        finally:
            if prev is None:
                os.environ.pop("XBOW_DISABLE_DML", None)
            else:
                os.environ["XBOW_DISABLE_DML"] = prev


if __name__ == "__main__":
    unittest.main()
