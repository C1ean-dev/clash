"""Persistent multiprocess rollout workers for parallel CPU simulation on multi-core machines."""

from __future__ import annotations

from contextlib import nullcontext
import multiprocessing as mp
from multiprocessing.connection import wait as wait_connections
import os
from pathlib import Path
import traceback

import numpy as np
import torch
from royalegym.protocol import MatchSetup

from xbow_arena import ArenaCapture, arena_enabled
from xbow_environment import battle_env
from xbow_metrics import check_training_stop
from xbow_replays import SEATS
from xbow_training import (
    DEFAULT_CARD_VOCAB, DEFAULT_UNIT_VOCAB, DEFAULT_V3_CHANNELS, XBowPolicy,
)


def default_rollout_workers(parallel_games: int, device: str) -> int:
    """Choose an efficient persistent worker count on CPU without oversubscribing physical cores."""
    if device != "cpu" or parallel_games <= 1:
        return 1
    cpus = os.cpu_count() or 4
    # Target ~half logical threads (physical cores on HT CPUs), capped at 12 and parallel_games.
    max_workers = max(1, min(12, cpus // 2))
    return min(int(parallel_games), max_workers)


def serialize_policy(policy: XBowPolicy) -> dict:
    """Extract lightweight CPU weights and architecture spec for IPC sync."""
    state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    return {
        "spec": policy.spec,
        "architecture": policy.architecture,
        "hand_start": policy.hand_start,
        "card_vocab": getattr(policy, "card_vocab", DEFAULT_CARD_VOCAB),
        "unit_vocab": getattr(policy, "unit_vocab", DEFAULT_UNIT_VOCAB),
        "channels": getattr(policy, "channels", DEFAULT_V3_CHANNELS),
        "obs_profile": policy.obs_profile,
        "placement_prior": state.get("placement_prior"),
        "state_dict": state,
    }


def _build_worker_policy(payload: dict) -> XBowPolicy:
    policy = XBowPolicy(
        *payload["spec"],
        architecture=payload["architecture"],
        placement_prior=payload.get("placement_prior"),
        hand_start=payload.get("hand_start", 0),
        card_vocab=payload.get("card_vocab", DEFAULT_CARD_VOCAB),
        unit_vocab=payload.get("unit_vocab", DEFAULT_UNIT_VOCAB),
        channels=payload.get("channels", DEFAULT_V3_CHANNELS),
        obs_profile=payload.get("obs_profile"),
    ).to(device="cpu", memory_format=torch.channels_last)
    policy.load_state_dict(payload["state_dict"])
    return policy.eval().requires_grad_(False)


def _pack_rows(rows):
    spatial, vector, mask, actions, logp, id_planes, advantage, returns = rows
    packed_mask = np.packbits(mask, axis=-1)
    return (
        spatial.astype(np.float16, copy=False),
        vector.astype(np.float16, copy=False),
        packed_mask,
        mask.shape[-1],
        actions,
        logp,
        id_planes,
        advantage,
        returns,
    )


def unpack_rows(packed):
    spatial, vector, packed_mask, action_count, actions, logp, id_planes, advantage, returns = packed
    mask = np.unpackbits(packed_mask, axis=-1, count=action_count).astype(bool)
    return (spatial, vector, mask, actions, logp, id_planes, advantage, returns)


def _worker_main(conn, obs_profile: str, reward_profile: str, output_dir: str):
    torch.set_num_threads(1)
    from xbow_selfplay import game_steps, scripted_move

    output = Path(output_dir)
    envs: dict[tuple[str, str], object] = {}
    models: dict[str, XBowPolicy] = {}

    def get_env(deck_blue: str, deck_red: str):
        key = (deck_blue, deck_red)
        env = envs.get(key)
        if env is None:
            env = battle_env(deck_blue, deck_red, reward_profile=reward_profile, obs_profile=obs_profile)
            envs[key] = env
        return env

    try:
        while True:
            try:
                message = conn.recv()
            except EOFError:
                break
            kind = message[0]
            if kind == "shutdown":
                break
            if kind == "load_model":
                _, key, payload = message
                existing = models.get(key)
                if existing is not None and existing.spec == payload["spec"] and existing.architecture == payload["architecture"]:
                    existing.load_state_dict(payload["state_dict"])
                    existing.eval()
                else:
                    models[key] = _build_worker_policy(payload)
                continue
            if kind == "run_match":
                _, index, job = message
                try:
                    env = get_env(job["deck_blue"], job["deck_red"])
                    policy = models[job["policy_key"]]
                    rival_key = job["rival_key"]
                    rival_policy = models[rival_key] if rival_key is not None else None
                    setup = MatchSetup(decks=job["setup_decks"]) if job["setup_decks"] is not None else None
                    show_live = job["capture_arena"] and arena_enabled()
                    capture_ctx = (
                        ArenaCapture(
                            output, job["stage"], env, rival=job["rival_name"],
                            seat=job["seat"], number=job["number"], **job["metadata"],
                        )
                        if show_live else nullcontext()
                    )
                    with capture_ctx as capture:
                        game = game_steps(
                            env, policy, job["seat"], job["seed"], "cpu",
                            collect=job["collect"], capture=capture,
                            scripted=rival_policy is None, setup=setup,
                            rival_policy=rival_policy if job["collect_both"] else None,
                        )
                        rng = np.random.default_rng([job["seed"], 1 - job["seat"]])
                        obs, generators = next(game)
                        own, foe = job["seat"], 1 - job["seat"]
                        while True:
                            decisions = [None, None]
                            decisions[own] = policy.decide(obs[SEATS[own]], "cpu", generator=generators[own])
                            if rival_policy is None:
                                action = scripted_move(env, foe, obs[SEATS[foe]], rng, job["rival_name"])
                                decisions[foe] = (action, None, None)
                            else:
                                decisions[foe] = rival_policy.decide(obs[SEATS[foe]], "cpu", generator=generators[foe])
                            try:
                                obs, generators = game.send(decisions)
                            except StopIteration as done:
                                result = done.value
                                break
                    if job["collect"]:
                        if job["collect_both"]:
                            (s0, r0), (s1, r1) = result
                            result = ((s0, _pack_rows(r0)), (s1, _pack_rows(r1)))
                        else:
                            s0, r0 = result
                            result = (s0, _pack_rows(r0))
                    conn.send(("result", index, result))
                except Exception:
                    conn.send(("error", index, traceback.format_exc()))
    finally:
        for env in envs.values():
            try:
                env.close()
            except Exception:
                pass
        conn.close()


class RolloutWorkerPool:
    """Manage persistent CPU simulation workers with lazy weight sync and dynamic scheduling."""

    def __init__(self, num_workers: int, obs_profile: str, reward_profile: str, output: Path):
        self.num_workers = max(1, int(num_workers))
        self.obs_profile = obs_profile
        self.reward_profile = reward_profile
        self.output = Path(output)
        self._ctx = mp.get_context("spawn")
        self._conns = []
        self._procs = []
        self._versions: list[dict[str, int]] = []
        self._model_versions: dict[str, int] = {}
        self._model_payloads: dict[str, dict] = {}
        self._model_refs: dict[str, tuple[int, int]] = {}

    def __enter__(self):
        for _ in range(self.num_workers):
            parent_conn, child_conn = self._ctx.Pipe()
            proc = self._ctx.Process(
                target=_worker_main,
                args=(child_conn, self.obs_profile, self.reward_profile, str(self.output)),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self._conns.append(parent_conn)
            self._procs.append(proc)
            self._versions.append({})
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        for conn in self._conns:
            try:
                conn.send(("shutdown",))
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
        for proc in self._procs:
            proc.join(timeout=3.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
        self._conns.clear()
        self._procs.clear()
        self._versions.clear()

    def publish_model(self, key: str, policy: XBowPolicy, *, force: bool = False):
        """Record a model version; workers receive the weights lazily before their next match."""
        current_ver = self._model_versions.get(key, 0)
        if not force and key in self._model_refs and self._model_refs[key] == (id(policy), current_ver):
            return
        new_ver = current_ver + 1
        self._model_versions[key] = new_ver
        self._model_payloads[key] = serialize_policy(policy)
        self._model_refs[key] = (id(policy), new_ver)

    def _sync_worker(self, worker_id: int, keys: tuple[str, ...]):
        conn = self._conns[worker_id]
        synced = self._versions[worker_id]
        for key in keys:
            if key is None:
                continue
            ver = self._model_versions[key]
            if synced.get(key) != ver:
                conn.send(("load_model", key, self._model_payloads[key]))
                synced[key] = ver

    def run_matches(self, own_deck: str, stage: str, policy: XBowPolicy, matches, *,
                    collect: bool = False, collect_both: bool = False,
                    capture_arena: bool = True, policy_key: str = "policy",
                    force_sync_policy: bool = True, rival_keys: dict[int, str] | None = None,
                    **metadata):
        """Run matches across persistent workers, yielding (match, result) as each completes."""
        self.publish_model(policy_key, policy, force=force_sync_policy)
        rival_key_by_obj: dict[int, str] = {}
        jobs = []
        capture_metadata = {**metadata, "joint": collect_both or metadata.get("joint", False)}
        for index, match in enumerate(matches):
            r_policy = match.rival.policy
            if r_policy is None:
                r_key = None
            elif rival_keys is not None and id(r_policy) in rival_keys:
                r_key = rival_keys[id(r_policy)]
            else:
                obj_id = id(r_policy)
                r_key = rival_key_by_obj.get(obj_id)
                if r_key is None:
                    r_key = f"rival:{match.rival.name}:{obj_id}"
                    rival_key_by_obj[obj_id] = r_key
                self.publish_model(r_key, r_policy, force=False)
            decks = (own_deck, match.rival.deck)
            if match.seat == 1:
                decks = decks[::-1]
            setup_decks = list(match.setup.decks) if match.setup is not None else None
            jobs.append((index, (policy_key, r_key), {
                "deck_blue": decks[0],
                "deck_red": decks[1],
                "own_deck": own_deck,
                "rival_name": match.rival.name,
                "rival_deck": match.rival.deck,
                "policy_key": policy_key,
                "rival_key": r_key,
                "seat": match.seat,
                "seed": match.seed,
                "number": match.number,
                "setup_decks": setup_decks,
                "collect": collect,
                "collect_both": collect_both,
                "stage": stage,
                "capture_arena": capture_arena,
                "metadata": capture_metadata,
            }))

        queue = iter(jobs)
        in_flight: dict[object, int] = {}

        def dispatch(worker_id: int) -> bool:
            try:
                index, needed_keys, job = next(queue)
            except StopIteration:
                return False
            self._sync_worker(worker_id, needed_keys)
            conn = self._conns[worker_id]
            conn.send(("run_match", index, job))
            in_flight[conn] = worker_id
            return True

        for worker_id in range(min(self.num_workers, len(jobs))):
            dispatch(worker_id)

        while in_flight:
            check_training_stop()
            ready = wait_connections(list(in_flight), timeout=0.1)
            for conn in ready:
                worker_id = in_flight.pop(conn)
                msg = conn.recv()
                if msg[0] == "error":
                    _, idx, tb = msg
                    raise RuntimeError(f"Erro no worker de simulacao (partida {idx + 1}):\n{tb}")
                _, idx, result = msg
                if collect:
                    if collect_both:
                        (s0, r0), (s1, r1) = result
                        result = ((s0, unpack_rows(r0)), (s1, unpack_rows(r1)))
                    else:
                        s0, r0 = result
                        result = (s0, unpack_rows(r0))
                dispatch(worker_id)
                yield matches[idx], result
