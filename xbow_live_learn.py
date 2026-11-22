"""Continual offline fine-tuning (AWR / DAgger + Reverse KL) from recorded live matches."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from xbow_environment import PPO_GAMMA, pack_id_planes
from xbow_live_overlay import infer_match_result
from xbow_public_feed import PublicFeed
from xbow_replays import DECKS
from xbow_training import (
    MirrorSpec, bc_optimizer, frozen_policy, load_checkpoint, masked_kl, mirror_sample,
    save_checkpoint,
)


@dataclass
class LiveSample:
    spatial: np.ndarray
    vector: np.ndarray
    mask: np.ndarray
    id_planes: np.ndarray | None
    action: int
    ret: float


def _frame_board_value(frame: dict) -> float:
    own = frame["own_towers"]
    opp = frame["opponent_towers"]
    crown_diff = (int(own.get("crowns", 0)) - int(opp.get("crowns", 0))) / 3.0
    own_t = sum(hp / max(1, m) for hp, m in zip(own["hp"], own["max_hp"]))
    opp_t = sum(hp / max(1, m) for hp, m in zip(opp["hp"], opp["max_hp"]))
    tower_diff = (own_t - opp_t) / 3.0
    return float(max(-1.0, min(1.0, 0.55 * crown_diff + 0.45 * tower_diff)))


def _infer_played_action(step: dict, next_step: dict | None, mask: np.ndarray,
                         parser) -> int | None:
    """Recover the actual action played at `step` (explicit, newly spawned unit, or matching suggestion)."""
    if "played_action" in step and step["played_action"] is not None:
        act = int(step["played_action"])
        return act if 0 <= act < len(mask) and mask[act] else None

    if next_step is None:
        return 0
    next_frame = next_step["frame"]
    own_events = [
        ev for ev in (next_frame.get("events") or ())
        if ev.get("side") == "own" and ev.get("card") is not None
    ]
    if not own_events:
        return 0

    card = own_events[0]["card"]
    hand = step["frame"].get("hand") or ()
    try:
        slot = list(hand).index(card)
    except ValueError:
        return None

    # Prefer a newly appearing own unit/spell of that card in the next frame.
    prev_positions = [
        tuple(u["position"]) for u in (step["frame"].get("units") or ())
        if u.get("side") == "own" and u.get("card") == card
    ]
    candidates = [
        u["position"] for u in (next_frame.get("units") or ())
        if u.get("side") == "own" and u.get("card") == card
    ] + [
        s["own_aim"] or s["position"] for s in (next_frame.get("spells") or ())
        if s.get("side") == "own" and s.get("card") == card
    ]
    best_pos = None
    best_dist = -1.0
    for pos in candidates:
        if not prev_positions:
            best_pos = pos
            break
        d = min((pos[0] - px) ** 2 + (pos[1] - py) ** 2 for px, py in prev_positions)
        if d > best_dist:
            best_dist = d
            best_pos = pos

    if best_pos is not None:
        x = max(0, min(parser.nx - 1, int(best_pos[0])))
        y = max(0, min(parser.ny - 1, int(best_pos[1])))
        action = parser.encode(slot, x, y)
        if mask[action]:
            return int(action)

    # Fallback: if the AI suggested the exact same card and it's legal, use that placement.
    sug = step.get("suggestion") or {}
    if sug.get("kind") == "play_card" and sug.get("card") == card:
        act = int(sug.get("action", 0))
        if 0 < act < len(mask) and mask[act]:
            return act
    return None


def load_live_matches(paths: list[Path], deck: str, obs_profile: str, *,
                      wins_only: bool = False, gamma: float = PPO_GAMMA) -> tuple[list[LiveSample], dict]:
    """Reconstruct PublicFeed observations and discounted returns from recorded live JSONL files."""
    feed = PublicFeed(deck, obs_profile=obs_profile)
    parser = feed.env.action_parser
    samples: list[LiveSample] = []
    summary = {"files": 0, "matches_used": 0, "wins": 0, "losses": 0, "draws": 0, "plays": 0}
    try:
        for path in paths:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            steps = [r for r in rows if r.get("kind") == "step"]
            if len(steps) < 2:
                continue
            summary["files"] += 1
            header = next((r for r in rows if r.get("kind") == "header"), {})
            if header.get("deck") and header["deck"] != deck:
                continue
            footer = next((r for r in reversed(rows) if r.get("kind") == "footer"), {})
            result = footer.get("result") or infer_match_result(steps[-1]["frame"])
            if result not in ("win", "loss", "draw"):
                continue
            if wins_only and result != "win":
                continue

            outcome = 1.0 if result == "win" else -1.0 if result == "loss" else 0.0
            summary["matches_used"] += 1
            summary[{"win": "wins", "loss": "losses", "draw": "draws"}[result]] += 1

            # Build observations sequentially through PublicFeed (tracks enemy elixir/cycle & unit velocity).
            match_obs = []
            for idx, step in enumerate(steps):
                frame = dict(step["frame"])
                if idx == 0:
                    frame["reset"] = True
                try:
                    obs = feed.update(frame)
                except ValueError:
                    match_obs = []
                    break
                match_obs.append(obs)
            if not match_obs:
                continue

            total = len(match_obs)
            for idx, (step, obs) in enumerate(zip(steps, match_obs, strict=True)):
                next_step = steps[idx + 1] if idx + 1 < total else None
                mask = obs["action_mask"].astype(bool)
                action = _infer_played_action(step, next_step, mask, parser)
                if action is None:
                    continue
                discount = gamma ** (total - 1 - idx)
                board_v = _frame_board_value(step["frame"])
                ret = float(max(-1.0, min(1.0, discount * outcome + (1.0 - discount) * board_v)))
                samples.append(LiveSample(
                    spatial=obs["spatial"].astype(np.float16, copy=True),
                    vector=obs["vector"].astype(np.float16, copy=True),
                    mask=mask.copy(),
                    id_planes=pack_id_planes(obs),
                    action=action,
                    ret=ret,
                ))
                if action != 0:
                    summary["plays"] += 1
    finally:
        feed.close()
    summary["samples"] = len(samples)
    return samples, summary


def finetune_from_live(policy, samples: list[LiveSample], *, mirror_spec: MirrorSpec | None = None,
                       anchor_policy=None, epochs: int = 3, batch_size: int = 64, lr: float = 5e-5,
                       kl_coef: float = 0.15, awr_beta: float = 0.5,
                       wins_only: bool = False, device: str = "cpu", seed: int = 42) -> dict:
    """Fine-tune policy via Advantage-Weighted Regression (AWR) or win-filtered BC + Reverse KL anchor."""
    if not samples:
        raise ValueError("Nenhuma amostra valida encontrada nas partidas ao vivo.")

    anchor = frozen_policy(anchor_policy if anchor_policy is not None else policy)
    optimizer = bc_optimizer(policy, lr=lr, weight_decay=1e-2)
    rng = np.random.default_rng(seed)

    # Precompute current critic baselines for advantage estimation.
    policy.eval()
    baselines = np.empty(len(samples), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start:start + batch_size]
            sp = torch.as_tensor(np.stack([s.spatial for s in chunk]), dtype=torch.float32, device=device)
            vec = torch.as_tensor(np.stack([s.vector for s in chunk]), dtype=torch.float32, device=device)
            msk = torch.as_tensor(np.stack([s.mask for s in chunk]), dtype=torch.bool, device=device)
            ids = (
                torch.as_tensor(np.stack([s.id_planes for s in chunk]), device=device)
                if chunk[0].id_planes is not None else None
            )
            _, vals = policy.logits_and_value(sp, vec, msk, id_planes=ids)
            baselines[start:start + len(chunk)] = vals.cpu().numpy()

    returns = np.array([s.ret for s in samples], dtype=np.float32)
    advantages = returns - baselines

    if wins_only:
        awr_weights = np.ones(len(samples), dtype=np.float32)
    else:
        # Use signed advantages (never z-score an all-loss batch into positive weights!)
        # and do NOT artificially boost non-wait actions (which would destroy P(wait) / elixir saving).
        clipped_adv = np.clip(advantages, -2.0, 2.0)
        raw_w = np.where(returns > 0.0, np.exp(clipped_adv / max(0.05, float(awr_beta))),
                         np.exp(np.minimum(clipped_adv, 0.0) / max(0.05, float(awr_beta))))
        awr_weights = np.clip(raw_w, 0.02, 6.0).astype(np.float32)
        mean_w = float(awr_weights.mean())
        if mean_w > 1e-6:
            # Cap normalization so an all-loss batch stays gently weighted instead of amplifying blunders.
            awr_weights = awr_weights / max(1.0, mean_w)

    sample_weights = awr_weights

    policy.train()
    pol_losses, val_losses, kl_losses = [], [], []
    for _ in range(epochs):
        order = rng.permutation(len(samples))
        for start in range(0, len(order), batch_size):
            batch_idx = order[start:start + batch_size]
            sp_list, vec_list, msk_list, id_list, act_list = [], [], [], [], []
            for i in batch_idx:
                s = samples[int(i)]
                sp_i, vec_i, msk_i, act_i = s.spatial, s.vector, s.mask, s.action
                id_i = s.id_planes if s.id_planes is not None else np.empty(0, dtype=np.uint8)
                if mirror_spec is not None and rng.random() < 0.5:
                    sp_i, vec_i, msk_i, id_i, act_i = mirror_sample(
                        sp_i, vec_i, msk_i, id_i, act_i, mirror_spec,
                    )
                sp_list.append(sp_i)
                vec_list.append(vec_i)
                msk_list.append(msk_i)
                if s.id_planes is not None:
                    id_list.append(id_i)
                act_list.append(act_i)

            sp_t = torch.as_tensor(np.stack(sp_list), dtype=torch.float32, device=device)
            vec_t = torch.as_tensor(np.stack(vec_list), dtype=torch.float32, device=device)
            msk_t = torch.as_tensor(np.stack(msk_list), dtype=torch.bool, device=device)
            ids_t = torch.as_tensor(np.stack(id_list), device=device) if id_list else None
            act_t = torch.as_tensor(act_list, dtype=torch.long, device=device)
            ret_t = torch.as_tensor(returns[batch_idx], dtype=torch.float32, device=device)
            w_t = torch.as_tensor(sample_weights[batch_idx], dtype=torch.float32, device=device)

            dist, value_pred = policy(sp_t, vec_t, msk_t, id_planes=ids_t)
            policy_loss = -(w_t * dist.log_prob(act_t)).mean()
            value_loss = 0.5 * F.smooth_l1_loss(value_pred, ret_t)
            loss = policy_loss + value_loss
            if kl_coef > 0:
                with torch.no_grad():
                    anchor_logits, _ = anchor.logits_and_value(sp_t, vec_t, msk_t, id_planes=ids_t)
                kl = masked_kl(dist, anchor_logits, msk_t, direction="reverse")
                loss = loss + kl_coef * kl
                kl_losses.append(float(kl.detach().item()))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            pol_losses.append(float(policy_loss.detach().item()))
            val_losses.append(float(value_loss.detach().item()))

    policy.eval()
    return {
        "policy_loss": round(float(np.mean(pol_losses)), 5),
        "value_loss": round(float(np.mean(val_losses)), 5),
        "kl_to_previous": round(float(np.mean(kl_losses)), 5) if kl_losses else 0.0,
        "samples": len(samples),
        "epochs": epochs,
        "mode": "wins_only_bc" if wins_only else "awr_offline_rl",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", choices=DECKS, default="xbow_30")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint inicial (bc.pt ou ppo.pt)")
    parser.add_argument("--reference", type=Path, default=None,
                        help="Checkpoint ancora opcional (ex.: bc.pt) para Reverse KL sem drift")
    parser.add_argument("--matches-dir", type=Path, default=Path("runs/live_matches"),
                        help="Pasta ou arquivo JSONL com partidas ao vivo gravadas")
    parser.add_argument("--output", type=Path, required=True, help="Caminho do checkpoint ajustado (.pt)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--kl-coef", type=float, default=0.15)
    parser.add_argument("--awr-beta", type=float, default=0.5)
    parser.add_argument("--wins-only", action="store_true",
                        help="Clona apenas partidas vencidas (DAgger) em vez de AWR em vitorias+derrotas")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    layout = torch.channels_last
    policy, saved = load_checkpoint(args.checkpoint, args.deck, "cpu", memory_format=layout)
    anchor_policy = None
    if args.reference is not None and args.reference.is_file():
        anchor_policy, _ = load_checkpoint(args.reference, args.deck, "cpu", memory_format=layout)
    paths = (
        [args.matches_dir]
        if args.matches_dir.is_file()
        else sorted(args.matches_dir.glob("*.jsonl"))
    )
    if not paths:
        raise SystemExit(f"Nenhuma partida gravada encontrada em {args.matches_dir}")

    samples, data_summary = load_live_matches(
        paths, args.deck, policy.obs_profile, wins_only=args.wins_only,
    )
    mirror_spec = None
    if not args.no_mirror:
        feed = PublicFeed(args.deck, obs_profile=policy.obs_profile)
        try:
            mirror_spec = MirrorSpec.from_env(feed.env)
        finally:
            feed.close()

    train_metrics = finetune_from_live(
        policy, samples, mirror_spec=mirror_spec, anchor_policy=anchor_policy, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, kl_coef=args.kl_coef,
        awr_beta=args.awr_beta, wins_only=args.wins_only, seed=args.seed,
    )
    merged_metrics = {
        **(saved.get("metrics") or {}),
        "live_finetune": {**data_summary, **train_metrics},
    }
    save_checkpoint(args.output, policy, args.deck, saved["phase"], merged_metrics, saved.get("trainer"))
    print(json.dumps({"output": str(args.output), "data": data_summary, "train": train_metrics}, indent=2))


if __name__ == "__main__":
    main()
