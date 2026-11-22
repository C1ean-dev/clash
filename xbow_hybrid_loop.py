"""Hybrid Live <-> Synthetic Policy Loop for Clash Royale X-Bow / IceBow.

Closes the continual self-improvement cycle between real matches and RoyaleSim:
  1. Live Meta Analysis (`analyze_live_meta`): scans `runs/live_matches/*.jsonl` to identify
     the opponent archetypes and cards that most frequently beat the live agent (greatly enriched
     by real enemy card names from Phase 1 YOLO detection).
  2. Live AWR Consolidation: safely updates weights from positive/winning live trajectories via
     Advantage-Weighted Regression anchored to `bc.pt` with Reverse KL.
  3. Meta-Prioritized Synthetic PPO (`RoyaleSim`): starts from the live-adapted checkpoint and
     drills the exact 6-archetype meta + self-play league using a PFSP pool seeded by live losses.
  4. Matched-Seed A/B Promotion Gate (`evaluate_ab_in_sim`): compares incumbent vs. candidate on
     identical seeds, seats, and rivals before atomically promoting to `ppo_live.pt` (creating a
     `.bak.pt` backup so live performance never regresses).
  5. Optional Companion Daemon (`daemon`): watches `runs/live_matches/` and triggers hybrid cycles
     automatically every N new matches while `xbow_live_trainer.py` hot-reloads updated weights.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import batched
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter, sleep

import numpy as np
import torch

from xbow_environment import RIVAL_DECKS, battle_env
from xbow_live_learn import finetune_from_live, load_live_matches
from xbow_live_overlay import infer_match_result
from xbow_public_feed import PublicFeed
from xbow_replays import DECKS
from xbow_selfplay import (
    Match,
    PFSPPool,
    Rival,
    play_wave,
    rollout_tensors,
    summarize,
)
from xbow_training import (
    MirrorSpec,
    frozen_policy,
    load_checkpoint,
    optimize_ppo,
    save_checkpoint,
)

# All 6 competitive RoyaleSim archetypes paired with their tactical opponent profile in xbow_selfplay.
ARCHETYPE_RIVALS: tuple[tuple[str, str], ...] = (
    ("hog_push", "hog_26"),
    ("rg_patient", "royal_giant"),
    ("splashyard_defend", "splashyard"),
    ("log_bait_push", "log_bait"),
    ("golem_patient", "golem_beatdown"),
    ("pekka_push", "pekka_bs"),
)

# Signature cards carry extra weight when matching noisy live enemy observations to RoyaleSim archetypes.
SIGNATURE_CARDS: dict[str, frozenset[str]] = {
    "hog_26": frozenset({"hog-rider", "musketeer", "cannon", "ice-golem"}),
    "royal_giant": frozenset({"royal-giant", "fisherman", "hunter", "royal-ghost"}),
    "splashyard": frozenset({"graveyard", "baby-dragon", "ice-wizard", "tombstone", "poison"}),
    "log_bait": frozenset({"goblin-barrel", "princess", "goblin-gang", "inferno-tower", "rocket"}),
    "golem_beatdown": frozenset({"golem", "night-witch", "lumberjack", "mega-minion", "lightning"}),
    "pekka_bs": frozenset({"pekka", "battle-ram", "bandit", "electro-wizard", "magic-archer"}),
}


def classify_enemy_archetype(enemy_cards: Counter[str]) -> str | None:
    """Map observed enemy cards from a live match to the closest RoyaleSim rival name."""
    if not enemy_cards:
        return None
    # Ignore generic HP-bar fallback 'knight'; require at least one specific (YOLO) enemy card
    # so pre-YOLO HP-bar-only recordings never bias archetypes containing knight.
    observed = {c for c in enemy_cards if c != "knight"}
    if not observed:
        return None

    best_rival = None
    best_score = 0.0
    for rival_name, deck_key in ARCHETYPE_RIVALS:
        deck_cards = set(RIVAL_DECKS[deck_key])
        overlap = observed & deck_cards
        if not overlap:
            continue
        sig_hits = len(observed & SIGNATURE_CARDS.get(deck_key, frozenset()))
        score = len(overlap) + 1.75 * sig_hits
        if score > best_score:
            best_score = score
            best_rival = rival_name
    return best_rival


def analyze_live_meta(
    matches_dir: Path = Path("runs/live_matches"),
    *,
    deck: str = "xbow_30",
    max_files: int = 80,
) -> dict:
    """Analyze recorded live matches to extract per-archetype win rates and PFSP training seeds."""
    matches_dir = Path(matches_dir)
    paths = (
        [matches_dir]
        if matches_dir.is_file()
        else (sorted(matches_dir.glob("*.jsonl"))[-max_files:] if matches_dir.is_dir() else [])
    )

    overall = Counter()
    enemy_cards_total: Counter[str] = Counter()
    arch_results: dict[str, Counter[str]] = defaultdict(Counter)
    classified_matches = 0

    for path in paths:
        try:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError:
            continue
        steps = [r for r in rows if r.get("kind") == "step"]
        if len(steps) < 2:
            continue
        header = next((r for r in rows if r.get("kind") == "header"), {})
        if header.get("deck") and header["deck"] != deck:
            continue
        footer = next((r for r in reversed(rows) if r.get("kind") == "footer"), {})
        result = footer.get("result") or infer_match_result(steps[-1]["frame"])
        if result not in ("win", "loss", "draw"):
            continue
        overall[result] += 1

        match_enemy_cards: Counter[str] = Counter()
        for step in steps:
            frame = step.get("frame") or {}
            for u in frame.get("units") or ():
                if u.get("side") == "opponent" and u.get("card"):
                    match_enemy_cards[str(u["card"])] += 1
            for ev in frame.get("events") or ():
                if ev.get("side") == "opponent" and ev.get("card"):
                    match_enemy_cards[str(ev["card"])] += 1
        for card, cnt in match_enemy_cards.items():
            if card != "knight" or len(match_enemy_cards) == 1:
                enemy_cards_total[card] += cnt

        arch = classify_enemy_archetype(match_enemy_cards)
        if arch is not None:
            arch_results[arch][result] += 1
            classified_matches += 1

    # Build PFSP initial win-rate seeds: harder live archetypes get lower EMA -> higher synthetic sampling weight!
    archetype_stats: dict[str, dict] = {}
    pfsp_seed: dict[str, float] = {}
    for rival_name, deck_key in ARCHETYPE_RIVALS:
        c = arch_results.get(rival_name, Counter())
        total = int(c["win"] + c["loss"] + c["draw"])
        if total > 0:
            raw_wr = (c["win"] + 0.5 * c["draw"]) / total
            # Shrink gently toward 0.5 for small sample sizes so 1 lucky/unlucky game doesn't overfit.
            shrunk = (raw_wr * total + 0.5 * 2.0) / (total + 2.0)
            pfsp_seed[rival_name] = round(max(0.12, min(0.85, shrunk)), 4)
        else:
            raw_wr = None
            pfsp_seed[rival_name] = 0.45
        archetype_stats[rival_name] = {
            "deck": deck_key,
            "wins": int(c["win"]),
            "losses": int(c["loss"]),
            "draws": int(c["draw"]),
            "total": total,
            "win_rate": round(raw_wr, 4) if raw_wr is not None else None,
            "pfsp_seed_wr": pfsp_seed[rival_name],
        }

    total_games = int(overall["win"] + overall["loss"] + overall["draw"])
    return {
        "matches_scanned": total_games,
        "classified_matches": classified_matches,
        "overall": {
            "wins": int(overall["win"]),
            "losses": int(overall["loss"]),
            "draws": int(overall["draw"]),
            "win_rate": round((overall["win"] + 0.5 * overall["draw"]) / total_games, 4) if total_games else None,
        },
        "top_enemy_cards": enemy_cards_total.most_common(12),
        "archetypes": archetype_stats,
        "pfsp_seed": pfsp_seed,
    }


def _run_matches_local(
    output: Path,
    stage: str,
    deck: str,
    policy,
    matches: list[Match],
    *,
    obs_profile: str,
    reward_profile: str,
    parallel_games: int,
    device: str,
    collect: bool,
    pool=None,
) -> list[tuple[Match, tuple[dict, object]]]:
    if pool is not None and len(matches) > 1:
        return list(pool.run_matches(
            deck, stage, policy, matches,
            collect=collect, capture_arena=False,
            policy_key=f"{stage}:{id(policy)}", force_sync_policy=True,
        ))
    envs: dict[tuple[str, str], list] = defaultdict(list)
    results_out: list[tuple[Match, tuple[dict, object]]] = []
    try:
        for wave in batched(matches, max(1, int(parallel_games))):
            used = Counter()
            wave_envs = []
            for match in wave:
                decks = (deck, match.rival.deck)
                if match.seat == 1:
                    decks = decks[::-1]
                slot = used[decks]
                if slot == len(envs[decks]):
                    envs[decks].append(
                        battle_env(*decks, reward_profile=reward_profile, obs_profile=obs_profile)
                    )
                wave_envs.append(envs[decks][slot])
                used[decks] += 1
            wave_res = play_wave(
                output, stage, policy, wave, wave_envs, device,
                collect=collect, capture_arena=False,
            )
            results_out.extend(zip(wave, wave_res, strict=True))
    finally:
        for env_list in envs.values():
            for env in env_list:
                try:
                    env.close()
                except Exception:
                    pass
    return results_out


def _policy_summary_score(games: list[dict]) -> dict:
    s = summarize(games)
    total = max(1, int(s["games"]))
    win_score = (s["wins"] + 0.5 * s["draws"]) / total
    avg_plays = float(np.mean([g.get("plays", 0) for g in games])) if games else 0.0
    avg_steps = float(np.mean([max(1, g.get("steps", 1)) for g in games])) if games else 1.0
    play_rate = avg_plays / max(1.0, avg_steps)
    return {
        **s,
        "win_score": round(win_score, 4),
        "avg_plays": round(avg_plays, 2),
        "play_rate": round(play_rate, 4),
    }


def evaluate_ab_in_sim(
    incumbent,
    candidate,
    deck: str,
    eval_rivals: list[Rival],
    output: Path,
    *,
    games_per_rival: int = 2,
    seed: int = 2026,
    reward_profile: str = "potential",
    parallel_games: int = 4,
    device: str = "cpu",
    min_delta_score: float = -0.02,
    pool=None,
) -> dict:
    """Matched-seed head-to-head A/B evaluation of candidate vs. incumbent in RoyaleSim."""
    rng = np.random.default_rng(seed)
    matches: list[Match] = []
    number = 0
    for rival in eval_rivals:
        for g in range(max(1, int(games_per_rival))):
            number += 1
            matches.append(Match(rival, g % 2, int(rng.integers(2**31)), number))

    obs_profile = candidate.obs_profile
    inc_raw = _run_matches_local(
        output, "hybrid_eval_incumbent", deck, incumbent, matches,
        obs_profile=obs_profile, reward_profile=reward_profile,
        parallel_games=parallel_games, device=device, collect=False, pool=pool,
    )
    cand_raw = _run_matches_local(
        output, "hybrid_eval_candidate", deck, candidate, matches,
        obs_profile=obs_profile, reward_profile=reward_profile,
        parallel_games=parallel_games, device=device, collect=False, pool=pool,
    )

    inc_games = [game for _, game in inc_raw]
    cand_games = [game for _, game in cand_raw]
    inc_sum = _policy_summary_score(inc_games)
    cand_sum = _policy_summary_score(cand_games)

    per_rival = {}
    for rival in eval_rivals:
        ig = [game for m, game in inc_raw if m.rival.name == rival.name]
        cg = [game for m, game in cand_raw if m.rival.name == rival.name]
        per_rival[rival.name] = {
            "incumbent_score": _policy_summary_score(ig)["win_score"],
            "candidate_score": _policy_summary_score(cg)["win_score"],
        }

    delta_score = round(cand_sum["win_score"] - inc_sum["win_score"], 4)
    delta_reward = round(cand_sum["mean_reward"] - inc_sum["mean_reward"], 4)

    # Guard against policy collapse (spamming every frame or freezing and never playing cards).
    play_rate_ok = 0.015 <= cand_sum["play_rate"] <= 0.35
    score_ok = delta_score >= float(min_delta_score) and (
        delta_score > 0.0 or delta_reward >= -0.25
    )
    passed = bool(play_rate_ok and score_ok)
    if not play_rate_ok:
        reason = f"play_rate fora do intervalo seguro ({cand_sum['play_rate']:.4f})"
    elif not score_ok:
        reason = f"regressao no A/B (delta_score={delta_score:+.4f}, delta_reward={delta_reward:+.4f})"
    else:
        reason = f"aprovado no A/B (delta_score={delta_score:+.4f}, delta_reward={delta_reward:+.4f})"

    return {
        "games_evaluated": len(matches),
        "incumbent": inc_sum,
        "candidate": cand_sum,
        "per_rival": per_rival,
        "delta_score": delta_score,
        "delta_reward": delta_reward,
        "passed_gate": passed,
        "reason": reason,
    }


def _atomic_promote_checkpoint(
    dst: Path,
    policy,
    deck: str,
    phase: str,
    metrics: dict,
    trainer: dict | None = None,
) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file():
        shutil.copy2(dst, dst.with_suffix(".bak.pt"))
    tmp = dst.with_suffix(".pt.tmp")
    save_checkpoint(tmp, policy, deck, phase, metrics, trainer)
    tmp.replace(dst)


def run_hybrid_cycle(
    *,
    deck: str = "xbow_30",
    checkpoint: Path,
    bc_checkpoint: Path | None = None,
    output: Path | None = None,
    matches_dir: Path = Path("runs/live_matches"),
    skip_awr: bool = False,
    ppo_updates: int = 3,
    rollout_games: int = 8,
    batch_size: int = 256,
    lr: float = 6e-5,
    kl_coef: float = 0.20,
    eval_games_per_rival: int = 2,
    eval_rival_limit: int | None = None,
    parallel_games: int = 4,
    reward_profile: str = "potential",
    threads: int = 6,
    seed: int = 42,
    force_promote: bool = False,
    also_vision: bool = False,
    vision_dataset_dir: Path = Path("runs/vision_dataset"),
) -> dict:
    """Execute one complete Live AWR -> Meta-Prioritized Synthetic PPO -> A/B Gate -> Promote cycle."""
    began = perf_counter()
    checkpoint = Path(checkpoint)
    output = Path(output) if output is not None else checkpoint
    if bc_checkpoint is None:
        candidate_bc = checkpoint.parent / "bc.pt"
        bc_checkpoint = candidate_bc if candidate_bc.is_file() else None
    else:
        bc_checkpoint = Path(bc_checkpoint)

    torch.set_num_threads(max(1, int(threads)))
    torch.manual_seed(int(seed))
    device = "cpu"
    layout = torch.channels_last

    incumbent, saved_ckpt = load_checkpoint(checkpoint, deck, device, memory_format=layout)
    incumbent_frozen = frozen_policy(incumbent)
    policy, _ = load_checkpoint(checkpoint, deck, device, memory_format=layout)
    if bc_checkpoint is not None and bc_checkpoint.is_file():
        bc_policy, _ = load_checkpoint(bc_checkpoint, deck, device, memory_format=layout)
        reference = frozen_policy(bc_policy)
    else:
        reference = incumbent_frozen

    # 1. Analyze live meta, sync discovered player decks, and run optional Live AWR consolidation
    from xbow_deck_discovery import DiscoveredDeckStore
    deck_store = DiscoveredDeckStore()
    deck_store.sync_from_live_matches(matches_dir, deck=deck)
    discovered_rows = deck_store.active_rival_decks(max_decks=6)
    discovered_pfsp = deck_store.pfsp_seeds(max_decks=6)
    meta = analyze_live_meta(matches_dir, deck=deck)
    awr_report = {"skipped": True, "reason": "skip_awr=True" if skip_awr else "no_live_matches"}
    if not skip_awr and matches_dir.is_dir():
        paths = sorted(matches_dir.glob("*.jsonl"))[-40:]
        if paths:
            samples, data_summary = load_live_matches(paths, deck, policy.obs_profile)
            if len(samples) >= 8 and (data_summary.get("wins", 0) > 0 or any(s.ret > 0.05 for s in samples)):
                feed = PublicFeed(deck, obs_profile=policy.obs_profile)
                try:
                    mirror_spec = MirrorSpec.from_env(feed.env)
                finally:
                    feed.close()
                awr_metrics = finetune_from_live(
                    policy, samples, mirror_spec=mirror_spec, anchor_policy=reference,
                    epochs=2, batch_size=64, lr=5e-5, kl_coef=0.15, awr_beta=0.5, seed=seed,
                )
                awr_report = {"skipped": False, **data_summary, **awr_metrics}
            else:
                awr_report = {
                    "skipped": True,
                    "reason": "amostras insuficientes ou sem vitorias/lances positivos",
                    **data_summary,
                }

    # 2. Build 6-archetype meta + discovered player decks + self-play league seeded with live weaknesses
    other_deck = "icebow" if deck == "xbow_30" else "xbow_30"
    meta_rivals = [Rival(rname, dkey) for rname, dkey in ARCHETYPE_RIVALS]
    discovered_rivals: list[Rival] = []
    pfsp_initial = dict(meta["pfsp_seed"])
    for row in discovered_rows:
        did = str(row["id"])
        style = str(row.get("play_style") or "push")
        rname = f"dyn_{style}_{did}"
        discovered_rivals.append(Rival(rname, did))
        pfsp_initial[rname] = discovered_pfsp.get(did, 0.40)

    league: list[Rival] = [
        Rival("bc", deck, reference),
        Rival("incumbent_live", deck, incumbent_frozen),
        Rival("scripted", other_deck),
        *meta_rivals,
        *discovered_rivals,
    ]
    league_dir = output.parent / "league"
    if league_dir.is_dir():
        for snap in sorted(league_dir.glob("*.pt"))[-2:]:
            try:
                prev_pol, _ = load_checkpoint(snap, deck, device, memory_format=layout)
                if prev_pol.obs_profile == policy.obs_profile:
                    league.append(Rival(f"snapshot_{snap.stem}", deck, frozen_policy(prev_pol)))
            except Exception:
                continue

    pfsp = PFSPPool(initial_wr=pfsp_initial, alpha=0.25, floor=0.15, max_share=0.28)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(lr))
    rng = np.random.default_rng(int(seed))

    from xbow_gpu_rollout import create_rollout_pool, directml_available
    use_pool = (rollout_games >= 4 or eval_games_per_rival >= 2) and (parallel_games > 1 or directml_available())
    pool_ctx = (
        create_rollout_pool(
            max(2, parallel_games), policy.obs_profile, reward_profile, output.parent,
            parallel_games=max(parallel_games, rollout_games),
        )
        if use_pool else None
    )

    sim_updates: list[dict] = []
    sorted_meta_rivals = sorted(meta_rivals, key=lambda r: meta["pfsp_seed"].get(r.name, 0.5))
    if eval_rival_limit is not None and int(eval_rival_limit) > 0:
        sorted_meta_rivals = sorted_meta_rivals[:int(eval_rival_limit)]
    eval_rivals = [Rival("incumbent_live", deck, incumbent_frozen), *sorted_meta_rivals]

    if pool_ctx is not None:
        pool_ctx.__enter__()
    try:
        # 3. Synthetic PPO reinforcement updates in RoyaleSim (RX 6600 DirectML + multi-worker Xeon)
        for upd in range(max(0, int(ppo_updates))):
            u_began = perf_counter()
            rival_names = [r.name for r in league]
            sampled = pfsp.sample_indices(rival_names, max(1, int(rollout_games)), rng)
            matches = [
                Match(league[idx], (upd + i) % 2, int(rng.integers(2**31)), i + 1)
                for i, idx in enumerate(sampled)
            ]
            played = _run_matches_local(
                output.parent, "hybrid_ppo", deck, policy, matches,
                obs_profile=policy.obs_profile, reward_profile=reward_profile,
                parallel_games=parallel_games, device=device, collect=True, pool=pool_ctx,
            )
            games, chunks = [], []
            for m, (game, rows) in played:
                games.append(game)
                chunks.append(rows)
                pfsp.record(m.rival.name, game["result"])
            sim_s = perf_counter() - u_began

            opt_began = perf_counter()
            data = rollout_tensors(chunks, device)
            divergence, ppo_stats = optimize_ppo(
                policy, reference, optimizer, data, int(batch_size), 3, float(kl_coef),
                kl_direction="reverse", return_stats=True,
            )
            del data
            opt_s = perf_counter() - opt_began
            sim_updates.append({
                "update": upd + 1,
                **summarize(games),
                "kl_bc_to_ppo": round(float(divergence), 5),
                "ppo": ppo_stats,
                "rivals": dict(Counter(m.rival.name for m in matches)),
                "simulation_seconds": round(sim_s, 3),
                "optimization_seconds": round(opt_s, 3),
            })

        # 4. Matched-seed A/B evaluation gate (candidate vs. pre-cycle incumbent, prioritizing hardest live matchups)
        eval_report = evaluate_ab_in_sim(
            incumbent_frozen, policy, deck, eval_rivals, output.parent,
            games_per_rival=eval_games_per_rival, seed=int(seed) + 777,
            reward_profile=reward_profile, parallel_games=parallel_games, device=device,
            pool=pool_ctx,
        )
    finally:
        if pool_ctx is not None:
            pool_ctx.__exit__(None, None, None)

    promoted = bool(force_promote or eval_report["passed_gate"])
    cycle_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "deck": deck,
        "source_checkpoint": str(checkpoint),
        "output_checkpoint": str(output),
        "bc_checkpoint": str(bc_checkpoint) if bc_checkpoint else None,
        "discovered_decks_used": [r.deck for r in discovered_rivals],
        "live_meta": meta,
        "awr": awr_report,
        "synthetic_updates": sim_updates,
        "ab_eval": eval_report,
        "promoted": promoted,
        "force_promote": bool(force_promote),
        "elapsed_seconds": round(perf_counter() - began, 2),
    }

    if promoted:
        merged_metrics = {
            **(saved_ckpt.get("metrics") or {}),
            "hybrid_loop": {
                "timestamp": cycle_summary["timestamp"],
                "ab_delta_score": eval_report["delta_score"],
                "ab_candidate_score": eval_report["candidate"]["win_score"],
                "ab_incumbent_score": eval_report["incumbent"]["win_score"],
                "live_matches_scanned": meta["matches_scanned"],
                "ppo_updates": len(sim_updates),
            },
        }
        _atomic_promote_checkpoint(
            output, policy, deck, saved_ckpt.get("phase", "ppo"),
            merged_metrics, saved_ckpt.get("trainer"),
        )
        # Save cycle snapshot into league/ for future self-play diversity
        league_dir.mkdir(parents=True, exist_ok=True)
        existing_snaps = sorted(league_dir.glob("*.pt"))
        next_idx = (max((int(p.stem) for p in existing_snaps if p.stem.isdigit()), default=0)) + 1
        save_checkpoint(league_dir / f"{next_idx:05d}.pt", policy, deck, "ppo", merged_metrics)

    # 5. Optional Vision YOLO refresh in lockstep
    if also_vision:
        try:
            from xbow_vision_train import generate_synthetic_dataset, train_vision_yolo
            live_v = len(list((vision_dataset_dir / "live" / "images").glob("*.jpg"))) if (vision_dataset_dir / "live" / "images").is_dir() else 0
            if live_v >= 10:
                generate_synthetic_dataset(vision_dataset_dir, count=200, deck=deck)
                cycle_summary["vision_train"] = train_vision_yolo(
                    vision_dataset_dir, epochs=6, imgsz=480, batch=8,
                )
            else:
                cycle_summary["vision_train"] = {"skipped": True, "live_frames": live_v}
        except Exception as exc:
            cycle_summary["vision_train"] = {"error": str(exc)}

    # Append telemetry to hybrid_loop.json
    journal_path = output.parent / "hybrid_loop.json"
    history = []
    if journal_path.is_file():
        try:
            loaded = json.loads(journal_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                history = loaded[-49:]
        except Exception:
            pass
    history.append(cycle_summary)
    tmp_j = journal_path.with_suffix(".json.tmp")
    tmp_j.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_j.replace(journal_path)
    return cycle_summary


def hybrid_status(checkpoint: Path, matches_dir: Path = Path("runs/live_matches"), deck: str = "xbow_30") -> dict:
    checkpoint = Path(checkpoint)
    journal_path = checkpoint.parent / "hybrid_loop.json"
    history = []
    if journal_path.is_file():
        try:
            loaded = json.loads(journal_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                history = loaded
        except Exception:
            pass
    return {
        "checkpoint": str(checkpoint),
        "exists": checkpoint.is_file(),
        "cycles_completed": len(history),
        "promotions": sum(1 for c in history if c.get("promoted")),
        "latest_cycle": history[-1] if history else None,
        "live_meta": analyze_live_meta(matches_dir, deck=deck),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_meta = sub.add_parser("meta", help="Analisar o meta real coletado em runs/live_matches/")
    p_meta.add_argument("--deck", choices=DECKS, default="xbow_30")
    p_meta.add_argument("--matches-dir", type=Path, default=Path("runs/live_matches"))

    def _add_cycle_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--deck", choices=DECKS, default="xbow_30")
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--bc-checkpoint", type=Path, default=None)
        p.add_argument("--output", type=Path, default=None)
        p.add_argument("--matches-dir", type=Path, default=Path("runs/live_matches"))
        p.add_argument("--skip-awr", action="store_true")
        p.add_argument("--ppo-updates", type=int, default=3)
        p.add_argument("--rollout-games", type=int, default=8)
        p.add_argument("--batch-size", type=int, default=256)
        p.add_argument("--lr", type=float, default=6e-5)
        p.add_argument("--kl-coef", type=float, default=0.20)
        p.add_argument("--eval-games-per-rival", type=int, default=2)
        p.add_argument("--parallel-games", type=int, default=4)
        p.add_argument("--threads", type=int, default=6)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--force-promote", action="store_true")
        p.add_argument("--also-vision", action="store_true")

    p_reinf = sub.add_parser("reinforce", help="Rodar 1 ciclo hibrido completo (Live AWR -> PPO sintético -> Gate A/B)")
    _add_cycle_args(p_reinf)

    p_daemon = sub.add_parser("daemon", help="Monitorar runs/live_matches/ e rodar ciclos hibridos a cada N partidas")
    _add_cycle_args(p_daemon)
    p_daemon.add_argument("--every-matches", type=int, default=5)
    p_daemon.add_argument("--poll-seconds", type=float, default=30.0)

    p_stat = sub.add_parser("status", help="Exibir historico dos ciclos hibridos e meta ao vivo")
    p_stat.add_argument("--deck", choices=DECKS, default="xbow_30")
    p_stat.add_argument("--checkpoint", type=Path, required=True)
    p_stat.add_argument("--matches-dir", type=Path, default=Path("runs/live_matches"))

    args = parser.parse_args()
    if args.cmd == "meta":
        print(json.dumps(analyze_live_meta(args.matches_dir, deck=args.deck), indent=2, ensure_ascii=False))
    elif args.cmd == "status":
        print(json.dumps(hybrid_status(args.checkpoint, args.matches_dir, args.deck), indent=2, ensure_ascii=False))
    elif args.cmd == "reinforce":
        res = run_hybrid_cycle(
            deck=args.deck,
            checkpoint=args.checkpoint,
            bc_checkpoint=args.bc_checkpoint,
            output=args.output,
            matches_dir=args.matches_dir,
            skip_awr=args.skip_awr,
            ppo_updates=args.ppo_updates,
            rollout_games=args.rollout_games,
            batch_size=args.batch_size,
            lr=args.lr,
            kl_coef=args.kl_coef,
            eval_games_per_rival=args.eval_games_per_rival,
            parallel_games=args.parallel_games,
            threads=args.threads,
            seed=args.seed,
            force_promote=args.force_promote,
            also_vision=args.also_vision,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
    elif args.cmd == "daemon":
        last_count = len(list(args.matches_dir.glob("*.jsonl"))) if args.matches_dir.is_dir() else 0
        cycle_idx = 0
        print(
            f"[HYBRID DAEMON] Monitorando {args.matches_dir} (atual={last_count} jogos, "
            f"ciclo a cada +{args.every_matches} jogos)…",
            flush=True,
        )
        try:
            while True:
                cur_count = len(list(args.matches_dir.glob("*.jsonl"))) if args.matches_dir.is_dir() else 0
                if cur_count - last_count >= max(1, int(args.every_matches)):
                    cycle_idx += 1
                    print(f"[HYBRID DAEMON] Disparando ciclo #{cycle_idx} ({cur_count} partidas gravadas)…", flush=True)
                    target_ckpt = args.output if (args.output and args.output.is_file()) else args.checkpoint
                    res = run_hybrid_cycle(
                        deck=args.deck,
                        checkpoint=target_ckpt,
                        bc_checkpoint=args.bc_checkpoint,
                        output=args.output or args.checkpoint,
                        matches_dir=args.matches_dir,
                        skip_awr=args.skip_awr,
                        ppo_updates=args.ppo_updates,
                        rollout_games=args.rollout_games,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        kl_coef=args.kl_coef,
                        eval_games_per_rival=args.eval_games_per_rival,
                        parallel_games=args.parallel_games,
                        threads=args.threads,
                        seed=args.seed + cycle_idx * 100,
                        force_promote=args.force_promote,
                        also_vision=args.also_vision,
                    )
                    last_count = cur_count
                    print(
                        f"[HYBRID DAEMON] Ciclo #{cycle_idx} concluido: promovido={res['promoted']} · "
                        f"{res['ab_eval']['reason']} ({res['elapsed_seconds']}s)",
                        flush=True,
                    )
                sleep(max(2.0, float(args.poll_seconds)))
        except KeyboardInterrupt:
            print("[HYBRID DAEMON] Encerrado pelo usuario.", flush=True)


if __name__ == "__main__":
    main()
