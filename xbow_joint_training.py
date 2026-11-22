"""Learn from both sides of each match against a rotating roster of decks."""

from collections import Counter
from contextlib import ExitStack, closing
import hashlib
from itertools import batched, count
import json
from shutil import copyfile
from time import perf_counter

import numpy as np
import torch
from royalegym.protocol import MatchSetup

from xbow_environment import JOINT_REWARD_PROFILES, RIVAL_DECKS, all_rival_decks, battle_env, deck_card_ids
from xbow_metrics import check_training_stop, defer_training_stop, training_journal
from xbow_gpu_rollout import GpuBatchedRolloutPool, create_rollout_pool, directml_available
from xbow_rollout_pool import RolloutWorkerPool, default_rollout_workers
from xbow_selfplay import (
    DEFAULT_PARALLEL_GAMES, Match, PFSPPool, Rival, play_wave, restore_training_state,
    resume_history, rollout_tensors, summarize, training_state,
)
from xbow_training import (
    XBowPolicy, checkpoint_data, load_checkpoint, optimize_ppo, policy_from_checkpoint,
    save_checkpoint, scheduled_kl_coef, stratified_ppo_subsample,
)


ANCHOR_RIVALS = (
    ("scripted", "icebow"),
    ("rg_patient", "royal_giant"),
    ("hog_push", "hog_26"),
)


def match_setup(fixed_deck, rival_deck, xbow_seat):
    decks = [fixed_deck, rival_deck] if xbow_seat == 0 else [rival_deck, fixed_deck]
    return MatchSetup(decks=decks)


def _warm_start_free(rival_bc_path, target_policy, device, layout):
    saved = torch.load(rival_bc_path, map_location=device, weights_only=True)
    if saved.get("phase") != "bc":
        raise ValueError("O warm-start do rival requer um checkpoint da fase BC.")
    if tuple(saved.get("spec", ())) != target_policy.spec or saved.get("architecture", 1) != target_policy.architecture:
        raise ValueError("Checkpoint BC do rival incompativel com a arquitetura atual.")
    loaded = policy_from_checkpoint(saved, saved["deck"], device, memory_format=layout)
    target_policy.load_state_dict(loaded.state_dict())
    return target_policy


def train_pair(args, device, bc_path, output, report, *, reward_profile="potential", evaluation_games=None):
    continuous = getattr(args, "continuous", False)
    kl_direction = getattr(args, "kl_direction", "reverse")
    kl_decay = not getattr(args, "no_kl_decay", False)
    if evaluation_games is None:
        evaluation_games = args.evaluate
    layout = torch.channels_last if device == "cpu" else torch.preserve_format
    reference, bc = load_checkpoint(bc_path, args.deck, device, memory_format=layout)
    xbow, saved = load_checkpoint(output / "ppo.pt", args.deck, device, memory_format=layout)
    trainer = saved.get("trainer")
    with bc_path.open("rb") as source:
        reference_hash = hashlib.file_digest(source, "sha256").hexdigest()
    if bc["phase"] != "bc" or reference.spec != xbow.spec:
        raise ValueError("A X-Bow precisa de um checkpoint BC compatível.")
    if saved["phase"] != "ppo" or not trainer or trainer["reference_hash"] != reference_hash:
        raise ValueError("A X-Bow precisa de um checkpoint PPO do mesmo treino de imitação.")
    reference.requires_grad_(False)
    optimizer = torch.optim.AdamW(xbow.parameters(), lr=1e-4)
    rng = np.random.default_rng(args.seed)
    previous = None if continuous else resume_history(output, saved["metrics"])
    start = saved["metrics"]["update"]
    total = None if continuous else start + args.ppo_updates
    updates = count(start + 1) if continuous else range(start + 1, total + 1)
    parallel_games = (args.parallel_games or DEFAULT_PARALLEL_GAMES) if continuous else (args.parallel_games or 1)
    requested_workers = getattr(args, "rollout_workers", None)
    rollout_workers = (
        default_rollout_workers(parallel_games, device)
        if requested_workers is None else max(1, int(requested_workers))
    )
    if device != "cpu" or parallel_games <= 1:
        rollout_workers = 1
    obs_profile = xbow.obs_profile
    session_games = 0
    progress = {
        "joint": True, "continuous": continuous, "session_start": start,
        "parallel_games": parallel_games, "rollout_workers": rollout_workers,
    }
    use_pool = (
        rollout_workers > 1
        or (requested_workers is None and parallel_games > 1 and device == "cpu" and directml_available())
    )
    with ExitStack() as stack, defer_training_stop(continuous):
        worker_pool = (
            stack.enter_context(create_rollout_pool(
                rollout_workers, obs_profile, reward_profile, output,
                parallel_games=parallel_games,
            ))
            if use_pool else None
        )
        env_count = 1 if worker_pool is not None else parallel_games
        envs = [stack.enter_context(closing(battle_env(
            args.deck, reward_profile=reward_profile, obs_profile=obs_profile,
        ))) for _ in range(env_count)]
        env = envs[0]
        card_ids = {card.name: card.card_id for card in env.engine.cards()}
        active_rivals = all_rival_decks(max_discovered=10)
        rival_decks = {name: deck_card_ids(cards, card_ids) for name, cards in active_rivals.items()}
        anchor_decks = {deck: deck_card_ids(deck, card_ids) for _, deck in ANCHOR_RIVALS}
        rotation = tuple(rival_decks)
        fixed = env.state_mutator.decks[0]
        free_saved = trainer.get("free_model")
        if free_saved is None:
            obs, _ = env.reset(seed=args.seed)
            free = XBowPolicy.from_observation(
                obs["red"], env=env, architecture=xbow.architecture,
            ).to(device=device, memory_format=layout)
            rival_bc = getattr(args, "rival_bc_checkpoint", None)
            if rival_bc is not None:
                _warm_start_free(rival_bc, free, device, layout)
            free_update = 0
            free_matches = 0
        else:
            free = policy_from_checkpoint(free_saved, "free", device, memory_format=layout)
            free_update = free_saved["metrics"]["update"]
            free_matches = free_saved["metrics"].get("matches_played", free_update)
        free_optimizer = torch.optim.AdamW(free.parameters(), lr=1e-4)
        if free_saved is not None:
            free_optimizer.load_state_dict(free_saved["trainer"]["optimizer"])
        restore_training_state(trainer, optimizer, rng, device)
        pfsp = PFSPPool(trainer.get("pfsp_wr"))

        def matchup(deck, seed, seat, number):
            return Match(Rival(deck, deck, free), seat, seed, number,
                         match_setup(fixed, rival_decks[deck], seat))

        def anchor_matchup(anchor_idx, seed, seat, number):
            name, deck = ANCHOR_RIVALS[anchor_idx % len(ANCHOR_RIVALS)]
            return Match(Rival(name, deck, None), seat, seed, number,
                         match_setup(fixed, anchor_decks[deck], seat))

        report("ppo_start", done=start, total=total, resumed=True, completed_games=session_games,
               reward_profile=reward_profile, **progress)
        if continuous:
            print(f"Treino contínuo retomado do checkpoint {start} (workers={rollout_workers}).", flush=True)
        for update in updates:
            check_training_stop(at_boundary=True)
            began = perf_counter()
            include_anchor = parallel_games >= 4 and (update % 2 == 0)
            free_slots = parallel_games - (1 if include_anchor else 0)
            if parallel_games == 1:
                chosen_decks = [rotation[free_matches % len(rotation)]]
            else:
                deck_indices = pfsp.sample_indices(list(rotation), free_slots, rng)
                chosen_decks = [rotation[idx] for idx in deck_indices]
            matches = [
                matchup(deck, int(rng.integers(2**31)), (free_matches + offset) % 2,
                        session_games + offset + 1)
                for offset, deck in enumerate(chosen_decks)
            ]
            if include_anchor:
                matches.append(anchor_matchup(
                    update // 2, int(rng.integers(2**31)), update % 2,
                    session_games + parallel_games,
                ))
            rivals = dict(Counter(match.rival.name for match in matches))
            sole_rival = next(iter(rivals)) if len(rivals) == 1 else None
            report("ppo_match", update=update, done=update - 1, total=total, rival=sole_rival,
                   completed_games=session_games, **progress)
            free_matches_list = [m for m in matches if m.rival.policy is not None]
            anchor_matches_list = [m for m in matches if m.rival.policy is None]

            xbow_games, free_games = [], []
            xbow_trajectories, free_trajectories = [], []

            if worker_pool is not None:
                worker_pool.publish_model("free", free, force=True)
                rival_key_map = {id(free): "free"}
                if free_matches_list:
                    for match, ((xbow_game, xbow_rows), (free_game, free_rows)) in worker_pool.run_matches(
                        args.deck, "ppo", xbow, free_matches_list, collect=True, collect_both=True,
                        policy_key="xbow", force_sync_policy=True, rival_keys=rival_key_map, update=update,
                    ):
                        xbow_games.append(xbow_game)
                        xbow_trajectories.append(xbow_rows)
                        free_games.append(free_game)
                        free_trajectories.append(free_rows)
                        pfsp.record(match.rival.name, xbow_game["result"])
                        done_now = len(xbow_games)
                        if done_now == 1 or done_now % 8 == 0 or done_now == parallel_games:
                            report("ppo_game", update=update, total=total, done=done_now, games=parallel_games,
                                   rival=match.rival.name, completed_games=session_games + done_now, **progress)
                if anchor_matches_list:
                    for match, (xbow_game, xbow_rows) in worker_pool.run_matches(
                        args.deck, "ppo", xbow, anchor_matches_list, collect=True, collect_both=False,
                        policy_key="xbow", force_sync_policy=False, update=update,
                    ):
                        xbow_games.append(xbow_game)
                        xbow_trajectories.append(xbow_rows)
                        done_now = len(xbow_games)
                        report("ppo_game", update=update, total=total, done=done_now, games=parallel_games,
                               rival=match.rival.name, completed_games=session_games + done_now, **progress)
            else:
                if free_matches_list:
                    runs = play_wave(
                        output, "ppo", xbow, free_matches_list, envs[:len(free_matches_list)],
                        device, collect=True, collect_both=True, update=update,
                    )
                    for match, ((xbow_game, xbow_rows), (free_game, free_rows)) in zip(free_matches_list, runs, strict=True):
                        xbow_games.append(xbow_game)
                        xbow_trajectories.append(xbow_rows)
                        free_games.append(free_game)
                        free_trajectories.append(free_rows)
                        pfsp.record(match.rival.name, xbow_game["result"])
                    del runs
                if anchor_matches_list:
                    anchor_runs = play_wave(
                        output, "ppo", xbow, anchor_matches_list, envs[:len(anchor_matches_list)],
                        device, collect=True, collect_both=False, update=update,
                    )
                    for xbow_game, xbow_rows in anchor_runs:
                        xbow_games.append(xbow_game)
                        xbow_trajectories.append(xbow_rows)
                    del anchor_runs

            simulation = perf_counter() - began
            sim_gph = len(xbow_games) / max(1e-6, simulation) * 3600.0
            print(
                f"[Lote #{update}] {len(xbow_games)}/{parallel_games} jogos simulados em {simulation:.1f}s "
                f"({sim_gph:,.0f} j/h) — otimizando X-Bow e IA adversaria...",
                flush=True,
            )
            report("ppo_game", update=update, total=total, done=parallel_games, games=parallel_games,
                   rival=sole_rival, completed_games=session_games + len(xbow_games), **progress)
            began = perf_counter()
            ref_runner = None
            if isinstance(worker_pool, GpuBatchedRolloutPool):
                worker_pool.publish_model("bc_ref", reference, force=False)
                ref_runner = worker_pool._runners.get("bc_ref")

            data = rollout_tensors(xbow_trajectories, device)
            xbow_trajectories.clear()
            large_wave = device == "cpu" and len(data[3]) > 12_000
            if large_wave:
                data = stratified_ppo_subsample(data, max_samples=8192, rng=rng)
            ppo_epochs = 1 if large_wave else 3
            effective_kl = scheduled_kl_coef(args.kl_coef, update - 1) if kl_decay else args.kl_coef
            kl, xbow_ppo_stats = optimize_ppo(
                xbow, reference, optimizer, data, args.batch_size, ppo_epochs, effective_kl,
                kl_direction=kl_direction, return_stats=True, reference_runner=ref_runner,
            )
            del data
            free_ppo_stats = None
            if free_trajectories:
                data = rollout_tensors(free_trajectories, device)
                free_trajectories.clear()
                large_free = device == "cpu" and len(data[3]) > 12_000
                if large_free:
                    data = stratified_ppo_subsample(data, max_samples=8192, rng=rng)
                _, free_ppo_stats = optimize_ppo(
                    free, None, free_optimizer, data, args.batch_size, 1 if large_free else 3, 0,
                    ent_coef=0.012, wait_ent_coef=0.025, return_stats=True,
                )
                del data
            opt_seconds = perf_counter() - began
            free_update += 1
            free_matches += len(free_games)
            free_metrics = {
                **summarize(free_games), "update": free_update, "matches_played": free_matches,
                **({"ppo": free_ppo_stats} if free_ppo_stats else {}),
            }
            metrics = {
                **summarize(xbow_games), "update": update, "free": free_metrics,
                "reward_profile": reward_profile, "ppo": xbow_ppo_stats,
                "rollout_workers": rollout_workers,
                "match_results": [game["result"] for game in xbow_games],
                "kl_bc_to_ppo": kl, "rivals": rivals,
                "simulation_seconds": round(simulation, 3),
                "optimization_seconds": round(opt_seconds, 3),
            }
            if sole_rival and sole_rival in RIVAL_DECKS:
                metrics["free_deck"] = list(RIVAL_DECKS[sole_rival])
            # The latest checkpoint publishes both updates in the same atomic replace.
            free_model = checkpoint_data(free, "free", "ppo", free_metrics,
                                         {"optimizer": free_optimizer.state_dict()})
            state = training_state(
                reference_hash, optimizer, rng, device,
                free_model=free_model, reward_profile=reward_profile,
                pfsp_wr=dict(pfsp.ema_wr),
            )
            save_checkpoint(output / "ppo.pt", xbow, args.deck, "ppo", metrics, state)
            session_games += parallel_games
            total_gph = len(xbow_games) / max(1e-6, simulation + opt_seconds) * 3600.0
            wr_pct = (metrics["wins"] + 0.5 * metrics["draws"]) / max(1, metrics["games"]) * 100.0
            if not continuous:
                save_checkpoint(output / "league" / f"{update:05d}.pt", xbow, args.deck, "ppo", metrics)
                previous.append(metrics)
                print(f"Partida {update - start}/{args.ppo_updates}: {metrics}", flush=True)
            else:
                print(
                    f"[Co-evolucao #{update} | {session_games:,} jogos na sessao] "
                    f"WR={wr_pct:5.1f}% ({metrics['wins']}V-{metrics['losses']}D-{metrics['draws']}E) | "
                    f"sim={simulation:.1f}s opt={opt_seconds:.1f}s | {total_gph:,.0f} jogos/h | KL={kl:.4f}",
                    flush=True,
                )
            report("ppo_update", done=update, total=total, metrics=metrics,
                   completed_games=session_games, **progress)

        evaluation = {}
        if evaluation_games:
            evaluation_total = evaluation_games * len(rotation)
            report("evaluation_start", done=0, total=evaluation_total, joint=True)
            done_eval = 0
            for index, deck in enumerate(rotation):
                eval_matches = [
                    matchup(deck, 1_000_000 + args.seed + i // 2, (index + i) % 2,
                            index * evaluation_games + i + 1)
                    for i in range(evaluation_games)
                ]
                games = []
                if worker_pool is not None and len(eval_matches) > 1:
                    worker_pool.publish_model("free", free, force=True)
                    for _, game in worker_pool.run_matches(
                        args.deck, "evaluation", xbow, eval_matches, collect=False,
                        policy_key="xbow", force_sync_policy=True,
                        rival_keys={id(free): "free"}, model="ppo", joint=True,
                    ):
                        games.append(game)
                        done_eval += 1
                        report("evaluation_pair", done=done_eval, total=evaluation_total, joint=True)
                else:
                    for match in eval_matches:
                        games.extend(play_wave(output, "evaluation", xbow, [match], [env], device,
                                               model="ppo", joint=True))
                        done_eval += 1
                        report("evaluation_pair", done=done_eval, total=evaluation_total, joint=True)
                evaluation[f"ppo_vs_{deck}"] = summarize(games)
        (output / "selfplay.json").write_text(
            json.dumps({"updates": previous, "evaluation": evaluation}, indent=2), encoding="utf-8",
        )


def load_joint_models(path, deck, device):
    layout = torch.channels_last if device == "cpu" else torch.preserve_format
    xbow, saved = load_checkpoint(path, deck, device, memory_format=layout)
    rival = (saved.get("trainer") or {}).get("free_model")
    if saved["phase"] != "ppo" or not rival or rival["phase"] != "ppo":
        raise ValueError("A comparação requer um checkpoint PPO com os dois modelos do treino conjunto.")
    return xbow, policy_from_checkpoint(rival, "free", device, memory_format=layout)


def evaluate_against_initial(args, device, output, models, initial_models, seeds):
    parallel = min(args.evaluate, args.parallel_games or DEFAULT_PARALLEL_GAMES)
    obs_profile = models[0].obs_profile
    evaluation = {}
    with ExitStack() as stack:
        envs = [stack.enter_context(closing(battle_env(
            args.deck, reward_profile="potential", obs_profile=obs_profile,
        ))) for _ in range(parallel)]
        card_ids = {card.name: card.card_id for card in envs[0].engine.cards()}
        fixed_deck = deck_card_ids(args.deck, card_ids)
        for role, (policy, opponent) in enumerate(zip(models, reversed(initial_models), strict=True)):
            name = ("xbow_30", "rival_live")[role]
            evaluation[name] = {}
            for deck in RIVAL_DECKS:
                rival_deck = deck_card_ids(deck, card_ids)
                opponent_deck = deck if role == 0 else args.deck
                rival = Rival(opponent_deck, opponent_deck, opponent)
                matches = (
                    Match(rival, xbow_seat ^ role, seed, index * 2 + xbow_seat + 1,
                          match_setup(fixed_deck, rival_deck, xbow_seat))
                    for index, seed in enumerate(seeds) for xbow_seat in (0, 1)
                )
                games = []
                for wave in batched(matches, parallel):
                    games.extend(play_wave(output, "evaluation", policy, wave, envs[:len(wave)],
                                           device, capture_arena=False))
                evaluation[name][deck] = summarize(games)
                print(f"Avaliação {name}, {deck}: {evaluation[name][deck]['wins']}/{len(games)} vitórias",
                      flush=True)
    return evaluation


def compare_rewards(args, device, bc_path, source):
    root = args.compare_rewards
    root.mkdir(parents=True)
    initial = root / "initial.pt"
    reference = root / "bc.pt"
    copyfile(source / "ppo.pt", initial)
    copyfile(bc_path, reference)
    initial_models = load_joint_models(initial, args.deck, device)
    # Training seeds are below 2**31. Both evaluation arms share seeds and seats.
    seeds = [2**31 + args.seed + index for index in range(args.evaluate // 2)]
    result = {
        "initial_checkpoint": str(initial.resolve()),
        "training_games_per_profile": args.ppo_updates,
        "evaluation_games_per_deck_and_model": args.evaluate,
        "evaluation_seeds": seeds, "evaluation_reward_profile": "potential", "profiles": {},
    }
    for profile in JOINT_REWARD_PROFILES:
        # The dashboard skips checkpoints inside hidden directories.
        output = root / f".{profile}" / args.deck
        with training_journal(output, args.deck) as report:
            copyfile(initial, output / "ppo.pt")
            print(f"Comparação de recompensas: {profile}", flush=True)
            train_pair(args, device, reference, output, report,
                       reward_profile=profile, evaluation_games=0)
            models = load_joint_models(output / "ppo.pt", args.deck, device)
            result["profiles"][profile] = {
                "checkpoint": str((output / "ppo.pt").resolve()),
                "versus_initial": evaluate_against_initial(args, device, output, models,
                                                           initial_models, seeds),
            }
            del models
    report_path = root / "reward_comparison.json"
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(report_path)
    print(f"Resultado A/B: {report_path.resolve()}", flush=True)
