"""Learn from both sides of each match against a rotating roster of decks."""

from contextlib import ExitStack, closing
import hashlib
from itertools import batched
import json
from shutil import copyfile
from time import perf_counter

import numpy as np
import torch
from royalegym.protocol import MatchSetup

from xbow_environment import JOINT_REWARD_PROFILES, RIVAL_DECKS, battle_env, deck_card_ids
from xbow_metrics import check_training_stop, training_journal
from xbow_selfplay import (
    DEFAULT_PARALLEL_GAMES, Match, Rival, play_wave, restore_training_state, resume_history,
    summarize, training_state,
)
from xbow_training import (
    XBowPolicy, checkpoint_data, load_checkpoint, optimize_ppo, policy_from_checkpoint,
    save_checkpoint,
)


def match_setup(fixed_deck, rival_deck, xbow_seat):
    decks = [fixed_deck, rival_deck] if xbow_seat == 0 else [rival_deck, fixed_deck]
    return MatchSetup(decks=decks)


def train_pair(args, device, bc_path, output, report, *, reward_profile="potential", evaluation_games=None):
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
    previous = resume_history(output, saved["metrics"])
    start = saved["metrics"]["update"]
    total = start + args.ppo_updates
    with closing(battle_env(args.deck, reward_profile=reward_profile)) as env:
        card_ids = {card.name: card.card_id for card in env.engine.cards()}
        rival_decks = {name: deck_card_ids(name, card_ids) for name in RIVAL_DECKS}
        rotation = tuple(rival_decks)
        fixed = env.state_mutator.decks[0]
        free_saved = trainer.get("free_model")
        if free_saved is None:
            obs, _ = env.reset(seed=args.seed)
            free = XBowPolicy.from_observation(obs["red"]).to(device=device, memory_format=layout)
            free_update = 0
        else:
            free = policy_from_checkpoint(free_saved, "free", device, memory_format=layout)
            free_update = free_saved["metrics"]["update"]
        free_optimizer = torch.optim.AdamW(free.parameters(), lr=1e-4)
        if free_saved is not None:
            free_optimizer.load_state_dict(free_saved["trainer"]["optimizer"])
        restore_training_state(trainer, optimizer, rng, device)

        def matchup(deck, seed, seat, number):
            return Match(Rival(deck, deck, free), seat, seed, number,
                         match_setup(fixed, rival_decks[deck], seat))

        report("ppo_start", done=start, total=total, resumed=True, parallel_games=1,
               joint=True, reward_profile=reward_profile)
        for update in range(start + 1, total + 1):
            check_training_stop()
            began = perf_counter()
            seed = int(rng.integers(2**31))
            rival_deck = rotation[free_update % len(rotation)]
            match = matchup(rival_deck, seed, free_update % 2, update - start)
            report("ppo_match", update=update, done=update - 1, total=total, rival=rival_deck, joint=True)
            [(xbow_run, free_run)] = play_wave(
                output, "ppo", xbow, [match], [env], device, collect=True,
                collect_both=True, update=update,
            )
            simulation = perf_counter() - began
            report("ppo_game", update=update, total=total, done=1, games=1,
                   rival=rival_deck, parallel_games=1, joint=True)
            began = perf_counter()
            xbow_game, rows = xbow_run
            del xbow_run
            data = [torch.as_tensor(row, device=device) for row in rows]
            del rows
            kl = optimize_ppo(xbow, reference, optimizer, data, args.batch_size, 3, args.kl_coef)
            del data
            free_game, rows = free_run
            del free_run
            data = [torch.as_tensor(row, device=device) for row in rows]
            del rows
            optimize_ppo(free, None, free_optimizer, data, args.batch_size, 3, 0)
            del data
            free_update += 1
            free_metrics = {**summarize([free_game]), "update": free_update}
            metrics = {
                **summarize([xbow_game]), "update": update, "free": free_metrics,
                "reward_profile": reward_profile,
                "free_deck": list(RIVAL_DECKS[rival_deck]),
                "kl_bc_to_ppo": kl, "rivals": {rival_deck: 1},
                "simulation_seconds": round(simulation, 3),
                "optimization_seconds": round(perf_counter() - began, 3),
            }
            # The latest checkpoint publishes both updates in the same atomic replace.
            free_model = checkpoint_data(free, "free", "ppo", free_metrics,
                                         {"optimizer": free_optimizer.state_dict()})
            state = training_state(reference_hash, optimizer, rng, device,
                                   free_model=free_model, reward_profile=reward_profile)
            save_checkpoint(output / "ppo.pt", xbow, args.deck, "ppo", metrics, state)
            save_checkpoint(output / "league" / f"{update:05d}.pt", xbow, args.deck, "ppo", metrics)
            previous.append(metrics)
            report("ppo_update", done=update, total=total, metrics=metrics, parallel_games=1, joint=True)
            print(f"Partida {update - start}/{args.ppo_updates}: {metrics}", flush=True)

        evaluation = {}
        if evaluation_games:
            evaluation_total = evaluation_games * len(rotation)
            report("evaluation_start", done=0, total=evaluation_total, joint=True)
            for index, deck in enumerate(rotation):
                games = []
                for i in range(evaluation_games):
                    number = index * evaluation_games + i + 1
                    match = matchup(deck, 1_000_000 + args.seed + i // 2, (index + i) % 2, number)
                    games.extend(play_wave(output, "evaluation", xbow, [match], [env], device, model="ppo"))
                    report("evaluation_pair", done=number, total=evaluation_total, joint=True)
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
    evaluation = {}
    with ExitStack() as stack:
        envs = [stack.enter_context(closing(battle_env(args.deck, reward_profile="potential")))
                for _ in range(parallel)]
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
