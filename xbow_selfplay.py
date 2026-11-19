"""Train from a BC checkpoint against frozen rivals and evaluate held-out games."""

from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
from itertools import batched
import json

import numpy as np
import torch
from royalegym.opponents import PatientOpponent, PushOpponent
from royaleimitate.replays import CARD_NAMES

from xbow_arena import ArenaCapture, KEEP_CLIPS, arena_enabled
from xbow_environment import battle_env
from xbow_replays import SEATS
from xbow_training import frozen_policy, load_checkpoint, optimize_ppo, save_checkpoint
from xbow_metrics import check_training_stop


@dataclass(frozen=True)
class Rival:
    name: str
    deck: str
    policy: object = None


DEFAULT_PARALLEL_GAMES = 4


@dataclass(frozen=True)
class Match:
    rival: Rival
    seat: int
    seed: int
    number: int


TACTICAL_RIVALS = {
    "hog_push": PushOpponent(),
    "rg_patient": PatientOpponent(),
}


def scripted_move(env, seat, obs, rng, kind):
    mask = obs["action_mask"]
    if kind in TACTICAL_RIVALS:
        return TACTICAL_RIVALS[kind].act(obs, mask, rng)
    if kind == "random":
        return int(rng.choice(np.flatnonzero(mask))) if rng.random() < 0.35 else 0

    state = env.battle_state
    player = state.players[seat]
    if player.elixir_milli < 6500:
        return 0
    names = [card.name for card in env.engine.cards()]
    own, foe = state.players[seat], state.players[1 - seat]
    aim_x = 3 if foe.tower_hp[1] <= foe.tower_hp[2] else 14
    own_x = 3 if own.tower_hp[1] <= own.tower_hp[2] else 14
    priority = [
        ("tesla", 8, 11, 7500), ("x-bow", aim_x, 13, 8250),
        ("knight", own_x, 10, 8500), ("ice-wizard", own_x, 6, 8750),
        ("archers", own_x, 6, 8750), ("rocket", aim_x, 25, 9500),
        ("fireball", aim_x, 25, 9500), ("skeletons", own_x, 3, 9250),
        ("electro-spirit", aim_x, 12, 9250), ("the-log", aim_x, 26, 9750),
        ("tornado", aim_x, 24, 9750),
    ]
    parser = env.action_parser
    slots = {names[card]: slot for slot, card in enumerate(player.hand)
             if 0 <= card < len(names)}
    for key, x, y, elixir in priority:
        slot = slots.get(CARD_NAMES[key])
        if slot is None or player.elixir_milli < elixir:
            continue
        action = parser.encode(slot, x, y)
        if mask[action]:
            return action
    return 0


def advantage_targets(rewards, values, bootstrap, gamma=0.997, gae_lambda=0.95):
    advantages = np.empty(len(rewards), dtype=np.float32)
    running = 0.0
    next_value = bootstrap
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value - values[step]
        running = delta + gamma * gae_lambda * running
        advantages[step] = running
        next_value = values[step]
    return advantages, advantages + np.asarray(values, dtype=np.float32)


def game_steps(env, policy, seat, seed, device, collect=False, capture=None):
    """Yield the observations until the wave supplies both decisions for the next step."""
    generators = [None, None] if collect else [
        torch.Generator(device=device).manual_seed(2 * seed + team) for team in (0, 1)
    ]
    obs, _ = env.reset(seed=seed)
    own, other = SEATS[seat], SEATS[1 - seat]
    spatial, vector, mask, actions, old_logp, rewards, values = [], [], [], [], [], [], []
    reward_total = elixir_trade = lock_count = plays = steps = 0
    while env.agents:
        (action, logp, value), response = yield obs, generators
        if capture:
            actions_by_side = (action, response) if seat == 0 else (response, action)
            capture.observe(env.battle_state, actions_by_side)
        if collect:
            spatial.append(obs[own]["spatial"].copy())
            vector.append(obs[own]["vector"].copy())
            mask.append(np.array(obs[own]["action_mask"], dtype=bool))
            actions.append(action)
            old_logp.append(logp)
            values.append(value)
        obs, reward, terminated, truncated, _ = env.step({own: action, other: response})
        reward_total += reward[own]
        if collect:
            rewards.append(reward[own])
        terms = env.reward_fn.terms_for(seat)
        elixir_trade += terms.get("ElixirTradeReward", 0.0) / 0.15
        lock_count += terms["XBowLockReward"] / 0.04
        plays += int(action != 0)
        steps += 1

    state = env.battle_state
    result = (
        "draw" if state.winner not in (0, 1)
        else "win" if state.winner == seat else "loss"
    )
    stats = {
        "result": result, "reward": reward_total, "locks": round(lock_count),
        "elixir_trade": elixir_trade, "plays": plays, "steps": steps,
    }
    if capture:
        capture.observe(state, force=True)
        capture.finish({"status": result, **stats})
    if not collect:
        return stats
    bootstrap = 0.0
    if truncated[own] and not terminated[own] and own in obs:
        _, _, bootstrap = policy.decide(obs[own], device)
    advantage, returns = advantage_targets(rewards, values, bootstrap)
    return stats, (
        np.stack(spatial), np.stack(vector), np.stack(mask), np.asarray(actions),
        np.asarray(old_logp, dtype=np.float32), advantage, returns,
    )


@torch.no_grad()
def decide_batch(policy, inputs, device):
    if len(inputs) == 1:
        obs, generator = inputs[0]
        return [policy.decide(obs, device, generator=generator)]
    observations, generators = zip(*inputs, strict=True)
    spatial, vector, mask = [
        torch.as_tensor(np.stack([obs[key] for obs in observations]), device=device)
        for key in ("spatial", "vector", "action_mask")
    ]
    distribution, values = policy(spatial, vector, mask.bool())
    if all(generator is None for generator in generators):
        actions = distribution.sample()
    else:
        actions = torch.cat([
            torch.multinomial(row, 1, generator=generator)
            for row, generator in zip(distribution.probs, generators, strict=True)
        ])
    return list(zip(actions.tolist(), distribution.log_prob(actions).tolist(),
                    values.tolist(), strict=True))


def play_wave(output, stage, policy, matches, environments, device, *, collect=False, **metadata):
    """Advance independent matches together and batch requests for the same policy."""
    results = [None] * len(matches)
    first_captured = max(0, len(matches) - KEEP_CLIPS) if arena_enabled() else len(matches)
    with ExitStack() as stack:
        pending = {}
        for index, (match, env) in enumerate(zip(matches, environments, strict=True)):
            check_training_stop()
            capture = None
            if index >= first_captured:
                capture = stack.enter_context(ArenaCapture(
                    output, stage, env, rival=match.rival.name, seat=match.seat,
                    number=match.number, **metadata,
                ))
            game = game_steps(env, policy, match.seat, match.seed, device, collect, capture)
            rng = np.random.default_rng([match.seed, 1 - match.seat])
            obs, generators = next(game)
            pending[index] = (game, rng, obs, generators)
        while pending:
            check_training_stop()
            groups = defaultdict(list)
            decisions = {index: [None, None] for index in pending}
            for index, (_, rng, obs, generators) in pending.items():
                match = matches[index]
                own, foe = match.seat, 1 - match.seat
                groups[policy].append((index, own, obs[SEATS[own]], generators[own]))
                if match.rival.policy is None:
                    action = scripted_move(environments[index], foe, obs[SEATS[foe]], rng, match.rival.name)
                    decisions[index][foe] = (action, None, None)
                else:
                    groups[match.rival.policy].append((index, foe, obs[SEATS[foe]], generators[foe]))
            for model, requests in groups.items():
                predictions = decide_batch(model, [(obs, gen) for _, _, obs, gen in requests], device)
                for (index, seat, _, _), prediction in zip(requests, predictions, strict=True):
                    decisions[index][seat] = prediction
            for index, (game, rng, _, _) in tuple(pending.items()):
                own = matches[index].seat
                try:
                    obs, generators = game.send((decisions[index][own], decisions[index][1 - own][0]))
                except StopIteration as done:
                    results[index] = done.value
                    del pending[index]
                else:
                    pending[index] = (game, rng, obs, generators)
    return results


def rollout_tensors(trajectories, device):
    """Combine one column at a time so the original rollout arrays can be released."""
    columns = [list(column) for column in zip(*trajectories, strict=True)]
    trajectories.clear()
    data = []
    for column in columns:
        data.append(torch.as_tensor(np.concatenate(column), device=device))
        column.clear()
    return data


def summarize(games):
    results = Counter(row["result"] for row in games)
    return {
        "games": len(games), "wins": results["win"], "losses": results["loss"],
        "draws": results["draw"], "mean_locks": float(np.mean([row["locks"] for row in games])),
        "mean_elixir_trade": float(np.mean([row["elixir_trade"] for row in games])),
        "mean_reward": float(np.mean([row["reward"] for row in games])),
        "play_rate": sum(row["plays"] for row in games) / sum(row["steps"] for row in games),
    }


def run_selfplay(args, device, bc_path, output, report_progress=None):
    policy, saved = load_checkpoint(bc_path, args.deck, device)
    if saved["phase"] != "bc":
        raise ValueError("O checkpoint de referencia precisa vir da fase BC.")
    reference = frozen_policy(policy)
    with bc_path.open("rb") as checkpoint:
        reference_hash = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    start_update = 0
    if args.evaluate_checkpoint or args.resume_ppo:
        policy, ppo_saved = load_checkpoint(
            args.evaluate_checkpoint or output / "ppo.pt", args.deck, device,
        )
        if ppo_saved["phase"] != "ppo" or policy.spec != reference.spec:
            raise ValueError("Checkpoint PPO incompativel com a referencia BC.")
        trainer = ppo_saved.get("trainer")
        if trainer and trainer["reference_hash"] != reference_hash:
            raise ValueError("Este PPO foi treinado com outro checkpoint BC.")
        if args.resume_ppo:
            if trainer is None:
                raise ValueError("O checkpoint PPO antigo nao inclui o estado do treinador.")
            start_update = ppo_saved["metrics"]["update"]
    other_deck = "icebow" if args.deck == "xbow_30" else "xbow_30"
    rivals = [
        Rival("bc", args.deck, reference), Rival("scripted", other_deck),
        Rival("hog_push", "hog_26"), Rival("rg_patient", "royal_giant"),
    ]
    other_path = output.parent / other_deck / "bc.pt"
    if other_path.is_file():
        other_bc, _ = load_checkpoint(other_path, other_deck, device)
        rivals.append(Rival("other_bc", other_deck, frozen_policy(other_bc)))
    evaluation_rivals = rivals + [Rival("random", args.deck)]
    league = list(rivals)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    rng = np.random.default_rng(args.seed)
    if args.resume_ppo:
        optimizer.load_state_dict(trainer["optimizer"])
        rng.bit_generator.state = trainer["numpy_rng"]
        torch.set_rng_state(trainer["torch_rng"].cpu())
        if device == "cuda" and trainer["cuda_rng"]:
            torch.cuda.set_rng_state_all([state.cpu() for state in trainer["cuda_rng"]])
        snapshots = sorted((output / "league").glob("*.pt"))
        snapshots = [path for path in snapshots if int(path.stem) <= start_update]
        for path in snapshots[-6:]:
            previous, _ = load_checkpoint(path, args.deck, device)
            league.append(Rival(f"snapshot_{int(path.stem)}", args.deck, frozen_policy(previous)))
    parallel_games = getattr(args, "parallel_games", None) or DEFAULT_PARALLEL_GAMES
    envs = defaultdict(list)

    def play_matches(stage, tested, matches, *, collect=False, **metadata):
        for wave in batched(matches, parallel_games):
            used = Counter()
            environments = []
            for match in wave:
                decks = (args.deck, match.rival.deck)
                if match.seat == 1:
                    decks = decks[::-1]
                slot = used[decks]
                if slot == len(envs[decks]):
                    envs[decks].append(battle_env(*decks))
                environments.append(envs[decks][slot])
                used[decks] += 1
            results = play_wave(output, stage, tested, wave, environments, device,
                                collect=collect, **metadata)
            yield from zip(wave, results, strict=True)

    ppo_metrics = []
    previous_updates = []
    if args.resume_ppo:
        existing = output / "selfplay.json"
        if existing.is_file():
            history = json.loads(existing.read_text(encoding="utf-8")).get("updates", [])
            previous_updates = [row for row in history if row["update"] <= start_update]
    try:
        final_update = start_update + args.ppo_updates
        if report_progress and args.ppo_updates:
            report_progress("ppo_start", done=start_update, total=final_update,
                            resumed=args.resume_ppo, parallel_games=min(parallel_games, args.rollout_games))
        for update in range(start_update, final_update):
            games, chunks = [], []
            matches = [
                Match(league[(update + i) % len(league)], (update + i) % 2,
                      int(rng.integers(2**31)), i + 1)
                for i in range(args.rollout_games)
            ]
            concurrent = min(parallel_games, len(matches))
            for match, (game, rows) in play_matches("ppo", policy, matches, collect=True, update=update + 1):
                games.append(game)
                chunks.append(rows)
                if report_progress:
                    report_progress("ppo_game", update=update + 1, total=final_update,
                                    done=match.number, games=args.rollout_games, rival=match.rival.name,
                                    parallel_games=concurrent)
            del rows
            data = rollout_tensors(chunks, device)
            divergence = optimize_ppo(
                policy, reference, optimizer, data, args.batch_size, 3, args.kl_coef,
            )
            del data
            metrics = {**summarize(games), "kl_bc_to_ppo": divergence,
                       "rivals": dict(Counter(match.rival.name for match in matches)), "update": update + 1}
            print(f"PPO {update + 1}/{final_update}: {metrics}", flush=True)
            ppo_metrics.append(metrics)
            save_checkpoint(output / "league" / f"{update + 1:05d}.pt", policy, args.deck,
                            "ppo", metrics)
            save_checkpoint(output / "ppo.pt", policy, args.deck, "ppo", metrics, {
                "reference_hash": reference_hash, "optimizer": optimizer.state_dict(),
                "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
            })
            league.append(Rival(f"snapshot_{update + 1}", args.deck, frozen_policy(policy)))
            if report_progress:
                report_progress("ppo_update", done=update + 1, total=final_update, metrics=metrics,
                                parallel_games=concurrent)
            if len(league) > len(rivals) + 6:
                del league[len(rivals)]
        evaluations = {}
        if args.evaluate:
            concurrent = min(parallel_games, args.evaluate)
            if report_progress:
                total = len(evaluation_rivals) * (2 if ppo_metrics or args.evaluate_checkpoint else 1)
                report_progress("evaluation_start", done=0, total=total, parallel_games=concurrent)
            for rival in evaluation_rivals:
                for label, tested in (("bc", reference), ("ppo", policy)):
                    if label == "ppo" and not (ppo_metrics or args.evaluate_checkpoint):
                        break
                    matches = [
                        Match(rival, i % 2, 1_000_000 + args.seed + i // 2, i + 1)
                        for i in range(args.evaluate)
                    ]
                    games = [game for _, game in play_matches("evaluation", tested, matches, model=label)]
                    name = f"{label}_vs_{rival.name}"
                    evaluations[name] = summarize(games)
                    print(f"{name}: {evaluations[name]}", flush=True)
                    if report_progress:
                        report_progress("evaluation_pair", done=len(evaluations), total=total,
                                        parallel_games=concurrent)
        output.mkdir(parents=True, exist_ok=True)
        report = "evaluation.json" if args.evaluate and not ppo_metrics else "selfplay.json"
        (output / report).write_text(
            json.dumps({"updates": previous_updates + ppo_metrics,
                        "evaluation": evaluations}, indent=2),
            encoding="utf-8",
        )
    finally:
        for pool in envs.values():
            for env in pool:
                env.close()
