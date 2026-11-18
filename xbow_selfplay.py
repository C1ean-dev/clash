"""Train from a BC checkpoint against frozen rivals and evaluate held-out games."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import torch
from royalegym.opponents import PatientOpponent, PushOpponent
from royaleimitate.replays import CARD_NAMES

from xbow_arena import ArenaCapture, arena_enabled
from xbow_environment import battle_env
from xbow_replays import SEATS
from xbow_training import frozen_policy, load_checkpoint, optimize_ppo, save_checkpoint
from xbow_metrics import check_training_stop


@dataclass(frozen=True)
class Rival:
    name: str
    deck: str
    policy: object = None


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


def play_game(env, policy, rival, seat, seed, device, collect=False, capture=None):
    rng = np.random.default_rng([seed, 1 - seat])
    generators = [None, None] if collect else [
        torch.Generator(device=device).manual_seed(2 * seed + team) for team in (0, 1)
    ]
    obs, _ = env.reset(seed=seed)
    own, other = SEATS[seat], SEATS[1 - seat]
    spatial, vector, mask, actions, old_logp, rewards, values = [], [], [], [], [], [], []
    reward_total = elixir_trade = lock_count = plays = steps = 0
    while env.agents:
        if capture:
            check_training_stop()
        action, logp, value = policy.decide(obs[own], device, generator=generators[seat])
        if rival.policy is None:
            response = scripted_move(env, 1 - seat, obs[other], rng, rival.name)
        else:
            response, _, _ = rival.policy.decide(
                obs[other], device, generator=generators[1 - seat],
            )
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


def traced_game(output, stage, env, policy, rival, seat, seed, device, *, collect=False, **meta):
    if not arena_enabled():
        return play_game(env, policy, rival, seat, seed, device, collect)
    with ArenaCapture(output, stage, env, rival=rival.name, seat=seat, **meta) as capture:
        return play_game(env, policy, rival, seat, seed, device, collect, capture)


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
    envs = {}

    def get_env(rival, seat):
        key = (args.deck, rival.deck) if seat == 0 else (rival.deck, args.deck)
        if key not in envs:
            envs[key] = battle_env(*key)
        return envs[key]

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
                            resumed=args.resume_ppo)
        for update in range(start_update, final_update):
            games, chunks = [], []
            rival_counts = Counter()
            for i in range(args.rollout_games):
                check_training_stop()
                rival = league[(update + i) % len(league)]
                rival_counts[rival.name] += 1
                seat = (update + i) % 2
                game, rows = traced_game(
                    output, "ppo",
                    get_env(rival, seat), policy, rival, seat,
                    int(rng.integers(2**31)), device, collect=True, update=update + 1, number=i + 1,
                )
                games.append(game)
                chunks.append(rows)
                if report_progress:
                    report_progress("ppo_game", update=update + 1, total=final_update,
                                    done=i + 1, games=args.rollout_games, rival=rival.name)
            data = [torch.as_tensor(np.concatenate(column), device=device)
                    for column in zip(*chunks, strict=True)]
            divergence = optimize_ppo(
                policy, reference, optimizer, data, args.batch_size, 3, args.kl_coef,
            )
            metrics = {**summarize(games), "kl_bc_to_ppo": divergence,
                       "rivals": dict(rival_counts), "update": update + 1}
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
                report_progress("ppo_update", done=update + 1, total=final_update, metrics=metrics)
            if len(league) > len(rivals) + 6:
                del league[len(rivals)]
        evaluations = {}
        if args.evaluate:
            if report_progress:
                total = len(evaluation_rivals) * (2 if ppo_metrics or args.evaluate_checkpoint else 1)
                report_progress("evaluation_start", done=0, total=total)
            for rival in evaluation_rivals:
                for label, tested in (("bc", reference), ("ppo", policy)):
                    if label == "ppo" and not (ppo_metrics or args.evaluate_checkpoint):
                        break
                    games = []
                    for i in range(args.evaluate):
                        check_training_stop()
                        seed = 1_000_000 + args.seed + i // 2
                        seat = i % 2
                        games.append(traced_game(
                            output, "evaluation", get_env(rival, seat), tested, rival, seat, seed,
                            device, model=label, number=i + 1,
                        ))
                    name = f"{label}_vs_{rival.name}"
                    evaluations[name] = summarize(games)
                    print(f"{name}: {evaluations[name]}", flush=True)
                    if report_progress:
                        report_progress("evaluation_pair", done=len(evaluations), total=total)
        output.mkdir(parents=True, exist_ok=True)
        report = "evaluation.json" if args.evaluate and not ppo_metrics else "selfplay.json"
        (output / report).write_text(
            json.dumps({"updates": previous_updates + ppo_metrics,
                        "evaluation": evaluations}, indent=2),
            encoding="utf-8",
        )
    finally:
        for env in envs.values():
            env.close()
