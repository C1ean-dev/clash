"""Train from a BC checkpoint against frozen rivals and evaluate held-out games."""

from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
from itertools import batched
import json
from time import perf_counter

import numpy as np
import torch
from royalegym.opponents import DefendOpponent, PatientOpponent, PushOpponent
from royalegym.protocol import MatchSetup
from royaleimitate.replays import CARD_NAMES

from xbow_arena import ArenaCapture, arena_enabled
from xbow_environment import (
    ELIXIR_TRADE_WEIGHT, PPO_GAMMA, XBOW_LOCK_WEIGHT, XBowLockReward, battle_env, pack_id_planes,
)
from xbow_replays import SEATS
from xbow_gpu_rollout import GpuBatchedRolloutPool, create_rollout_pool, directml_available
from xbow_rollout_pool import RolloutWorkerPool, default_rollout_workers
from xbow_training import (
    frozen_policy, load_checkpoint, optimize_ppo, save_checkpoint, scheduled_kl_coef,
    stratified_ppo_subsample, warmup_value_head,
)
from xbow_metrics import check_training_stop, read_run_journal


@dataclass(frozen=True)
class Rival:
    name: str
    deck: str
    policy: object = None


DEFAULT_PARALLEL_GAMES = 4
CPU_TRAINING_PARALLEL_GAMES = 8


@dataclass(frozen=True)
class Match:
    rival: Rival
    seat: int
    seed: int
    number: int
    setup: MatchSetup | None = None


TACTICAL_RIVALS = {
    "hog_push": PushOpponent(),
    "rg_patient": PatientOpponent(),
    "splashyard_defend": DefendOpponent(),
    "log_bait_push": PushOpponent(),
    "golem_patient": PatientOpponent(),
    "pekka_push": PushOpponent(),
}


def scripted_move(env, seat, obs, rng, kind):
    mask = obs["action_mask"]
    if kind in TACTICAL_RIVALS:
        return TACTICAL_RIVALS[kind].act(obs, mask, rng)
    if isinstance(kind, str) and kind.startswith("dyn_"):
        if kind.startswith("dyn_patient_"):
            return TACTICAL_RIVALS["golem_patient"].act(obs, mask, rng)
        if kind.startswith("dyn_defend_"):
            return TACTICAL_RIVALS["splashyard_defend"].act(obs, mask, rng)
        return TACTICAL_RIVALS["hog_push"].act(obs, mask, rng)
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


def advantage_targets(rewards, values, bootstrap, gamma=PPO_GAMMA, gae_lambda=0.95):
    advantages = np.empty(len(rewards), dtype=np.float32)
    running = 0.0
    next_value = bootstrap
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value - values[step]
        running = delta + gamma * gae_lambda * running
        advantages[step] = running
        next_value = values[step]
    return advantages, advantages + np.asarray(values, dtype=np.float32)


class PFSPPool:
    """Prioritized Fictitious Self-Play matchmaker focusing on lowest win-rate opponents."""

    def __init__(self, initial_wr: dict[str, float] | None = None, *,
                 alpha: float = 0.25, floor: float = 0.15, max_share: float = 0.28):
        self.alpha = float(alpha)
        self.floor = float(floor)
        self.max_share = float(max_share)
        self.ema_wr: dict[str, float] = {k: float(v) for k, v in (initial_wr or {}).items()}

    def record(self, name: str, result: str):
        score = 1.0 if result == "win" else 0.5 if result == "draw" else 0.0
        prev = self.ema_wr.get(name, 0.5)
        self.ema_wr[name] = (1.0 - self.alpha) * prev + self.alpha * score

    def weights(self, names: list[str]) -> np.ndarray:
        n = len(names)
        if n == 0:
            return np.empty(0, dtype=np.float64)
        raw = np.array(
            [(1.0 - self.ema_wr.get(name, 0.5)) ** 2 + self.floor for name in names],
            dtype=np.float64,
        )
        total = raw.sum()
        probs = raw / total if total > 0 else np.full(n, 1.0 / n)
        # Water-filling cap prevents a single hard matchup (e.g. pekka_bs) from starving the rest of the meta.
        if n >= 4 and self.max_share > (1.0 / n) + 1e-6:
            capped = np.zeros(n, dtype=bool)
            for _ in range(n):
                over = (~capped) & (probs > self.max_share)
                if not over.any():
                    break
                probs[over] = self.max_share
                capped |= over
                free = ~capped
                if not free.any():
                    break
                rem = max(0.0, 1.0 - float(probs[capped].sum()))
                free_sum = float(probs[free].sum())
                probs[free] = probs[free] * (rem / free_sum) if free_sum > 0 else rem / int(free.sum())
            probs = probs / probs.sum()
        return probs

    def sample_indices(self, names: list[str], count: int, rng) -> list[int]:
        probs = self.weights(names)
        return [int(idx) for idx in rng.choice(len(names), size=count, replace=True, p=probs)]


class Rollout:
    def __init__(self, collect):
        self.frames = ([], [], []) if collect else None
        self.id_planes = [] if collect else None
        self.actions, self.logp, self.rewards, self.values = [], [], [], []
        self.reward = self.locks = self.plays = self.steps = 0
        self.reward_step_abs = Counter()

    def observe(self, obs, decision, reward, terms, locks):
        action, logp, value = decision
        if self.frames is not None:
            spatial, vector, mask = self.frames
            spatial.append(obs["spatial"].astype(np.float16, copy=False))
            vector.append(obs["vector"].astype(np.float16, copy=False))
            mask.append(np.asarray(obs["action_mask"], dtype=bool))
            ids = pack_id_planes(obs)
            if ids is not None:
                self.id_planes.append(ids)
            self.actions.append(action)
            self.logp.append(logp)
            self.values.append(value)
            self.rewards.append(reward)
        self.reward += reward
        for name, value in terms.items():
            self.reward_step_abs[name] += abs(value)
        self.locks += locks
        self.plays += int(action != 0)
        self.steps += 1

    def stats(self, env, seat):
        state = env.battle_state
        winner = state.winner
        result = "draw" if winner not in (0, 1) else "win" if winner == seat else "loss"
        damage = [sum(player.tower_max_hp) - sum(player.tower_hp) for player in state.players]
        terms = env.reward_terms(seat)
        stats = {
            "result": result, "reward": self.reward, "locks": round(self.locks),
            "plays": self.plays, "steps": self.steps, "reward_terms": terms,
            "reward_step_abs": dict(self.reward_step_abs),
            "tower_damage_dealt": damage[1 - seat], "tower_damage_taken": damage[seat],
            "elixir_leaked": env.obs_builder.memory[seat].leaked_elixir(),
        }
        if "ElixirTradeReward" in terms:
            stats["elixir_trade"] = terms["ElixirTradeReward"] / ELIXIR_TRADE_WEIGHT
        return stats

    def finish(self, bootstrap):
        advantage, returns = advantage_targets(self.rewards, self.values, bootstrap)
        packed_ids = np.stack(self.id_planes) if self.id_planes else None
        return (*map(np.stack, self.frames), np.asarray(self.actions),
                np.asarray(self.logp, dtype=np.float32), packed_ids, advantage, returns)


def game_steps(env, policy, seat, seed, device, collect=False, capture=None, *,
               scripted=False, rival_policy=None, setup=None):
    """Yield the observations until the wave supplies both decisions for the next step."""
    generators = [None, None] if collect else [
        torch.Generator(device=device).manual_seed(2 * seed + team) for team in (0, 1)
    ]
    options = {"setup": setup}
    if scripted:
        options["mask_only"] = (SEATS[1 - seat],)
    obs, _ = env.reset(seed=seed, options=options)
    if capture:
        capture.observe(env.battle_state)
    players = [(seat, policy, Rollout(collect))]
    if rival_policy is not None:
        players.append((1 - seat, rival_policy, Rollout(collect)))
    lock_monitor = None
    if not any(isinstance(term, XBowLockReward) for term, _ in env.reward_fn.terms):
        lock_monitor = XBowLockReward()
        lock_monitor.bind(env.engine)
    while env.agents:
        decisions = yield obs, generators
        actions = tuple(decision[0] for decision in decisions)
        previous_state = env.battle_state
        if capture:
            capture.observe(previous_state, actions)
        before = obs
        obs, reward, terminated, truncated, _ = env.step(dict(zip(SEATS, actions, strict=True)))
        for team, _, rollout in players:
            agent = SEATS[team]
            terms = env.reward_fn.terms_for(team)
            locks = (terms["XBowLockReward"] / XBOW_LOCK_WEIGHT if lock_monitor is None
                     else lock_monitor.new_locks(team, previous_state, env.battle_state))
            rollout.observe(before[agent], decisions[team], reward[agent], terms, locks)

    stats = players[0][2].stats(env, seat)
    if capture:
        capture.close()
    if not collect:
        return stats
    results = []
    for team, model, rollout in players:
        agent = SEATS[team]
        bootstrap = 0.0
        if truncated[agent] and not terminated[agent] and agent in obs:
            _, _, bootstrap = model.decide(obs[agent], device)
        results.append((stats if team == seat else rollout.stats(env, team), rollout.finish(bootstrap)))
    return tuple(results) if rival_policy is not None else results[0]


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
    id_planes = None
    if getattr(policy, "architecture", 2) >= 3 and pack_id_planes(observations[0]) is not None:
        id_planes = torch.as_tensor(
            np.stack([pack_id_planes(obs) for obs in observations]), device=device,
        )
    distribution, values = policy(spatial, vector, mask.bool(), id_planes=id_planes)
    if all(generator is None for generator in generators):
        actions = distribution.sample()
    else:
        actions = torch.cat([
            torch.multinomial(row, 1, generator=generator)
            for row, generator in zip(distribution.probs, generators, strict=True)
        ])
    return list(zip(actions.tolist(), distribution.log_prob(actions).tolist(),
                    values.tolist(), strict=True))


def play_wave(output, stage, policy, matches, environments, device, *,
              collect=False, collect_both=False, capture_arena=True, **metadata):
    """Advance independent matches together and batch requests for the same policy."""
    results = [None] * len(matches)
    show_live = capture_arena and arena_enabled()
    capture_metadata = {**metadata, "joint": collect_both or metadata.get("joint", False)}
    with ExitStack() as stack:
        pending = {}
        for index, (match, env) in enumerate(zip(matches, environments, strict=True)):
            check_training_stop()
            capture = None
            if show_live:
                capture = stack.enter_context(ArenaCapture(
                    output, stage, env, rival=match.rival.name, seat=match.seat,
                    number=match.number, **capture_metadata,
                ))
            game = game_steps(env, policy, match.seat, match.seed, device, collect, capture,
                              scripted=match.rival.policy is None, setup=match.setup,
                              rival_policy=match.rival.policy if collect_both else None)
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
                try:
                    obs, generators = game.send(decisions[index])
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
        if column[0] is None:
            data.append(None)
            column.clear()
            continue
        packed = column[0] if len(column) == 1 else np.concatenate(column)
        data.append(torch.as_tensor(packed, device=device))
        del packed
        column.clear()
    return data


def summarize(games):
    results = Counter(row["result"] for row in games)
    summary = {
        "games": len(games), "wins": results["win"], "losses": results["loss"],
        "draws": results["draw"], "mean_locks": float(np.mean([row["locks"] for row in games])),
        "win_rate": results["win"] / len(games),
        "mean_reward": float(np.mean([row["reward"] for row in games])),
        "play_rate": sum(row["plays"] for row in games) / sum(row["steps"] for row in games),
    }
    for key in ("elixir_trade", "tower_damage_dealt", "tower_damage_taken", "elixir_leaked"):
        if key in games[0]:
            summary[f"mean_{key}"] = sum(row[key] for row in games) / len(games)
    for field in ("reward_terms", "reward_step_abs"):
        totals = defaultdict(float)
        for row in games:
            for term, value in row[field].items():
                totals[term] += value
        summary[f"mean_{field}"] = {term: value / len(games) for term, value in totals.items()}
    return summary


def resume_history(output, latest):
    existing = output / "selfplay.json"
    try:
        history = json.loads(existing.read_text(encoding="utf-8")).get("updates", [])
    except (FileNotFoundError, ValueError):
        history = []
    update = latest["update"]
    recorded = {row["update"]: row for row in history if row["update"] <= update}
    for event in read_run_journal(output):
        if event.get("event") == "ppo_update" and event["done"] <= update:
            recorded[event["done"]] = event["metrics"]
    recorded[update] = latest
    return [recorded[number] for number in sorted(recorded)]


def restore_training_state(trainer, optimizer, rng, device):
    optimizer.load_state_dict(trainer["optimizer"])
    rng.bit_generator.state = trainer["numpy_rng"]
    torch.set_rng_state(trainer["torch_rng"].cpu())
    if device == "cuda" and trainer["cuda_rng"]:
        torch.cuda.set_rng_state_all([state.cpu() for state in trainer["cuda_rng"]])


def training_state(reference_hash, optimizer, rng, device, **extra):
    return {
        "reference_hash": reference_hash, "optimizer": optimizer.state_dict(),
        "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
        **extra,
    }


def run_selfplay(args, device, bc_path, output, report_progress=None):
    layout = torch.channels_last if device == "cpu" else torch.preserve_format
    policy, saved = load_checkpoint(bc_path, args.deck, device, memory_format=layout)
    if saved["phase"] != "bc":
        raise ValueError("O checkpoint de referencia precisa vir da fase BC.")
    reference = frozen_policy(policy)
    with bc_path.open("rb") as checkpoint:
        reference_hash = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    start_update = 0
    if args.evaluate_checkpoint or args.resume_ppo:
        policy, ppo_saved = load_checkpoint(
            args.evaluate_checkpoint or output / "ppo.pt", args.deck, device,
            memory_format=layout,
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
    obs_profile = policy.obs_profile
    reward_profile = getattr(args, "reward", None) or "potential"
    kl_direction = getattr(args, "kl_direction", "reverse")
    kl_decay = not getattr(args, "no_kl_decay", False)
    other_deck = "icebow" if args.deck == "xbow_30" else "xbow_30"
    rivals = [
        Rival("bc", args.deck, reference), Rival("scripted", other_deck),
        Rival("hog_push", "hog_26"), Rival("rg_patient", "royal_giant"),
    ]
    other_path = output.parent / other_deck / "bc.pt"
    if other_path.is_file():
        other_bc, _ = load_checkpoint(other_path, other_deck, device, memory_format=layout)
        if other_bc.obs_profile == obs_profile:
            rivals.append(Rival("other_bc", other_deck, frozen_policy(other_bc)))
    evaluation_rivals = rivals + [Rival("random", args.deck)]
    league = list(rivals)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    rng = np.random.default_rng(args.seed)
    extra_trainer = {}
    pfsp = PFSPPool()
    if args.resume_ppo:
        restore_training_state(trainer, optimizer, rng, device)
        if "free_model" in trainer:
            extra_trainer["free_model"] = trainer["free_model"]
        if "pfsp_wr" in trainer:
            pfsp = PFSPPool(trainer["pfsp_wr"])
        snapshots = sorted((output / "league").glob("*.pt"))
        snapshots = [path for path in snapshots if int(path.stem) <= start_update]
        for path in snapshots[-6:]:
            previous, _ = load_checkpoint(path, args.deck, device, memory_format=layout)
            league.append(Rival(f"snapshot_{int(path.stem)}", args.deck, frozen_policy(previous)))
    default_parallel = (
        CPU_TRAINING_PARALLEL_GAMES
        if args.ppo_updates and device == "cpu" and torch.get_num_threads() >= 4
        else DEFAULT_PARALLEL_GAMES
    )
    parallel_games = getattr(args, "parallel_games", None) or default_parallel
    requested_workers = getattr(args, "rollout_workers", None)
    rollout_workers = (
        default_rollout_workers(parallel_games, device)
        if requested_workers is None else max(1, int(requested_workers))
    )
    if device != "cpu":
        rollout_workers = 1
    use_pool = (
        rollout_workers > 1
        or (requested_workers is None and device == "cpu" and directml_available())
    )
    worker_pool = (
        create_rollout_pool(
            rollout_workers, obs_profile, reward_profile, output,
            parallel_games=max(parallel_games, args.rollout_games),
        )
        if use_pool else None
    )
    envs = defaultdict(list)

    def play_matches(stage, tested, matches, *, collect=False, policy_key="policy",
                     force_sync_policy=True, **metadata):
        if worker_pool is not None and len(matches) > 1:
            yield from worker_pool.run_matches(
                args.deck, stage, tested, matches, collect=collect,
                policy_key=policy_key, force_sync_policy=force_sync_policy, **metadata,
            )
            return
        for wave in batched(matches, parallel_games):
            used = Counter()
            environments = []
            for match in wave:
                decks = (args.deck, match.rival.deck)
                if match.seat == 1:
                    decks = decks[::-1]
                slot = used[decks]
                if slot == len(envs[decks]):
                    envs[decks].append(battle_env(
                        *decks, reward_profile=reward_profile, obs_profile=obs_profile,
                    ))
                environments.append(envs[decks][slot])
                used[decks] += 1
            results = play_wave(output, stage, tested, wave, environments, device,
                                collect=collect, **metadata)
            yield from zip(wave, results, strict=True)

    ppo_metrics = []
    previous_updates = resume_history(output, ppo_saved["metrics"]) if args.resume_ppo else []

    try:
        if worker_pool is not None:
            worker_pool.__enter__()
        final_update = start_update + args.ppo_updates
        if report_progress and args.ppo_updates:
            report_progress("ppo_start", done=start_update, total=final_update,
                            resumed=args.resume_ppo,
                            parallel_games=min(parallel_games, args.rollout_games),
                            rollout_workers=rollout_workers)
        for update in range(start_update, final_update):
            check_training_stop()
            began = perf_counter()
            games, chunks = [], []
            # PFSP sampling focuses more games on lower win-rate rivals while keeping exploration.
            rival_names = [rival.name for rival in league]
            sampled = pfsp.sample_indices(rival_names, args.rollout_games, rng)
            assignments = [
                (index, (update + i) % 2, int(rng.integers(2**31)))
                for i, index in enumerate(sampled)
            ]
            matches = [
                Match(league[index], seat, seed, number)
                for number, (index, seat, seed) in enumerate(
                    sorted(assignments, key=lambda assignment: assignment[0]), 1)
            ]
            concurrent = min(parallel_games, len(matches))
            if report_progress:
                report_progress("ppo_match", update=update + 1, done=update, total=final_update,
                                games=len(matches), parallel_games=concurrent)
            done_games = 0
            for match, (game, rows) in play_matches("ppo", policy, matches, collect=True, update=update + 1):
                games.append(game)
                chunks.append(rows)
                pfsp.record(match.rival.name, game["result"])
                done_games += 1
                if report_progress:
                    report_progress("ppo_game", update=update + 1, total=final_update,
                                    done=done_games, games=args.rollout_games, rival=match.rival.name,
                                    parallel_games=concurrent)
            del rows
            simulation_seconds = perf_counter() - began
            sim_gph = len(games) / max(1e-6, simulation_seconds) * 3600.0
            print(
                f"[PPO {update + 1}/{final_update}] {len(games)} jogos simulados em {simulation_seconds:.1f}s "
                f"({sim_gph:,.0f} j/h) — otimizando politica...",
                flush=True,
            )
            began = perf_counter()
            data = rollout_tensors(chunks, device)
            chunks.clear()
            if update == 0 and not args.resume_ppo:
                warmup_value_head(policy, data, args.batch_size, updates=5)
            ref_runner = None
            if isinstance(worker_pool, GpuBatchedRolloutPool):
                worker_pool.publish_model("bc_ref", reference, force=False)
                ref_runner = worker_pool._runners.get("bc_ref")
            large_wave = device == "cpu" and len(data[3]) > 12_000
            if large_wave:
                data = stratified_ppo_subsample(data, max_samples=8192, rng=rng)
            ppo_epochs = 1 if large_wave else 3
            effective_kl = scheduled_kl_coef(args.kl_coef, update) if kl_decay else args.kl_coef
            divergence, ppo_stats = optimize_ppo(
                policy, reference, optimizer, data, args.batch_size, ppo_epochs, effective_kl,
                kl_direction=kl_direction, return_stats=True, reference_runner=ref_runner,
            )
            del data
            metrics = {**summarize(games), "kl_bc_to_ppo": divergence,
                       "ppo": ppo_stats, "reward_profile": reward_profile,
                       "rollout_workers": rollout_workers,
                       "simulation_seconds": round(simulation_seconds, 3),
                       "optimization_seconds": round(perf_counter() - began, 3),
                       "rivals": dict(Counter(match.rival.name for match in matches)), "update": update + 1}
            print(f"PPO {update + 1}/{final_update}: {metrics}", flush=True)
            ppo_metrics.append(metrics)
            save_checkpoint(output / "league" / f"{update + 1:05d}.pt", policy, args.deck,
                            "ppo", metrics)
            state = training_state(
                reference_hash, optimizer, rng, device,
                pfsp_wr=dict(pfsp.ema_wr), **extra_trainer,
            )
            save_checkpoint(output / "ppo.pt", policy, args.deck, "ppo", metrics, state)
            league.append(Rival(f"snapshot_{update + 1}", args.deck, frozen_policy(policy)))
            if report_progress:
                report_progress("ppo_update", done=update + 1, total=final_update, metrics=metrics,
                                parallel_games=concurrent)
            if len(league) > len(rivals) + 6:
                del league[len(rivals)]
        evaluations = {}
        if args.evaluate:
            has_ppo = bool(ppo_metrics or args.evaluate_checkpoint or args.resume_ppo)
            concurrent = min(parallel_games, args.evaluate)
            if report_progress:
                total = len(evaluation_rivals) * (2 if has_ppo else 1)
                report_progress("evaluation_start", done=0, total=total, parallel_games=concurrent)
            for rival in evaluation_rivals:
                for label, tested in (("bc", reference), ("ppo", policy)):
                    if label == "ppo" and not has_ppo:
                        break
                    matches = [
                        Match(rival, i % 2, 1_000_000 + args.seed + i // 2, i + 1)
                        for i in range(args.evaluate)
                    ]
                    games = [
                        game for _, game in play_matches(
                            "evaluation", tested, matches, model=label,
                            policy_key=f"eval:{label}", force_sync_policy=True,
                        )
                    ]
                    name = f"{label}_vs_{rival.name}"
                    evaluations[name] = summarize(games)
                    print(f"{name}: {evaluations[name]}", flush=True)
                    if report_progress:
                        report_progress("evaluation_pair", done=len(evaluations), total=total,
                                        parallel_games=concurrent)
        output.mkdir(parents=True, exist_ok=True)
        report = "evaluation.json" if args.evaluate and not (ppo_metrics or args.resume_ppo) else "selfplay.json"
        (output / report).write_text(
            json.dumps({"updates": previous_updates + ppo_metrics,
                        "evaluation": evaluations}, indent=2),
            encoding="utf-8",
        )
    finally:
        if worker_pool is not None:
            worker_pool.close()
        for pool in envs.values():
            for env in pool:
                env.close()
