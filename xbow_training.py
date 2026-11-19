"""Masked policy, behavioral cloning, and PPO losses shared by both decks."""

from collections import Counter
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryFile
import zlib

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, kl_divergence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from royaleimitate.replays import CARD_NAMES
from xbow_arena import ArenaCapture, arena_enabled
from xbow_metrics import check_training_stop

from xbow_replays import reconstruct, winning_replays


PPO_REFERENCE_CACHE_BYTES = 128 * 1024**2


class XBowPolicy(nn.Module):
    def __init__(self, spatial_channels: int, vector_size: int, actions: int, architecture=2,
                 placement_prior=None, hand_start=0):
        super().__init__()
        self.spec = (spatial_channels, vector_size, actions)
        self.architecture = architecture
        self.hand_start = hand_start
        self.register_buffer("placement_prior", placement_prior)
        conditioned = architecture == 2
        self.board = nn.Sequential(
            nn.Conv2d(spatial_channels + 2 * conditioned, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(),
        )
        self.context = nn.Sequential(
            nn.Linear(vector_size + 24 * 8 * 4, 128), nn.LayerNorm(128), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((8, 4))
        self.placement = nn.Conv2d(24 if conditioned else 40, 4, 1)
        self.condition = nn.Linear(128, 4 * 24 if conditioned else 16)
        self.slot_bias = nn.Linear(128, 4)
        self.special = nn.Linear(128, actions - 4 * 32 * 18)
        self.tile_bias = nn.Parameter(torch.zeros(4, 32, 18))
        self.value = nn.Linear(128, 1)
        if conditioned:
            y, x = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 18),
                                  indexing="ij")
            self.register_buffer("coordinates", torch.stack((x, y)), persistent=False)

    def forward(self, spatial, vector, mask):
        logits, value = self.logits_and_value(spatial, vector, mask)
        return Categorical(logits=logits, validate_args=False), value

    def logits_and_value(self, spatial, vector, mask):
        if self.architecture == 2:
            spatial = torch.cat((spatial, self.coordinates.expand(len(spatial), -1, -1, -1)), 1)
        board = self.board(spatial)
        context = self.context(torch.cat((self.pool(board).flatten(1), vector), dim=1))
        if self.architecture == 2:
            condition = self.condition(context).reshape(-1, 4, 24)
            compatibility = torch.einsum("bsc,bcyx->bsyx", condition, board) / 24**0.5
            tiles = (self.placement(board) + self.tile_bias + compatibility).flatten(2)
            if self.placement_prior is not None:
                classes = len(self.placement_prior)
                hand = vector[:, self.hand_start:self.hand_start + 4 * classes]
                cards = hand.reshape(-1, 4, classes).argmax(-1)
                tiles = tiles + self.placement_prior[cards]
            legal_tiles = mask[:, 1:1 + 4 * 32 * 18].reshape(-1, 4, 32 * 18)
            tiles = tiles.masked_fill(~legal_tiles, -1e9).log_softmax(dim=-1)
        else:
            condition = self.condition(context)[:, :, None, None].expand(-1, -1, 32, 18)
            tiles = (self.placement(torch.cat((board, condition), dim=1)) + self.tile_bias).flatten(2)
        tiles = tiles + self.slot_bias(context).unsqueeze(-1)
        special = self.special(context)
        plays = torch.cat((tiles.flatten(1), special[:, 1:]), dim=1)
        plays = plays.masked_fill(~mask[:, 1:], -1e9).log_softmax(dim=1)
        logits = torch.cat((special[:, :1], plays), dim=1)
        return logits.masked_fill(~mask, -1e9), self.value(context).squeeze(-1)

    @classmethod
    def from_observation(cls, obs, placement_prior=None, hand_start=0):
        return cls(obs["spatial"].shape[0], obs["vector"].size, obs["action_mask"].size,
                   placement_prior=placement_prior, hand_start=hand_start)

    @torch.no_grad()
    def decide(self, obs, device, deterministic=False, generator=None):
        distribution, value = self(
            torch.as_tensor(obs["spatial"], device=device).unsqueeze(0),
            torch.as_tensor(obs["vector"], device=device).unsqueeze(0),
            torch.as_tensor(obs["action_mask"], device=device, dtype=torch.bool).unsqueeze(0),
        )
        if deterministic:
            choice = distribution.probs[:, 1:].argmax(-1) + 1
            action = torch.where(distribution.probs[:, 0] >= 0.5, 0, choice)
        elif generator is not None:
            action = torch.multinomial(distribution.probs, 1, generator=generator).squeeze(-1)
        else:
            action = distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item()), float(value.item())


def checkpoint_data(policy, deck, phase, metrics, trainer=None):
    return {
        "version": 1, "deck": deck, "phase": phase, "spec": policy.spec,
        "architecture": policy.architecture, "policy": policy.state_dict(), "metrics": metrics,
        "hand_start": policy.hand_start, "trainer": trainer,
    }


def save_checkpoint(path, policy, deck, phase, metrics, trainer=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as temporary:
        local = Path(temporary.name)
    try:
        torch.save(checkpoint_data(policy, deck, phase, metrics, trainer), local)
        os.replace(local, path)
    finally:
        local.unlink(missing_ok=True)


def load_checkpoint(path, deck, device, *, memory_format=torch.preserve_format):
    saved = torch.load(path, map_location=device, weights_only=True)
    return policy_from_checkpoint(saved, deck, device, memory_format=memory_format), saved


def policy_from_checkpoint(saved, deck, device, *, memory_format=torch.preserve_format):
    if saved.get("version") != 1 or saved.get("deck") != deck:
        raise ValueError(f"Checkpoint incompativel com {deck}")
    architecture = saved.get("architecture", 1)
    if architecture not in (1, 2):
        raise ValueError(f"Arquitetura desconhecida no checkpoint: {architecture}")
    policy = XBowPolicy(
        *saved["spec"], architecture=architecture,
        placement_prior=saved["policy"].get("placement_prior"),
        hand_start=saved.get("hand_start", 0),
    ).to(device=device, memory_format=memory_format)
    policy.load_state_dict(saved["policy"])
    return policy.eval()


def kata_weights(labels: np.ndarray, action_rate: float) -> np.ndarray:
    """KataCR Eq. 8, with the wait-frame weight before the first play."""
    ticks = np.arange(len(labels))
    latest = np.maximum.accumulate(np.where(labels != 0, ticks, -1))
    wait_weight = 1.0 / (1.0 - action_rate)
    return np.where(
        latest < 0, wait_weight,
        np.maximum(wait_weight, 1.0 / (action_rate * (ticks - latest + 1))),
    )


class ReplayFrames(Dataset):
    def __init__(self, matches, action_count):
        self.matches = matches
        self.ends = np.cumsum([len(match.actions) for match in matches])
        self.action_count = action_count
        self.timing_correction = None

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        match_id = int(np.searchsorted(self.ends, index, side="right"))
        frame = index - (self.ends[match_id - 1] if match_id else 0)
        match = self.matches[match_id]
        spatial, vector, mask = match.frame(frame)
        data = (
            np.array(spatial), np.array(vector),
            np.unpackbits(mask, count=self.action_count).astype(bool),
            match.actions[frame],
        )
        if self.timing_correction is not None:
            return (*data, self.timing_correction[index])
        return data

    def weights(self):
        plays = sum(np.count_nonzero(match.actions) for match in self.matches)
        rate = plays / len(self)
        if not 0 < rate < 1:
            return np.ones(len(self))
        return np.concatenate([kata_weights(match.actions, rate) for match in self.matches])


class DiskMatchSamples:
    """Store and read each compressed observation independently for random sampling."""

    def __init__(self, match, store):
        self.tag = match.tag
        self.actions = match.actions
        self.store = store
        self.dtype = np.dtype([
            (name, values.dtype, values.shape[1:])
            for name, values in (("spatial", match.spatial), ("vector", match.vector),
                                 ("mask", match.mask))
        ])
        self.offsets = np.empty(len(self.actions) + 1, dtype=np.int64)
        offset = store.tell()
        row = np.empty(1, dtype=self.dtype)
        for index, frame in enumerate(zip(match.spatial, match.vector, match.mask, strict=True)):
            if index % 64 == 0:
                check_training_stop()
            self.offsets[index] = offset
            row[0] = frame
            offset += store.write(zlib.compress(row, level=1))
        self.offsets[-1] = offset

    def frame(self, index):
        offset, end = self.offsets[index:index + 2]
        self.store.seek(offset)
        raw = zlib.decompress(self.store.read(end - offset))
        spatial, vector, mask = np.frombuffer(raw, dtype=self.dtype, count=1)[0]
        return spatial, vector, mask

    def hands(self, acted, hand):
        indices = np.flatnonzero(acted)
        vectors = np.empty((len(indices), hand.stop - hand.start),
                           dtype=self.dtype["vector"].base)
        for row, index in enumerate(indices):
            _, vector, _ = self.frame(index)
            vectors[row] = vector[hand]
        return vectors


def collect_replays(env, parquet_path, deck, limit, seed, frame_store=None, validation_tags=None,
                    report=None, arena_output=None, examples=None):
    counts, rejected = Counter(), Counter()
    matches = []
    fixed_validation = validation_tags is not None
    heldout = []
    remaining = set(validation_tags or ())
    if fixed_validation and (not remaining or len(remaining) >= limit):
        raise ValueError("O limite deve acomodar a validação fixa e pelo menos um replay de treino.")
    training_limit = limit - len(remaining)
    for replay in winning_replays(parquet_path, deck, counts):
        check_training_stop()
        is_validation = fixed_validation and replay.tag in remaining
        if fixed_validation and not is_validation and len(matches) >= training_limit:
            continue
        recording = (ArenaCapture(arena_output, "replays", env, replay=replay.tag,
                                  seat=replay.seat, number=len(matches) + len(heldout) + 1)
                     if arena_output is not None else nullcontext())
        with recording as capture:
            samples = reconstruct(env, replay, seed, capture)
            if capture and not isinstance(samples, str) and capture.example and examples is not None:
                examples[replay.tag] = (replay.seat, *capture.example)
        if isinstance(samples, str):
            if is_validation:
                raise ValueError(f"Replay reservado {replay.tag}: {samples}")
            rejected[samples] += 1
            continue
        if frame_store is not None:
            samples = DiskMatchSamples(samples, frame_store)
        if is_validation:
            heldout.append(samples)
            remaining.remove(replay.tag)
        else:
            matches.append(samples)
        total = len(matches) + len(heldout)
        if total % 10 == 0:
            print(f"{deck}: {total} replays reconstruidos", flush=True)
        if report:
            report("replays", done=total, total=limit, rejected=sum(rejected.values()))
        if total == limit:
            break
    total = len(matches) + len(heldout)
    if report:
        report("replays", done=total, total=limit, rejected=sum(rejected.values()),
               complete=True)
    print(f"{deck}: {dict(counts)} | reconstruidos={total} | recusas={dict(rejected)}",
          flush=True)
    if fixed_validation:
        if remaining or not matches:
            raise ValueError(f"Conjunto fixo incompleto: faltam {len(remaining)} replays reservados.")
        return matches, heldout
    if len(matches) < 2:
        raise ValueError("Sao necessarios dois replays reconstruidos para treino e validacao.")
    order = np.random.default_rng(seed).permutation(len(matches))
    validation_count = max(1, round(len(matches) * 0.2))
    return (
        [matches[i] for i in order[validation_count:]],
        [matches[i] for i in order[:validation_count]],
    )


@torch.no_grad()
def evaluate_clone(policy, batches, device, tactics=None):
    nll = count = play = expert_actions = play_probability = correct_slots = correct_tiles = 0
    tactic_totals, tactic_hits, tactic_placements = Counter(), Counter(), Counter()
    for spatial, vector, mask, target in batches:
        target = target.to(device)
        dist, _ = policy(spatial.to(device, torch.float32), vector.to(device, torch.float32),
                         mask.to(device))
        placement = dist.probs[:, 1:1 + 4 * 32 * 18].reshape(-1, 4, 32 * 18)
        predicted_slot = placement.sum(-1).argmax(-1)
        likely_tile = placement[torch.arange(len(target), device=device), predicted_slot].argmax(-1)
        predicted_card_action = 1 + predicted_slot * (32 * 18) + likely_tile
        nll -= dist.log_prob(target).sum().item()
        count += target.numel()
        expert_actions += (target != 0).sum().item()
        play_probability += (1 - dist.probs[:, 0]).sum().item()
        card_target = (target > 0) & (target <= 4 * 32 * 18)
        play += card_target.sum().item()
        correct_slots += (card_target & (predicted_slot == (target - 1) // (32 * 18))).sum().item()
        correct_tiles += (card_target & (predicted_card_action == target)).sum().item()
        if tactics is not None:
            hand, groups = tactics
            hold = vector[:, hand].reshape(len(target), 4, -1).argmax(-1).to(device)
            actual_slot = ((target - 1) // (32 * 18)).clamp(0, 3)
            trained_card = hold.gather(1, actual_slot[:, None]).squeeze(1)
            actual_tile = (target - 1) % (32 * 18)
            placement_with_card = placement[
                torch.arange(len(target), device=device), actual_slot,
            ].argmax(-1)
            distance = (
                (actual_tile // 18 - likely_tile // 18).abs()
                + (actual_tile % 18 - likely_tile % 18).abs()
            )
            conditional_distance = (
                (actual_tile // 18 - placement_with_card // 18).abs()
                + (actual_tile % 18 - placement_with_card % 18).abs()
            )
            for name, ids in groups.items():
                group = card_target & torch.isin(trained_card, ids)
                tactic_totals[name] += group.sum().item()
                hit = group & (predicted_slot == actual_slot) & (distance <= 2)
                tactic_hits[name] += hit.sum().item()
                tactic_placements[name] += (group & (conditional_distance <= 2)).sum().item()
    metrics = {
        "nll": nll / count, "frames": count, "card_plays": play,
        "conditional_card_accuracy": correct_slots / max(1, play),
        "conditional_card_and_tile_accuracy": correct_tiles / max(1, play),
        "mean_play_probability": play_probability / count, "expert_play_rate": expert_actions / count,
    }
    if tactics is not None:
        metrics["card_and_within_two_tiles"] = {
            name: {"correct": tactic_hits[name], "examples": total}
            for name, total in tactic_totals.items()
        }
        metrics["within_two_tiles_given_card"] = {
            name: {"correct": tactic_placements[name], "examples": total}
            for name, total in tactic_totals.items()
        }
    return metrics


def own_hand_field(env):
    offset = 0
    for field in env.obs_builder.vector_layout():
        if field.key == "own_hand_cards":
            return slice(offset, offset + field.size)
        offset += field.size
    raise ValueError("O ambiente nao informa o campo da mao propria.")


def fitted_placement_prior(matches, hand):
    """Smoothed per-card placement frequencies fitted on training matches only."""
    classes = (hand.stop - hand.start) // 4
    tiles = 32 * 18
    counts = np.zeros((classes, tiles), dtype=np.float32)
    for match in matches:
        acted = (match.actions > 0) & (match.actions <= 4 * tiles)
        slots, cells = np.divmod(match.actions[acted] - 1, tiles)
        hands = match.hands(acted, hand).reshape(-1, 4, classes)
        cards = hands[np.arange(len(slots)), slots].argmax(-1)
        np.add.at(counts, (cards, cells), 1)
    grid = torch.from_numpy(counts.reshape(classes, 1, 32, 18))
    grid = (grid + grid.flip(-1)) / 2
    kernel = torch.tensor([[[[1, 2, 1], [2, 8, 2], [1, 2, 1]]]], dtype=torch.float32) / 20
    smooth = nn.functional.conv2d(grid, kernel, padding=1).flatten(1)
    return 0.8 * (smooth + 0.05).log()


def tactical_fields(env, device):
    hand = own_hand_field(env)
    cards = env.engine.cards()
    groups = {
        "x-bow": [card.card_id for card in cards if card.name == CARD_NAMES["x-bow"]],
        "tesla": [card.card_id for card in cards if card.name == CARD_NAMES["tesla"]],
        "one_elixir": [card.card_id for card in cards if card.elixir == 1],
    }
    return hand, {name: torch.tensor(ids, device=device) for name, ids in groups.items()}


def replay_validation(env, args, device, output):
    reference, saved = load_checkpoint(args.bc_checkpoint, args.deck, device)
    if saved["phase"] != "bc":
        raise ValueError("A validacao de replays requer o checkpoint de imitação.")
    metrics = saved["metrics"]
    remaining = set(metrics["validation_matches"])
    if not remaining or remaining.intersection(metrics["training_matches"]):
        raise ValueError("Checkpoint sem divisão de partidas de validação válida.")
    matches = []
    for replay in winning_replays(args.parquet_path, args.deck, Counter()):
        if replay.tag not in remaining:
            continue
        reconstructed = reconstruct(env, replay, metrics.get("seed", args.seed))
        if isinstance(reconstructed, str):
            raise ValueError(f"{replay.tag}: {reconstructed}")
        matches.append(reconstructed)
        remaining.remove(replay.tag)
        if not remaining:
            break
    if remaining:
        raise ValueError(f"Faltam {len(remaining)} partidas de validação nos arquivos fornecidos.")
    batches = DataLoader(ReplayFrames(matches, reference.spec[-1]), batch_size=args.batch_size)
    tactics = tactical_fields(env, device)
    results = {"bc": evaluate_clone(reference, batches, device, tactics)}
    if args.evaluate_checkpoint:
        policy, ppo_checkpoint = load_checkpoint(args.evaluate_checkpoint, args.deck, device)
        if ppo_checkpoint["phase"] != "ppo" or policy.spec != reference.spec:
            raise ValueError("Checkpoint PPO incompatível com a referência BC.")
        results["ppo"] = evaluate_clone(policy, batches, device, tactics)
    output.mkdir(parents=True, exist_ok=True)
    (output / "replay_validation.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8",
    )
    print(json.dumps(results, indent=2), flush=True)


def clone_loss(dist, targets, timing_correction):
    """Use weighted samples for placement and restore the original play/wait rate."""
    played = targets.ne(0)
    log_wait = dist.logits[:, 0]
    log_play = dist.logits[:, 1:].logsumexp(dim=1)
    log_timing = torch.where(played, log_play, log_wait)
    log_choice = dist.log_prob(targets) - log_timing
    return -(timing_correction * log_timing + log_choice).mean()


def train_clone(env, args, device, path, report=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.disk_frames:
        with TemporaryFile(prefix="frames-", dir=path.parent) as frame_store:
            fit_clone(env, args, device, path, frame_store, report)
    else:
        fit_clone(env, args, device, path, report=report)


def fit_clone(env, args, device, path, frame_store=None, report=None):
    validation_tags = None
    if args.holdout_checkpoint:
        _, previous = load_checkpoint(args.holdout_checkpoint, args.deck, "cpu")
        if previous["phase"] != "bc":
            raise ValueError("A validação fixa deve vir de um checkpoint de imitação.")
        previous_metrics = previous["metrics"]
        if previous_metrics.get("seed", args.seed) != args.seed:
            raise ValueError("Use a mesma semente do checkpoint para reconstituir a validação fixa.")
        validation_tags = previous_metrics["validation_matches"]
    first_obs, _ = env.reset(seed=args.seed)
    if report:
        report("replays", done=0, total=args.matches, rejected=0)
    view_output = path.parent if arena_enabled() else None
    examples = {}
    train_matches, val_matches = collect_replays(
        env, args.parquet_path, args.deck, args.matches, args.seed, frame_store, validation_tags, report,
        view_output, examples,
    )
    if report:
        report("bc_start", total=args.epochs, training=len(train_matches),
               validation=len(val_matches))
    hand = own_hand_field(env)
    prior = fitted_placement_prior(train_matches, hand)
    policy = XBowPolicy.from_observation(first_obs["blue"], prior, hand.start).to(device)
    train = ReplayFrames(train_matches, policy.spec[-1])
    val = ReplayFrames(val_matches, policy.spec[-1])
    weights = train.weights()
    train.timing_correction = (weights.mean() / weights).astype(np.float32)
    sampler = WeightedRandomSampler(weights, len(train), replacement=True)
    training_batches = DataLoader(train, batch_size=args.batch_size, sampler=sampler)
    val_batches = DataLoader(val, batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4)
    with ExitStack() as recordings:
        probes = []
        for match, card in reversed(clone_examples(train_matches, examples)):
            seat, index, board = examples[match.tag]
            probe = clone_probe(match, index, board, seat, device, policy.spec[-1])
            capture = recordings.enter_context(
                ArenaCapture(view_output, "bc", env, replay=match.tag, seat=seat, probe_card=card)
            )
            probes.append((capture, probe))
        fit_clone_epochs(policy, train_matches, val_matches, training_batches, val_batches,
                         optimizer, args, device, path, report, probes)


def clone_examples(matches, examples):
    """Prefer different opening plays while keeping each comparison on a fixed board."""
    probe_limit = 3
    distinct, repeats = {}, []
    for match in matches:
        if match.tag not in examples:
            continue
        seat, _, board = examples[match.tag]
        card = next(move["card"] for move in board["moves"] if move["seat"] == seat)
        if card in distinct:
            if len(repeats) < probe_limit:
                repeats.append((match, card))
        else:
            distinct[card] = (match, card)
            if len(distinct) == probe_limit:
                break
    return (*distinct.values(), *repeats)[:probe_limit]


def clone_probe(match, index, board, seat, device, action_count):
    spatial, vector, mask, target = ReplayFrames([match], action_count)[index]
    features = (
        torch.as_tensor(spatial, dtype=torch.float32, device=device).unsqueeze(0),
        torch.as_tensor(vector, dtype=torch.float32, device=device).unsqueeze(0),
        torch.as_tensor(mask, device=device).unsqueeze(0),
    )
    return features, int(target), board, seat


@torch.no_grad()
def show_clone_example(capture, policy, probe, **metrics):
    features, target, board, seat = probe
    dist, _ = policy(*features)
    probs = dist.probs[0]
    best_play = int(probs[1:].argmax().item()) + 1
    chosen = 0 if probs[0] >= .5 else best_play
    hand = board["sides"][seat]["hand"]
    capture.frame(board, comparison={
        "human": capture.action(target, seat, hand), "model": capture.action(chosen, seat, hand),
        "play": float(1 - probs[0]), "target": float(probs[target]), "exact": chosen == target,
        "alternative": capture.action(best_play, seat, hand) if not chosen and probs[best_play] > 0 else None,
        **metrics,
    })


def fit_clone_epochs(policy, train_matches, val_matches, training_batches, val_batches,
                     optimizer, args, device, path, report, probes):
    best = float("inf")
    view_stride = max(15, len(training_batches) // 24)
    for epoch in range(args.epochs):
        policy.train()
        for batch, (spatial, vector, mask, target, correction) in enumerate(training_batches, 1):
            check_training_stop()
            dist, _ = policy(spatial.to(device, torch.float32), vector.to(device, torch.float32),
                             mask.to(device))
            loss = clone_loss(dist, target.to(device), correction.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            edge = batch in (1, len(training_batches))
            if report and (edge or batch % 15 == 0):
                report("bc_batch", epoch=epoch + 1, total=args.epochs, done=batch,
                       batches=len(training_batches), loss=loss.item())
            if probes and (edge or batch % view_stride == 0):
                for capture, probe in probes:
                    show_clone_example(capture, policy, probe, epoch=epoch + 1, batch=batch,
                                       batches=len(training_batches), loss=loss.item())
        policy.eval()
        check_training_stop()
        metrics = evaluate_clone(policy, val_batches, device)
        print(f"BC {epoch + 1}/{args.epochs}: {metrics}", flush=True)
        if report:
            report("bc_epoch", done=epoch + 1, total=args.epochs, metrics=metrics)
        if metrics["nll"] < best:
            best = metrics["nll"]
            save_checkpoint(path, policy, args.deck, "bc", {
                **metrics, "training_matches": [match.tag for match in train_matches],
                "validation_matches": [match.tag for match in val_matches],
                "epoch": epoch + 1, "seed": args.seed, "timing_importance_corrected": True,
            })


def optimize_ppo(policy, reference, optimizer, data, batch_size, updates, kl_coef):
    """Clipped PPO with the exact masked D_KL(BC || PPO) over joint actions."""
    spatial, vector, mask, actions, old_logp, advantage, returns = data
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    # The BC reference is frozen. Reuse its predictions for subsequent passes
    # when the CPU rollout fits the cache budget.
    reference_logits = None
    if (reference is not None and actions.device.type == "cpu" and updates > 1
            and mask.numel() * 4 <= PPO_REFERENCE_CACHE_BYTES):
        reference_logits = torch.empty(mask.shape, dtype=torch.float32, device=actions.device)
    divergences = []
    for epoch in range(updates):
        order = torch.randperm(len(actions), device=actions.device)
        for ids in order.split(batch_size):
            inputs = (spatial[ids].float(), vector[ids].float(), mask[ids])
            dist, value = policy(*inputs)
            ratio = (dist.log_prob(actions[ids]) - old_logp[ids]).exp()
            unclipped = ratio * advantage[ids]
            clipped = ratio.clamp(0.8, 1.2) * advantage[ids]
            loss = (
                -torch.minimum(unclipped, clipped).mean()
                + 0.5 * (returns[ids] - value).square().mean()
                - 0.005 * dist.entropy().mean()
            )
            if reference is not None:
                with torch.no_grad():
                    if reference_logits is not None and epoch > 0:
                        logits = reference_logits[ids]
                    else:
                        logits, _ = reference.logits_and_value(*inputs)
                        if reference_logits is not None:
                            reference_logits[ids] = logits
                    prior = Categorical(logits=logits, validate_args=False)
                divergence = kl_divergence(prior, dist).mean()
                loss = loss + kl_coef * divergence
                divergences.append(float(divergence.detach().item()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
    return float(np.mean(divergences)) if divergences else None


def frozen_policy(policy):
    return deepcopy(policy).eval().requires_grad_(False)
