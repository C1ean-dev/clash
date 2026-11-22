"""Select winning decks and reconstruct decision samples from IL_Replay."""

from collections import Counter, deque
from collections.abc import Iterator
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from royalegym.protocol import MatchSetup, ShuffleMode, ability_row
from royaleimitate.replays import CARD_NAMES, base_key, match_plan
from xbow_metrics import check_training_stop


DECKS = {
    "xbow_30": (
        "x-bow", "tesla", "archers", "knight", "skeletons",
        "electro-spirit", "fireball", "the-log",
    ),
    "icebow": (
        "x-bow", "tesla", "ice-wizard", "knight", "skeletons",
        "tornado", "rocket", "the-log",
    ),
}
SIDES = ("team", "opponent")
SEATS = ("blue", "red")
DECK_BY_CARDS = {frozenset(cards): deck for deck, cards in DECKS.items()}
REPLAY_COLUMNS = ("replay_tag", "result", "payload_json")


def parquet_parts(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        replay_dir = path / "replays" if (path / "replays").is_dir() else path
        parts = sorted(replay_dir.glob("part-*.parquet"))
        if parts:
            return parts
    raise ValueError(f"Nenhuma particao de replays encontrada em {path}")


@dataclass(frozen=True)
class WinningReplay:
    tag: str
    seat: int
    payload: dict


def replay_batches(part):
    with pq.ParquetFile(part) as parquet:
        missing = set(REPLAY_COLUMNS) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{part}: colunas ausentes: {', '.join(sorted(missing))}")
        yield from parquet.iter_batches(columns=REPLAY_COLUMNS, batch_size=512)


EXCLUDED_CASUAL_BATTLE_TYPES = frozenset({"friendly", "clanMate"})


def eligible_replay(tag, result, encoded, deck_by_cards, counts, *,
                    competitive_only=False, keep_tags=None):
    if result not in ("victory", "defeat"):
        return None
    seat = int(result == "defeat")
    try:
        payload = json.loads(encoded)
        battle = payload["battle"]
        players = [battle[side]["players"] for side in SIDES]
        if any(len(side) != 1 for side in players):
            return None
        cards = players[seat][0]["deck"]
        deck = deck_by_cards.get(frozenset(base_key(card["card_key"]) for card in cards))
        if len(cards) != 8 or deck is None:
            return None
        if not isinstance(tag, str) or not tag:
            raise ValueError("replay_tag ausente")
        if not isinstance(payload["events"], list):
            raise ValueError("events deve ser uma lista")
        if (competitive_only and (not keep_tags or tag not in keep_tags)
                and battle.get("battle_type") in EXCLUDED_CASUAL_BATTLE_TYPES):
            counts["casual_filtered"] += 1
            return None
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        counts["invalid_rows"] += 1
        return None
    return deck, WinningReplay(tag, seat, payload)


def winning_replays(
    path: Path, deck: str, counts: Counter, *,
    competitive_only=False, keep_tags=None, deck_cards=None,
) -> Iterator[WinningReplay]:
    cards = deck_cards if deck_cards is not None else DECKS[deck]
    deck_by_cards = {frozenset(cards): deck}
    seen: set[str] = set()
    for part in parquet_parts(path):
        for batch in replay_batches(part):
            tags, results, payloads = (batch.column(key).to_pylist() for key in REPLAY_COLUMNS)
            for tag, result, encoded in zip(tags, results, payloads, strict=True):
                counts["rows"] += 1
                match = eligible_replay(
                    tag, result, encoded, deck_by_cards, counts,
                    competitive_only=competitive_only, keep_tags=keep_tags,
                )
                if match is None:
                    continue
                if tag in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(tag)
                counts["winning_matches"] += 1
                yield match[1]


def candidate_replays(part):
    counts = Counter()
    can_filter_xbow = all("x-bow" in cards for cards in DECKS.values())
    for batch in replay_batches(part):
        payloads = batch.column("payload_json")
        if can_filter_xbow and (pa.types.is_string(payloads.type) or pa.types.is_large_string(payloads.type)):
            # Keep escaped JSON too. The regular parser determines deck eligibility.
            candidates = pc.or_(pc.match_substring(payloads, "x-bow"),
                                pc.match_substring(payloads, r"\u"))
            batch = batch.filter(candidates)
        tags, results, payloads = (batch.column(key).to_pylist() for key in REPLAY_COLUMNS)
        for tag, result, encoded in zip(tags, results, payloads, strict=True):
            match = eligible_replay(tag, result, encoded, DECK_BY_CARDS, counts)
            if match:
                yield match


def replay_tags(part):
    tags_by_deck = {deck: set() for deck in DECKS}
    for deck, replay in candidate_replays(part):
        tags_by_deck[deck].add(replay.tag)
    return {deck: sorted(tags) for deck, tags in tags_by_deck.items()}


ID_PLANE_KEYS = ("card_ids", "spell_ids", "unit_ids")


def board_value(state, seat: int) -> float:
    """Zero-sum, left/right-symmetric board advantage in [-1, 1] for BC critic pretraining."""
    own = state.players[seat]
    foe = state.players[1 - seat]
    crown_diff = (own.crowns - foe.crowns) / 3.0
    own_towers = sum(hp / max(1, maximum) for hp, maximum in zip(own.tower_hp, own.tower_max_hp))
    foe_towers = sum(hp / max(1, maximum) for hp, maximum in zip(foe.tower_hp, foe.tower_max_hp))
    tower_diff = (own_towers - foe_towers) / 3.0
    elixir_diff = max(-1.0, min(1.0, (own.elixir_milli - foe.elixir_milli) / 10000.0))
    return float(max(-1.0, min(1.0, 0.5 * crown_diff + 0.4 * tower_diff + 0.1 * elixir_diff)))


@dataclass(frozen=True)
class MatchSamples:
    tag: str
    spatial: np.ndarray
    vector: np.ndarray
    mask: np.ndarray
    actions: np.ndarray
    id_planes: np.ndarray | None = None
    values: np.ndarray | None = None

    def frame(self, index):
        ids = self.id_planes[index] if self.id_planes is not None else None
        return self.spatial[index], self.vector[index], self.mask[index], ids

    def hands(self, acted, hand):
        return self.vector[acted, hand]


def reconstruct(env, replay: WinningReplay, seed: int, capture=None) -> MatchSamples | str:
    names = {card.name: card.card_id for card in env.engine.cards()}
    try:
        plan = match_plan(replay.payload, frozenset(names))
    except (KeyError, IndexError, TypeError, ValueError):
        return "invalid_events"
    if isinstance(plan, str):
        return plan
    if not plan.plays[replay.seat]:
        return "no winning-side plays"
    players = [replay.payload["battle"][side]["players"][0] for side in SIDES]
    facts = [
        {CARD_NAMES.get(base_key(card["card_key"])): card for card in player["deck"]}
        for player in players
    ]
    levels = [[int(facts[seat][name].get("level") or 11) for name in deal]
              for seat, deal in enumerate(plan.decks)]
    forms = [[
        2 if facts[seat][name]["card_key"].endswith("-hero")
        else int(facts[seat][name]["card_key"].endswith("-ev1"))
        for name in deal
    ] for seat, deal in enumerate(plan.decks)]
    tower_levels = [int((player.get("tower_card") or {}).get("level") or 11)
                    for player in players]
    setup = MatchSetup(
        decks=[[names[name] for name in deal] for deal in plan.decks],
        shuffle=int(ShuffleMode.NONE), levels=levels, forms=forms, tower_levels=tower_levels,
    )
    try:
        obs, _ = env.reset(seed=seed, options={"setup": setup})
    except ValueError:
        return "simulator refused replay setup"
    plays = [iter(seat_plays) for seat_plays in plan.plays]
    pending = [deque(), deque()]
    for event in sorted(replay.payload["events"],
                        key=lambda event: (event["replay_tick_20hz"], event["source_index"])):
        if event["side"] not in SIDES:
            continue
        seat = SIDES.index(event["side"])
        if event["kind"] == "play_card":
            pending[seat].append((event["replay_tick_20hz"], next(plays[seat]), ()))
        elif event["kind"] == "activate_ability":
            candidates = tuple(
                names.get(CARD_NAMES.get(base_key(key)))
                for key in event.get("ability_source_candidates") or ()
            )
            pending[seat].append((event["replay_tick_20hz"], None, candidates))
    parser = env.action_parser
    winner = SEATS[replay.seat]
    spatial, vector, masks, labels, id_planes, values = [], [], [], [], [], []
    has_id_planes = all(key in obs[winner] for key in ID_PLANE_KEYS)

    # Stop at the final logged play. The eventual simulator winner is irrelevant.
    while env.agents and any(pending):
        if capture:
            check_training_stop()
        state = env.battle_state
        commands = dict.fromkeys(SEATS, 0)
        for seat, agent in enumerate(SEATS):
            if not pending[seat] or pending[seat][0][0] > state.tick:
                continue
            _, play, candidates = pending[seat].popleft()
            player = state.players[seat]
            legal = obs[agent]["action_mask"]
            if play is None:
                buttons = [
                    parser.n_tile_actions + button
                    for button, row in enumerate(player.abilities)
                    if legal[parser.n_tile_actions + button]
                    and (not candidates or ability_row(row).card_id in candidates)
                ]
                if len(buttons) != 1:
                    return "ability unavailable or ambiguous"
                action, = buttons
            else:
                card = names[play.card]
                if not (0 <= play.x < 18000 and 0 <= play.y < 32000):
                    return "placement outside arena"
                if card not in player.hand:
                    return "card not in hand"
                x, y = play.x * parser.pitch_div // 1000, play.y * parser.pitch_div // 1000
                if seat == 1:
                    x, y = parser.nx - 1 - x, parser.ny - 1 - y
                action = parser.encode(player.hand.index(card), x, y)
            if not legal[action]:
                return "play refused by engine"
            commands[agent] = action
        winning_obs = obs[winner]
        if capture:
            capture.observe(state, tuple(commands[agent] for agent in SEATS),
                            example_index=len(labels) if commands[winner] else None)
        spatial.append(winning_obs["spatial"].astype(np.float16))
        vector.append(winning_obs["vector"].astype(np.float16))
        masks.append(np.packbits(winning_obs["action_mask"]))
        if has_id_planes:
            id_planes.append(np.concatenate([winning_obs[k] for k in ID_PLANE_KEYS], axis=0).astype(np.uint8, copy=False))
        labels.append(commands[winner])
        values.append(board_value(state, replay.seat))
        obs, _, _, _, _ = env.step(commands)

    if any(pending):
        return "simulator ended before replay"
    return MatchSamples(
        replay.tag, np.stack(spatial), np.stack(vector), np.stack(masks), np.array(labels),
        np.stack(id_planes) if has_id_planes else None,
        np.asarray(values, dtype=np.float16),
    )
