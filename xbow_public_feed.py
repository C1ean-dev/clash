"""Turn chronological observations from a screen reader into policy inputs."""

from typing import Annotated, Literal

import msgspec
import numpy as np
from msgspec.structs import replace
from royalegym.protocol import (
    EMPTY_CARD, EntityKind, EntityState, SpellMotion, SpellState,
    STATUS_EVOLVED, STATUS_HERO,
)
from royaleimitate.replays import CARD_NAMES, base_key

from xbow_environment import battle_env


Count = Annotated[int, msgspec.Meta(ge=0)]
Positive = Annotated[int, msgspec.Meta(ge=1)]
X = Annotated[float, msgspec.Meta(ge=0, lt=18)]
Y = Annotated[float, msgspec.Meta(ge=0, lt=32)]
Position = tuple[X, Y]
Side = Literal["own", "opponent"]


class Towers(msgspec.Struct, forbid_unknown_fields=True):
    hp: tuple[Count, Count, Count]
    max_hp: tuple[Positive, Positive, Positive]
    crowns: Annotated[int, msgspec.Meta(ge=0, le=3)] = 0
    king_active: bool = False


class Unit(msgspec.Struct, forbid_unknown_fields=True):
    side: Side
    position: Position
    hp: Positive
    max_hp: Positive
    card: str | None = None
    kind: Literal["ground", "air", "building"] = "ground"
    radius: Annotated[float, msgspec.Meta(ge=0, le=4)] = 0.4
    deploying: bool = False
    stunned: bool = False
    form: Literal["basic", "evolved", "hero"] = "basic"
    footprint: tuple[Position, Position] | None = None


class Spell(msgspec.Struct, forbid_unknown_fields=True):
    side: Side
    position: Position
    card: str | None = None
    own_aim: Position | None = None


class Event(msgspec.Struct, forbid_unknown_fields=True):
    tick: Count
    side: Side
    card: str | None = None
    ability_cost: Annotated[int, msgspec.Meta(ge=0, le=10)] | None = None


class Frame(msgspec.Struct, forbid_unknown_fields=True):
    tick: Count
    hand: tuple[str | None, str | None, str | None, str | None]
    own_elixir_milli: Annotated[int, msgspec.Meta(ge=0, le=10000)]
    own_towers: Towers
    opponent_towers: Towers
    next_card: str | None = None
    units: list[Unit] = msgspec.field(default_factory=list)
    spells: list[Spell] = msgspec.field(default_factory=list)
    events: list[Event] = msgspec.field(default_factory=list)
    can_act: bool = True
    reset: bool = False


class UnitIdTracker:
    """Assign persistent UIDs across screen-reader frames by nearest-neighbor matching."""

    def __init__(self, max_distance_tiles: float = 4.5):
        self.max_dist_sq = float(max_distance_tiles) ** 2
        self.next_uid = 1
        self.prev: list[tuple[int, str, int, str, float, float]] = []

    def reset(self, first_uid: int):
        self.next_uid = int(first_uid)
        self.prev = []

    def assign(self, items: list[tuple[str, int, str, float, float]]) -> list[int]:
        """Match (side, card_id, kind, x, y) to previous frame tracks and return persistent UIDs."""
        if not items:
            self.prev = []
            return []
        candidates = []
        for cur_idx, (side, card_id, kind, x, y) in enumerate(items):
            for prev_idx, (uid, p_side, p_card, p_kind, px, py) in enumerate(self.prev):
                if side != p_side or card_id != p_card or kind != p_kind:
                    continue
                dist_sq = (x - px) ** 2 + (y - py) ** 2
                if dist_sq <= self.max_dist_sq:
                    candidates.append((dist_sq, cur_idx, prev_idx, uid))
        candidates.sort(key=lambda row: row[0])
        matched_cur: dict[int, int] = {}
        used_prev: set[int] = set()
        for _, cur_idx, prev_idx, uid in candidates:
            if cur_idx in matched_cur or prev_idx in used_prev:
                continue
            matched_cur[cur_idx] = uid
            used_prev.add(prev_idx)
        uids = []
        next_prev = []
        for cur_idx, (side, card_id, kind, x, y) in enumerate(items):
            uid = matched_cur.get(cur_idx)
            if uid is None:
                uid = self.next_uid
                self.next_uid += 1
            uids.append(uid)
            next_prev.append((uid, side, card_id, kind, x, y))
        self.prev = next_prev
        return uids


class PublicFeed:
    """One observer since the opening of a normal 1v1 battle."""

    def __init__(self, deck, *, obs_profile="v2"):
        self.obs_profile = obs_profile
        self.env = battle_env(deck, obs_profile=obs_profile)
        self.env.reset()
        self.opening = self.env.battle_state
        self.tick = None
        self.scale = self.env.engine.arena().subtile
        self.ids = {card.name: card.card_id for card in self.env.engine.cards()}
        self.name_by_id = {card.card_id: card.name for card in self.env.engine.cards()}
        self.unit_type_by_name = {name: idx for idx, name in enumerate(self.env.engine.unit_types())}
        self.keys = {key: self.ids[name] for key, name in CARD_NAMES.items() if name in self.ids}
        self.own_ids = frozenset(self.opening.players[0].deck)
        self.tracker = UnitIdTracker()

    def card(self, key):
        if key is None:
            return EMPTY_CARD
        try:
            return self.keys[base_key(key)]
        except KeyError as exc:
            raise ValueError(f"Carta desconhecida na observação: {key}") from exc

    def position(self, point):
        x, y = point
        return min(round(x * self.scale), 18 * self.scale - 1), min(
            round(y * self.scale), 32 * self.scale - 1,
        )

    def legal_actions(self, state, can_act):
        """Use public placement rules without the engine's relocation previews."""
        parser = self.env.action_parser
        oracle = parser.oracle
        mask = np.zeros(parser.n_actions, dtype=np.int8)
        mask[0] = 1
        if not can_act or state.tick < oracle.rules.deploy_lockout_ticks:
            return mask
        if oracle.rules.plays_stopped(state):
            return mask
        player = state.players[0]
        cells = parser.nx * parser.ny
        for slot, card_id in enumerate(player.hand):
            if card_id == EMPTY_CARD:
                continue
            card = oracle.cards[card_id]
            if player.elixir_milli < (1000 * card.elixir):
                continue
            grid = oracle.point_grid(state, 0, card, parser.pitch_div, moves=False)
            if card.card_kind == "BUILDING":
                grid = grid.copy()
                px, py = oracle.points(parser.pitch_div)
                clearance = card.radius + self.scale // 2
                for unit in state.entities:
                    if unit.kind == EntityKind.TROOP:
                        continue
                    left, bottom, right, top = unit.footprint or (
                        unit.x - unit.radius, unit.y - unit.radius,
                        unit.x + unit.radius, unit.y + unit.radius,
                    )
                    overlaps = (
                        (px >= left - clearance) & (px <= right + clearance)
                        & (py >= bottom - clearance) & (py <= top + clearance)
                    )
                    grid &= ~overlaps
            mask[1 + slot * cells:1 + (slot + 1) * cells] = grid.reshape(-1)
        return mask

    def update(self, message):
        frame = msgspec.convert(message, type=Frame)
        starting = self.tick is None or frame.reset
        previous = 0 if starting else self.tick
        if starting and frame.tick != 0:
            raise ValueError("A primeira observação de uma partida deve ter tick 0.")
        if not starting and frame.tick <= previous:
            raise ValueError("O relógio da partida deve avançar.")
        minute = 60_000 // self.opening.tick_ms
        before_regen = self.opening.regular_ticks - minute
        rate = 1 if frame.tick < before_regen else 2
        if frame.tick >= self.opening.regular_ticks + minute:
            rate = 3

        hand = [self.card(key) for key in frame.hand]
        following = self.card(frame.next_card)
        if any(card not in self.own_ids and card != EMPTY_CARD for card in (*hand, following)):
            raise ValueError("A mão observada não corresponde ao deck do checkpoint.")
        views = [frame.own_towers, frame.opponent_towers]
        if any(hp > maximum for view in views for hp, maximum in zip(view.hp, view.max_hp)):
            raise ValueError("A vida observada de uma torre excede sua vida máxima.")
        own, opponent = [
            replace(base, tower_hp=list(view.hp), tower_max_hp=list(view.max_hp),
                    crowns=view.crowns, king_active=view.king_active)
            for base, view in zip(self.opening.players, views, strict=True)
        ]
        own = replace(own, elixir_milli=frame.own_elixir_milli, hand=hand,
                      next_card=following, hand_costs=[])
        opponent = replace(opponent, elixir_milli=0, hand=[], next_card=EMPTY_CARD, deck=[],
                           forms=[], hand_costs=[], abilities=[], evo=[], pending=[])

        entities = [replace(tower, hp=views[tower.team].hp[tower.tower_slot],
                            max_hp=views[tower.team].max_hp[tower.tower_slot])
                    for tower in self.opening.entities
                    if views[tower.team].hp[tower.tower_slot] > 0]
        first_uid = 1 + max(tower.uid for tower in self.opening.entities)
        if starting:
            self.tracker.reset(first_uid)
        track_inputs = [
            (unit.side, self.card(unit.card), unit.kind, unit.position[0], unit.position[1])
            for unit in frame.units
        ]
        unit_uids = self.tracker.assign(track_inputs)
        for uid, unit in zip(unit_uids, frame.units, strict=True):
            if unit.hp > unit.max_hp:
                raise ValueError("A vida observada de uma unidade excede sua vida máxima.")
            x, y = self.position(unit.position)
            flags = {"basic": 0, "evolved": STATUS_EVOLVED, "hero": STATUS_HERO}
            radius_px = round(unit.radius * self.scale)
            footprint = None
            if unit.footprint is not None:
                left, right = unit.footprint
                if left[0] > right[0] or left[1] > right[1]:
                    raise ValueError("Os cantos da área ocupada pela construção estão invertidos.")
                footprint = (*self.position(left), *self.position(right))
            elif unit.kind == "building":
                # Screen readers omit explicit placement boxes; approximate from center + radius.
                footprint = (x - radius_px, y - radius_px, x + radius_px, y + radius_px)
            card_name = CARD_NAMES.get(base_key(unit.card)) if unit.card is not None else None
            unit_type = self.unit_type_by_name.get(card_name, 0) if card_name else 0
            entities.append(EntityState(
                uid=uid, team=int(unit.side == "opponent"), card_id=self.card(unit.card),
                kind=EntityKind.BUILDING if unit.kind == "building" else EntityKind.TROOP,
                tower_slot=-1, x=x, y=y, hp=unit.hp, max_hp=unit.max_hp,
                radius=radius_px, flying=unit.kind == "air",
                deploy_ticks=int(unit.deploying), stun_ticks=int(unit.stunned),
                status_flags=flags[unit.form], footprint=footprint,
                charge=0, unit_type=unit_type,
            ))
        spells = []
        for spell in frame.spells:
            if spell.side == "opponent" and spell.own_aim is not None:
                raise ValueError("O campo own_aim pertence apenas aos nossos feitiços.")
            x, y = self.position(spell.position)
            aim_x, aim_y = self.position(spell.own_aim or spell.position)
            spells.append(SpellState(int(spell.side == "opponent"), self.card(spell.card),
                                     SpellMotion.AREA, x, y, aim_x, aim_y, 0, 0, 0, 0))
        plays, presses = [[], []], [[], []]
        for event in sorted(frame.events, key=lambda item: item.tick):
            if not previous <= event.tick < frame.tick:
                raise ValueError("Envie só eventos novos, do tick anterior até antes do atual.")
            if (event.card is None) == (event.ability_cost is None):
                raise ValueError("Um evento contém uma carta ou um custo de habilidade.")
            team = int(event.side == "opponent")
            if event.card is not None:
                plays[team].append((event.tick, self.card(event.card)))
            else:
                presses[team].append((event.tick, event.ability_cost))

        state = replace(
            self.opening, tick=frame.tick, elixir_rate=rate,
            overtime=frame.tick >= self.opening.regular_ticks,
            players=[own, opponent], entities=entities, spells=spells,
        )
        mask = self.legal_actions(state, frame.can_act)
        builder = self.env.obs_builder
        if starting:
            builder.reset(state)
        builder.plays, builder.presses_paid = plays, presses
        policy_input = builder.build(state, 0, mask)
        self.tick = frame.tick
        return policy_input

    def public_summary(self):
        memory = self.env.obs_builder.memory[0]
        seen = [self.name_by_id[int(cid)] for cid in np.flatnonzero(memory.foe_seen) if int(cid) in self.name_by_id]
        queue = [self.name_by_id.get(int(cid), "?") for cid in memory.enemy_queue() if int(cid) != EMPTY_CARD]
        return {
            "tick": int(self.tick or 0),
            "own_elixir": round(memory.own_elixir_milli() / 1000.0, 2),
            "enemy_elixir": round(memory.enemy_elixir_milli() / 1000.0, 2),
            "enemy_queue": queue,
            "enemy_seen": seen,
        }

    def close(self):
        self.env.close()
