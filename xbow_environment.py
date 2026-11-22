"""RoyaleSim observations, legal actions, and X-Bow training rewards."""

from msgspec.structs import replace
import numpy as np
from royalegym import ClashParallelEnv, Reveal, RustEngine, SpatialObsBuilder, TileActionParser
from royalegym.obs import SAME_UNIT_ALIASES, build_vector
from royalegym.protocol import (
    DeployStatus, EMPTY_CARD, HAND_SIZE, STATUS_INVISIBLE, STATUS_UNDERGROUND,
    ability_row, status_of,
)
from royalegym.reward import (
    CombinedReward, ElixirTradeReward, RewardFunction, TowerHPReward, WinLossReward,
)
from royalegym.state_mutator import DefaultStateMutator
from royaleimitate.replays import CARD_NAMES

from xbow_replays import DECKS

RIVAL_DECKS = {
    "hog_26": (
        "hog-rider", "musketeer", "cannon", "fireball", "the-log",
        "skeletons", "ice-golem", "ice-spirit",
    ),
    "royal_giant": (
        "royal-giant", "fisherman", "hunter", "royal-ghost", "skeletons",
        "electro-spirit", "lightning", "the-log",
    ),
    "splashyard": (
        "graveyard", "poison", "baby-dragon", "ice-wizard", "knight",
        "tombstone", "barbarian-barrel", "tornado",
    ),
    "log_bait": (
        "goblin-barrel", "princess", "goblin-gang", "knight", "inferno-tower",
        "ice-spirit", "rocket", "the-log",
    ),
    "golem_beatdown": (
        "golem", "night-witch", "baby-dragon", "lumberjack", "mega-minion",
        "tornado", "lightning", "barbarian-barrel",
    ),
    "pekka_bs": (
        "pekka", "battle-ram", "bandit", "royal-ghost", "electro-wizard",
        "magic-archer", "poison", "zap",
    ),
}
OPENING_ELIXIR_MILLI = 6000  # RoyaleSim's standard full-match setup.
PPO_GAMMA = 0.999
JOINT_REWARD_PROFILES = ("legacy", "potential")
OBS_PROFILES = ("v2", "live", "v3")
DEFAULT_OBS_PROFILE = "v3"
ID_PLANE_KEYS = ("card_ids", "spell_ids", "unit_ids")
ID_PLANE_CHANNELS = 8  # 2 card_ids + 4 spell_ids + 2 unit_ids
SPELL_AIM_AFTER_TICKS = 10
ELIXIR_TRADE_WEIGHT = 0.15
XBOW_LOCK_WEIGHT = 0.04


def pack_id_planes(obs):
    """Pack uint8 identity planes (card_ids, spell_ids, unit_ids) into (8, 32, 18), or None."""
    if not all(key in obs for key in ID_PLANE_KEYS):
        return None
    return np.concatenate([obs[key] for key in ID_PLANE_KEYS], axis=0).astype(np.uint8, copy=False)


def obs_profile_for_architecture(architecture: int, obs_profile: str | None = None) -> str:
    if obs_profile in OBS_PROFILES:
        return obs_profile
    return "v3" if architecture >= 3 else "v2"


class XBowLockReward(RewardFunction):
    def bind(self, engine):
        self.xbow_id = next(card.card_id for card in engine.cards()
                            if card.name == CARD_NAMES["x-bow"])

    def locks(self, state, team):
        towers = {
            unit.uid for unit in state.entities
            if unit.team != team and unit.tower_slot >= 0 and unit.hp > 0
        }
        return {
            (unit.uid, unit.target_uid) for unit in state.entities
            if unit.team == team and unit.card_id == self.xbow_id and unit.hp > 0
            and unit.target_uid in towers
        }

    def new_locks(self, team, prev, state):
        return len(self.locks(state, team) - self.locks(prev, team))

    def get_reward(self, team, prev, state, results):
        return float(self.new_locks(team, prev, state))


class PublicSpatialBuilder(SpatialObsBuilder):
    """Count the enemy's elixir and cycle from accepted, public actions."""

    def reset(self, state):
        public_start = [
            replace(player, elixir_milli=OPENING_ELIXIR_MILLI, hand=[], next_card=EMPTY_CARD,
                    abilities=[], pending=[], deck=[], forms=[])
            for player in state.players
        ]
        super().reset(replace(state, players=public_start))
        for team, memory in self.memory.items():
            own = state.players[team]
            memory.start(
                state.tick, own.elixir_milli, OPENING_ELIXIR_MILLI, own.hand, own.next_card,
            )
        self.plays = [[], []]
        self.presses_paid = [[], []]

    def record_public_actions(self, before, results):
        self.plays = [[], []]
        self.presses_paid = [[], []]
        for result in results:
            if result.status != DeployStatus.OK:
                continue
            if result.hand_slot < HAND_SIZE:
                self.plays[result.team].append((result.tick, result.card_id))
            else:
                button = result.hand_slot - HAND_SIZE
                row = before.players[result.team].abilities[button]
                self.presses_paid[result.team].append((result.tick, ability_row(row).cost))

    def build(self, state, team, action_mask):
        visible = [
            unit for unit in state.entities if unit.team == team or not (
                (status_of(unit) or 0) & (STATUS_INVISIBLE | STATUS_UNDERGROUND)
            )
        ]
        return super().build(replace(state, entities=visible), team, action_mask)

    def _vector(self, state, team):
        memory = self.memory[team]
        own = state.players[team]
        if state.tick > memory.tick:
            memory.advance(
                state.tick, state.regular_ticks, state.overtime,
                self.plays[team], self.plays[1 - team],
                own_presses=self.presses_paid[team], foe_presses=self.presses_paid[1 - team],
            )
            memory.show_own_hand(own.hand, own.next_card)
        extra = {"enemy_card_cycle": True} if self.enemy_card_cycle else {}
        return build_vector(
            state, team, self.cards, self.max_mana, self.reveal, memory,
            enemy_last_card=self.enemy_last_card, evolutions=self.evolutions,
            evolution_progress=self.evolution_progress, button_slots=self.button_slots,
            enemy_queue=self.enemy_queue, **extra,
        )


class PublicBattleEnv(ClashParallelEnv):
    def _advance(self, commands):
        before = self.battle_state
        results = super()._advance(commands)
        self.obs_builder.record_public_actions(before, results)
        return results


def all_rival_decks(*, max_discovered: int = 10) -> dict[str, tuple[str, ...]]:
    """Return the 6 built-in meta decks plus up to `max_discovered` live/replay discovered decks."""
    decks = dict(RIVAL_DECKS)
    if max_discovered > 0:
        from xbow_deck_discovery import DiscoveredDeckStore
        for row in DiscoveredDeckStore().active_rival_decks(max_decks=max_discovered):
            cards = tuple(row.get("cards") or ())
            if len(cards) == 8:
                decks[str(row["id"])] = cards
    return decks


def deck_card_ids(deck, card_ids):
    if isinstance(deck, (list, tuple)):
        if len(deck) != 8:
            raise ValueError(f"Deck customizado precisa ter exatamente 8 cartas (recebido {len(deck)}).")
        return [card_ids[CARD_NAMES[key]] for key in deck]
    available = DECKS | RIVAL_DECKS
    if deck in available:
        return [card_ids[CARD_NAMES[key]] for key in available[deck]]
    from xbow_deck_discovery import load_discovered_decks
    discovered = load_discovered_decks()
    if deck in discovered:
        return [card_ids[CARD_NAMES[key]] for key in discovered[deck]]
    raise KeyError(f"Deck desconhecido: {deck}")


def battle_reward(profile):
    if profile == "potential":
        from royalelearn.rewards import default_potential_reward

        reward = default_potential_reward()
        reward.set_gamma(PPO_GAMMA)
        return reward
    if profile != "legacy":
        raise ValueError(f"Perfil de recompensa desconhecido: {profile}")
    return CombinedReward([
        (WinLossReward(), 1.0),
        (TowerHPReward(), 0.7),
        (ElixirTradeReward(), ELIXIR_TRADE_WEIGHT),
        (XBowLockReward(), XBOW_LOCK_WEIGHT),
    ])


def build_obs_builder(obs_profile=DEFAULT_OBS_PROFILE):
    if obs_profile == "v2":
        return PublicSpatialBuilder(
            reveal=Reveal(), enemy_queue=True, evolutions=True,
            evolution_progress=True, heroes=True, button_index=True,
        )
    if obs_profile == "live":
        # Sim-to-Real profile: every feature readable from a real screen reader + tracker,
        # without simulator-only internal flags (target_uid, windup, charge, hidden buffs).
        return PublicSpatialBuilder(
            reveal=Reveal(),
            card_identity=True,
            unit_identity=True,
            unit_aliases=SAME_UNIT_ALIASES,
            spell_identity=True,
            spell_aim_after_ticks=SPELL_AIM_AFTER_TICKS,
            unit_velocity=True,
            enemy_queue=True,
            enemy_card_cycle=True,
            evolutions=True,
            evolution_progress=True,
            heroes=True,
            button_index=True,
        )
    if obs_profile == "v3":
        return PublicSpatialBuilder(
            reveal=Reveal(),
            card_identity=True,
            unit_identity=True,
            unit_aliases=SAME_UNIT_ALIASES,
            spell_identity=True,
            spell_aim_after_ticks=SPELL_AIM_AFTER_TICKS,
            unit_status=True,
            unit_actions=True,
            unit_velocity=True,
            enemy_queue=True,
            enemy_card_cycle=True,
            evolutions=True,
            evolution_progress=True,
            heroes=True,
            button_index=True,
        )
    raise ValueError(f"Perfil de observacao desconhecido: {obs_profile}")


def battle_env(deck, rival_deck=None, *, decision_ms=500, reward_profile="legacy",
               obs_profile=DEFAULT_OBS_PROFILE):
    reward = battle_reward(reward_profile)
    engine = RustEngine()
    card_ids = {card.name: card.card_id for card in engine.cards()}
    decks = [deck_card_ids(name, card_ids) for name in (deck, rival_deck or deck)]
    return PublicBattleEnv(
        engine,
        state_mutator=DefaultStateMutator(decks=decks),
        decision_ms=decision_ms,
        action_parser=TileActionParser(ability_buttons=True),
        obs_builder=build_obs_builder(obs_profile),
        reward_fn=reward,
    )
