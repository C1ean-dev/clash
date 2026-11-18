"""RoyaleSim observations, legal actions, and X-Bow training rewards."""

from msgspec.structs import replace
from royalegym import ClashParallelEnv, Reveal, RustEngine, SpatialObsBuilder, TileActionParser
from royalegym.obs import build_vector
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
}
OPENING_ELIXIR_MILLI = 6000  # RoyaleSim's standard full-match setup.


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

    def get_reward(self, team, prev, state, results):
        return float(len(self.locks(state, team) - self.locks(prev, team)))


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
        return build_vector(
            state, team, self.cards, self.max_mana, self.reveal, memory,
            enemy_last_card=self.enemy_last_card, evolutions=self.evolutions,
            evolution_progress=self.evolution_progress, button_slots=self.button_slots,
            enemy_queue=self.enemy_queue,
        )


class PublicBattleEnv(ClashParallelEnv):
    def _advance(self, commands):
        before = self.battle_state
        results = super()._advance(commands)
        self.obs_builder.record_public_actions(before, results)
        return results


def battle_env(deck, rival_deck=None):
    available = DECKS | RIVAL_DECKS
    engine = RustEngine()
    card_ids = {card.name: card.card_id for card in engine.cards()}
    decks = [[card_ids[CARD_NAMES[key]] for key in available[name]]
             for name in (deck, rival_deck or deck)]
    return PublicBattleEnv(
        engine,
        state_mutator=DefaultStateMutator(decks=decks),
        decision_ms=500,
        action_parser=TileActionParser(ability_buttons=True),
        obs_builder=PublicSpatialBuilder(
            reveal=Reveal(), enemy_queue=True, evolutions=True,
            evolution_progress=True, heroes=True, button_index=True,
        ),
        reward_fn=CombinedReward([
            (WinLossReward(), 1.0),
            (TowerHPReward(), 0.7),
            (ElixirTradeReward(), 0.15),
            (XBowLockReward(), 0.04),
        ]),
    )
