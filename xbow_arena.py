"""Share a bounded set of current training frames with the dashboard."""

from collections import deque
from contextlib import suppress
import json
import os
from pathlib import Path
from time import monotonic
from uuid import uuid4


STAGES = ("replays", "bc", "ppo", "evaluation")
LIVE_FRAMES = 8
PUBLISH_INTERVAL = 0.12


def live_data(folder, stage):
    boards = []
    for path in folder.glob(f"{stage}-*.json"):
        try:
            boards.append(json.loads(path.read_bytes()))
        except (OSError, ValueError):
            continue
    return sorted(boards, key=lambda board: board["header"].get("number", 0))


def arena_enabled():
    return os.environ.get("XBOW_ARENA") == "1"


class ArenaCapture:
    def __init__(self, output, stage, env, **metadata):
        folder = Path(os.environ.get("XBOW_ARENA_DIR", output / "arena-live"))
        board_id = uuid4().hex
        self.path = folder / f"{stage}-{board_id}.json"
        self.temporary = self.path.with_suffix(".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.parser = env.action_parser
        self.cards = {card.card_id: card.name for card in env.engine.cards()}
        self.units = dict(enumerate(env.engine.unit_types()))
        self.header = {"id": board_id, "stage": stage, "width": self.parser.nx,
                       "height": self.parser.ny, **metadata}
        self.frames = deque(maxlen=LIVE_FRAMES)
        self.seq = 0
        self.last_ms = None
        self.last_publish = None
        self.example = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, error_type, error, traceback):
        self.close()

    def close(self):
        for path in (self.path, self.temporary):
            with suppress(PermissionError):
                path.unlink(missing_ok=True)
        self.frames.clear()

    def frame(self, board, **data):
        self.frames.append({"seq": self.seq, **board, **data})
        self.seq += 1
        now = monotonic()
        if self.last_publish is not None and now - self.last_publish < PUBLISH_INTERVAL:
            return
        snapshot = {"header": self.header, "frames": list(self.frames)}
        with suppress(PermissionError, FileNotFoundError):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.temporary.write_text(
                json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")), encoding="utf-8",
            )
            self.temporary.replace(self.path)
            self.last_publish = now

    def action(self, action, seat, hand):
        if action == 0:
            return {"kind": "wait"}
        if action >= self.parser.n_tile_actions:
            return {"kind": "ability", "card": "Habilidade"}
        slot, x, y = self.parser.decode(action)
        if seat == 1:
            x, y = self.parser.nx - 1 - x, self.parser.ny - 1 - y
        return {"kind": "card", "card": hand[slot], "x": x + .5, "y": y + .5}

    def observe(self, state, actions=(0, 0), *, example_index=None):
        ms = state.tick * state.tick_ms
        if not (any(actions) or self.last_ms is None or ms - self.last_ms >= 500):
            return
        pitch = self.parser.pitch
        sides = [
            {"hp": player.tower_hp, "elixir": round(player.elixir_milli / 1000, 1),
             "hand": [self.cards.get(card, "-") for card in player.hand]}
            for player in state.players
        ]
        pieces = [
            [unit.uid, unit.team, unit.kind, round(unit.x / pitch, 2), round(unit.y / pitch, 2),
             unit.hp, unit.max_hp, unit.target_uid,
             self.units.get(unit.unit_type) or self.cards.get(unit.card_id, "")]
            for unit in state.entities if unit.hp > 0
        ]
        moves = [{"seat": seat, **self.action(action, seat, sides[seat]["hand"])}
                 for seat, action in enumerate(actions) if action]
        board = {"ms": ms, "pieces": pieces, "sides": sides, "moves": moves}
        self.frame(board)
        self.last_ms = ms
        if example_index is not None and self.example is None:
            self.example = (example_index, board)
