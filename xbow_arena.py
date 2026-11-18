"""Record and read small, local clips of the states used by training."""

import json
import os
from time import time_ns
from uuid import uuid4


STAGES = ("replays", "bc", "ppo", "evaluation")
KEEP_CLIPS = 3


def clip_files(folder, stage):
    return sorted(folder.glob(f"{stage}-*.jsonl"), reverse=True)


def clip_data(folder, stage, clip_id, known, after):
    clips = []
    selected = None
    for path in clip_files(folder, stage):
        try:
            with path.open(encoding="utf-8") as stream:
                header = json.loads(stream.readline())
                clips.append(header)
                if selected is not None or (clip_id and header["id"] != clip_id):
                    continue
                frames, result = [], None
                offset = after if known == header["id"] else -1
                for line in stream:
                    if not line.endswith("\n"):
                        break
                    record = json.loads(line)
                    if "result" in record:
                        result = record["result"]
                    elif record["seq"] > offset:
                        frames.append(record)
                selected = {"header": header, "frames": frames, "result": result}
        except (OSError, ValueError):
            continue
    return {"clips": clips, "clip": selected}


def arena_enabled():
    return os.environ.get("XBOW_ARENA") == "1"


class ArenaCapture:
    def __init__(self, output, stage, env, **metadata):
        self.folder = output / "arena"
        self.stage = stage
        self.parser = env.action_parser
        self.cards = {card.card_id: card.name for card in env.engine.cards()}
        self.units = dict(enumerate(env.engine.unit_types()))
        self.header = {"id": uuid4().hex, "stage": stage, "width": self.parser.nx,
                       "height": self.parser.ny, **metadata}
        self.seq = 0
        self.last_ms = None
        self.example = None
        self.stream = None

    def __enter__(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        for path in clip_files(self.folder, self.stage)[KEEP_CLIPS - 1:]:
            try:
                path.unlink(missing_ok=True)
            except PermissionError:
                pass
        path = self.folder / f"{self.stage}-{time_ns():020d}-{self.header['id']}.jsonl"
        self.stream = path.open("w", encoding="utf-8", buffering=1)
        self.write(self.header)
        return self

    def __exit__(self, error_type, error, traceback):
        if error_type:
            self.finish({"status": "interrupted"})
        self.stream.close()

    def write(self, record):
        self.stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def frame(self, board, **data):
        self.write({"seq": self.seq, **board, **data})
        self.seq += 1

    def finish(self, result):
        self.write({"result": result})

    def action(self, action, seat, hand):
        if action == 0:
            return {"kind": "wait"}
        if action >= self.parser.n_tile_actions:
            return {"kind": "ability", "card": "Habilidade"}
        slot, x, y = self.parser.decode(action)
        if seat == 1:
            x, y = self.parser.nx - 1 - x, self.parser.ny - 1 - y
        return {"kind": "card", "card": hand[slot], "x": x + .5, "y": y + .5}

    def observe(self, state, actions=(0, 0), *, example_index=None, force=False):
        ms = state.tick * state.tick_ms
        if not (force or any(actions) or self.last_ms is None or ms - self.last_ms >= 500):
            return
        pitch = self.parser.pitch
        sides = [
            {"hp": player.tower_hp, "elixir": round(player.elixir_milli / 1000, 1),
             "hand": [self.cards.get(card, "—") for card in player.hand]}
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
