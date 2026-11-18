"""Convert public cr-bot frame exports to the X-Bow policy feed."""

import argparse
import json
import math
import sys
from time import sleep
from urllib.parse import urlencode
from urllib.request import urlopen

from royalegym import RustEngine
from royaleimitate.replays import CARD_NAMES, base_key

from xbow_replays import DECKS


# cr-bot projects its cropped frame into the KataCR arena.
REFERENCE_WIDTH, REFERENCE_HEIGHT = 568, 896
GRID = (-0.9320463320, 72.5462235650, 569.2610038610, 879.9748640483)
TICKS_PER_SECOND = 20
DECISION_TICKS = 10
ALIASES = {"xbow": "x-bow", "log": "the-log", "archer": "archers",
           "skeleton": "skeletons", "minion": "minions"}


def card_key(label):
    if not isinstance(label, str):
        return None
    key = "-".join(label.lower().strip().replace("_", " ").split())
    if key in ("", "none", "unknown", "?"):
        return None
    root = base_key(key.removesuffix("-evolution"))
    return ALIASES.get(root, root)


def arena_position(center, crop):
    left, top, width, height = crop
    if width <= 0 or height <= 0:
        raise ValueError("O recorte da arena deve ter dimensões positivas.")
    x0, y0, x1, y1 = GRID
    x = (center[0] - left) * REFERENCE_WIDTH / width
    y = (center[1] - top) * REFERENCE_HEIGHT / height
    cell = (18 * (x - x0) / (x1 - x0), 32 * (y1 - y) / (y1 - y0))
    return cell if 0 <= cell[0] < 18 and 0 <= cell[1] < 32 else None


class CaptureBridge:
    def __init__(self, deck, own_max=(4824, 3052, 3052), opponent_max=(4824, 3052, 3052)):
        engine = RustEngine()
        infos = {card.name: card for card in engine.cards()}
        self.catalogue = {key: infos[name] for key, name in CARD_NAMES.items() if name in infos}
        self.scale = engine.arena().subtile
        self.deck = frozenset(DECKS[deck])
        self.own_max = tuple(own_max)
        self.opponent_max = tuple(opponent_max)
        self.unrecognized = set()
        self.reset()

    def reset(self):
        self.start = None
        self.last_tick = None
        self.pending = []

    def tower_hp(self, raw, enemy):
        if len(raw) != 3 or any(hp is None for hp in raw):
            raise ValueError("Há torres sem vida legível no quadro.")
        left, king, right = (round(hp) for hp in raw)
        maximum = self.opponent_max if enemy else self.own_max
        hp = (king, right, left) if enemy else (king, left, right)
        if any(value < 0 or value > limit for value, limit in zip(hp, maximum)):
            raise ValueError("A vida das torres excede o intervalo da calibração.")
        return {"hp": hp, "max_hp": maximum,
                "king_active": king < maximum[0] or left == 0 or right == 0}

    def detections(self, captured, state):
        units, spells = [], []
        raw = captured.get("detections")
        if raw is None:
            raw = state.get("ally_units", []) + state.get("enemy_units", [])
        for row in raw:
            label = row.get("class_name", row.get("label"))
            key = card_key(label)
            card = self.catalogue.get(key)
            side = {"ally": "own", "enemy": "opponent"}.get(row.get("team"))
            center = row.get("center_px")
            if card is None or side is None or center is None:
                continue
            position = arena_position(center, state["arena_px"])
            if position is None:
                continue
            projected = {"side": side, "card": key, "position": position}
            if card.card_kind == "SPELL":
                spells.append(projected)
                continue
            maximum = max(1, card.hitpoints)
            fraction = row.get("estimated_hp")
            if fraction is None:
                fraction = 1.0
            if fraction <= 0:
                continue
            projected.update(
                hp=min(maximum, max(1, round(fraction * maximum))), max_hp=maximum,
                kind="building" if card.card_kind == "BUILDING" else "air" if card.flying else "ground",
                radius=card.radius / self.scale,
                form="evolved" if label.lower().endswith(("-evolution", "-ev1")) else "basic",
            )
            units.append(projected)
        return units, spells

    def push(self, captured):
        if not captured["in_game"]:
            self.reset()
            return None
        timestamp = float(captured["timestamp_s"])
        if not math.isfinite(timestamp):
            raise ValueError("Hora de captura inválida.")
        for side, events in (("own", captured.get("own_actions", [])),
                             ("opponent", captured.get("enemy_plays", []))):
            for event in events:
                if side == "opponent" and not (
                    event.get("clock_confirmed") or event.get("frame_confirmed")
                ):
                    continue
                key = card_key(event.get("card"))
                if key is not None and key in self.catalogue:
                    self.pending.append((timestamp, side, key))
                else:
                    self.unrecognized.add(str(event.get("card")))
        state = captured["record"].get("visual_state")
        if not state:
            return None
        new_match = self.start is None
        if new_match and (state.get("overtime") or float(state.get("time_left_s") or 0) < 178):
            raise ValueError("A captura começou após a abertura da partida.")
        start = timestamp if new_match else self.start
        tick = round((timestamp - start) * TICKS_PER_SECOND)
        if not new_match and tick - self.last_tick < DECISION_TICKS:
            return None
        hand = [card_key(label) for label in state["hand"]]
        next_card = card_key(state["next_card"])
        if len(hand) != 4 or any(key is not None and key not in self.deck
                                for key in (*hand, next_card)):
            raise ValueError("O deck lido na captura não corresponde ao selecionado.")
        own = self.tower_hp(state["tower_hp_self"], enemy=False)
        enemy = self.tower_hp(state["tower_hp_enemy"], enemy=True)
        own["crowns"] = 3 if enemy["hp"][0] == 0 else enemy["hp"].count(0)
        enemy["crowns"] = 3 if own["hp"][0] == 0 else own["hp"].count(0)
        units, spells = self.detections(captured, state)
        elixir = float(state["elixir"])
        if not math.isfinite(elixir) or not 0 <= elixir <= 10:
            raise ValueError("Leitura de elixir próprio fora do intervalo.")
        events = []
        if tick:
            for confirmed, side, card in self.pending:
                when = round((confirmed - start) * TICKS_PER_SECOND)
                events.append({
                    "tick": max(self.last_tick, min(when, tick - 1)), "side": side, "card": card,
                })
            self.pending.clear()
        self.start, self.last_tick = start, tick
        return {
            "tick": tick, "reset": new_match, "hand": hand, "next_card": next_card,
            "own_elixir_milli": round(elixir * 1000), "own_towers": own, "opponent_towers": enemy,
            "units": units, "spells": spells, "events": events, "can_act": any(hand),
        }


def capture_frames(base_url):
    cursor = -1
    while True:
        query = urlencode({"since": cursor, "limit": 100})
        with urlopen(f"{base_url.rstrip('/')}/api/frames?{query}", timeout=5) as response:
            frames = json.load(response)["frames"]
        for frame in frames:
            cursor = max(cursor, frame["frame_index"])
            yield frame
        if len(frames) < 100:
            sleep(0.25)


def jsonl_frames():
    for line in sys.stdin:
        yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", required=True, choices=DECKS)
    parser.add_argument("--capture-url", help="Base do servidor cr-bot. Sem URL, lê JSONL do stdin.")
    parser.add_argument("--own-max", nargs=3, type=int, default=(4824, 3052, 3052),
                        metavar=("REI", "ESQ", "DIR"))
    parser.add_argument("--opponent-max", nargs=3, type=int, default=(4824, 3052, 3052),
                        metavar=("REI", "ESQ", "DIR"))
    args = parser.parse_args()
    if min(*args.own_max, *args.opponent_max) < 1:
        parser.error("A vida máxima das torres precisa ser positiva.")
    bridge = CaptureBridge(args.deck, args.own_max, args.opponent_max)
    source = capture_frames(args.capture_url) if args.capture_url else jsonl_frames()
    try:
        for captured in source:
            try:
                public = bridge.push(captured)
                if public is not None:
                    print(json.dumps(public), flush=True)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                print(f"Quadro ignorado: {exc}", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if bridge.unrecognized:
            print(f"Eventos com carta desconhecida: {', '.join(sorted(bridge.unrecognized))}",
                  file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
