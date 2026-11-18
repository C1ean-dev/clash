"""Index the AlphaClash manifest without requesting any video files."""

import json
import re
from threading import Lock
from urllib.parse import quote


ALPHACLASH = "https://huggingface.co/datasets/josefbednar/alphaclash-replays"
PAGE_SIZE = 20
STEM = re.compile(r"\d{8}-\d{6}_[01]{16}")
SHARD = re.compile(r"shard-\d{4}")


class VideoCatalog:
    def __init__(self, path):
        self.path = path
        self.lock = Lock()
        self.version = None
        self.records = ()

    def read(self):
        with self.lock:
            try:
                version = self.path.stat().st_mtime_ns
            except FileNotFoundError:
                return None
            if version == self.version:
                return self.records

            from xbow_replays import DECK_BY_CARDS

            records = []
            with self.path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    stem, shard = row.get("stem", ""), row.get("shard", "")
                    if not STEM.fullmatch(stem) or not SHARD.fullmatch(shard):
                        continue
                    players = row.get("decks") or {}
                    cards = [[card.replace("_", "-") for card in players.get(f"player{side}", [])
                              if isinstance(card, str)] for side in (0, 1)]
                    decks = [DECK_BY_CARDS.get(frozenset(hand)) if len(hand) == 8 else None for hand in cards]
                    records.append({
                        "id": stem, "shard": shard, "arena": row.get("arena"), "cards": cards,
                        "decks": decks, "winner": row.get("winner"), "crowns": row.get("crowns"),
                        "seconds": row.get("duration_s"), "speed": row.get("speed_multiplier"),
                    })
            self.records = tuple(sorted(records, key=lambda row: row["id"], reverse=True))
            self.version = version
            return self.records

    def page(self, page, deck, wins, search):
        records = self.read()
        if records is None:
            return {"available": False, "total": 0, "found": 0, "page": 1, "pages": 1, "rows": []}
        matches = []
        for row in records:
            if deck:
                winner = row["winner"]
                if wins and (winner not in (0, 1) or row["decks"][winner] != deck):
                    continue
                if deck not in row["decks"]:
                    continue
            if search and search not in f"{row['id']} {row['arena']} {' '.join(sum(row['cards'], []))}":
                continue
            matches.append(row)
        pages = max(1, (len(matches) + PAGE_SIZE - 1) // PAGE_SIZE)
        page = min(max(page, 1), pages)
        start = (page - 1) * PAGE_SIZE
        rows = []
        for row in matches[start:start + PAGE_SIZE]:
            path = f"replays/{quote(row['shard'])}/{quote(row['id'])}.mp4"
            rows.append({**row, "video": f"{ALPHACLASH}/resolve/main/{path}", "source": f"{ALPHACLASH}/blob/main/{path}"})
        return {"available": True, "total": len(records), "found": len(matches), "page": page,
                "pages": pages, "rows": rows}
