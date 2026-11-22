"""Dynamic 119-card support & automatic opponent deck discovery from live matches and replays.

Whenever a real opponent plays a deck not already in the built-in roster, this module:
  1. Extracts verified enemy cards (ignoring generic HP-bar fallbacks)
  2. Completes partial 4-7 card observations with realistic role-balanced fill cards (or stores
     exact 8-card decks as `verified`)
  3. Persists deduplicated decks + live win/loss records in `runs/discovered_decks.json`
  4. Feeds the hardest/most relevant discovered decks automatically into `xbow_environment.py`,
     `xbow_hybrid_loop.py`, `xbow_joint_training.py` (Rival Live), and the Recommended Pipeline
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from royaleimitate.replays import CARD_NAMES, base_key

from xbow_live_overlay import infer_match_result
from xbow_replays import DECKS

DEFAULT_DISCOVERED_DECKS_PATH = Path("runs/discovered_decks.json")

# All 119 simulator-verified Clash Royale card slugs.
ALL_SIMULATOR_CARDS: tuple[str, ...] = tuple(sorted(CARD_NAMES.keys()))
ALL_SIMULATOR_CARDS_SET: frozenset[str] = frozenset(ALL_SIMULATOR_CARDS)

# Role buckets for realistic completion of partial 4..7-card live observations and playstyle inference.
SMALL_SPELLS: tuple[str, ...] = ("the-log", "zap", "arrows", "barbarian-barrel", "giant-snowball")
BIG_SPELLS: tuple[str, ...] = ("fireball", "poison", "lightning", "rocket", "earthquake")
BUILDINGS: tuple[str, ...] = ("cannon", "tesla", "tombstone", "goblin-cage", "inferno-tower", "bomb-tower")
CYCLE_FILLERS: tuple[str, ...] = ("skeletons", "ice-spirit", "electro-spirit", "knight", "archers", "goblins", "bats", "ice-golem")

BEATDOWN_TANKS: frozenset[str] = frozenset({
    "golem", "lava-hound", "electro-giant", "giant", "royal-giant", "giant-skeleton", "goblin-giant", "sparky",
})
CONTROL_CARDS: frozenset[str] = frozenset({
    "graveyard", "x-bow", "mortar", "tornado", "ice-wizard", "tombstone", "inferno-tower", "bomb-tower",
})
PUSH_CARDS: frozenset[str] = frozenset({
    "hog-rider", "ram-rider", "battle-ram", "royal-hogs", "royal-recruits", "goblin-barrel",
    "wall-breakers", "balloon", "elite-barbarians", "mega-knight", "pekka", "bandit", "prince",
})


def canonical_card_slug(raw: str) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    slug = base_key(raw.strip().lower().replace("_", "-"))
    return slug if slug in ALL_SIMULATOR_CARDS_SET else None


def infer_play_style(cards: tuple[str, ...] | list[str]) -> str:
    """Choose the best scripted opponent style ('patient', 'defend', or 'push') for an 8-card deck."""
    cset = set(cards)
    if cset & BEATDOWN_TANKS:
        return "patient"
    if len(cset & CONTROL_CARDS) >= 2:
        return "defend"
    if cset & PUSH_CARDS:
        return "push"
    return "patient" if len(cset & set(BUILDINGS)) >= 1 else "push"


def complete_partial_deck(
    card_counts: Counter[str],
    *,
    min_unique_cards: int = 4,
) -> tuple[tuple[str, ...], str] | None:
    """Convert observed enemy card counts into a clean 8-card simulator deck + status ('verified'|'completed')."""
    cleaned: Counter[str] = Counter()
    for raw, cnt in card_counts.items():
        slug = canonical_card_slug(raw)
        if slug is not None and cnt > 0:
            cleaned[slug] += int(cnt)

    # Ignore matches that ONLY have generic HP-bar 'knight' fallback and fewer than min_unique_cards real cards.
    non_knight = {k: v for k, v in cleaned.items() if k != "knight"}
    if not non_knight:
        return None
    if len(cleaned) < min_unique_cards:
        return None

    # Sort by observation frequency (then alphabetically for determinism) and take up to 8 unique cards.
    ranked = [c for c, _ in sorted(cleaned.items(), key=lambda kv: (-kv[1], kv[0]))]
    chosen = ranked[:8]
    if len(chosen) == 8:
        return tuple(sorted(chosen)), "verified"

    chosen_set = set(chosen)
    # Fill missing 1..4 slots realistically: 1 small spell, 1 big spell, 1 building/cycle card.
    fill_groups = (SMALL_SPELLS, BIG_SPELLS, CYCLE_FILLERS, BUILDINGS)
    for group in fill_groups:
        if len(chosen) >= 8:
            break
        if any(c in chosen_set for c in group):
            continue
        for cand in group:
            if cand not in chosen_set and cand in ALL_SIMULATOR_CARDS_SET:
                chosen.append(cand)
                chosen_set.add(cand)
                break
    for cand in CYCLE_FILLERS + SMALL_SPELLS + BIG_SPELLS:
        if len(chosen) >= 8:
            break
        if cand not in chosen_set and cand in ALL_SIMULATOR_CARDS_SET:
            chosen.append(cand)
            chosen_set.add(cand)

    if len(chosen) != 8:
        return None
    return tuple(sorted(chosen)), "completed"


def deck_signature(cards: tuple[str, ...] | list[str]) -> str:
    ordered = sorted(str(c) for c in cards)
    return hashlib.sha1(",".join(ordered).encode("utf-8")).hexdigest()[:8]


def friendly_deck_label(cards: tuple[str, ...] | list[str]) -> str:
    """Create a readable short label highlighting headline win-condition / signature cards."""
    priority = list(BEATDOWN_TANKS) + list(PUSH_CARDS) + list(CONTROL_CARDS)
    ordered = sorted(cards, key=lambda c: (0 if c in priority else 1, c))
    head = [c.replace("-", " ").title() for c in ordered[:3]]
    return " · ".join(head)


def _builtin_deck_lookup() -> dict[frozenset[str], str]:
    # Lazy import RIVAL_DECKS to avoid circular import when xbow_environment imports us.
    from xbow_environment import RIVAL_DECKS
    table: dict[frozenset[str], str] = {}
    for name, cards in (DECKS | RIVAL_DECKS).items():
        table[frozenset(cards)] = name
    return table


class DiscoveredDeckStore:
    """Persistent JSON store of opponent decks discovered from live play or replay datasets."""

    def __init__(self, path: Path = DEFAULT_DISCOVERED_DECKS_PATH):
        self.path = Path(path)
        self.data = self._load()

    def _load(self) -> dict:
        base = {"version": 1, "decks": {}, "processed_files": [], "updated_at": None}
        if self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("decks"), dict):
                    base.update(loaded)
            except Exception:
                pass
        return base

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def register_deck(
        self,
        cards: tuple[str, ...] | list[str],
        *,
        status: str = "verified",
        source: str = "live",
        result: str | None = None,
        prefix: str = "live",
        persist: bool = True,
    ) -> dict | None:
        """Add or update an 8-card opponent deck; returns entry or None if invalid/built-in."""
        norm = tuple(sorted({c for c in (canonical_card_slug(x) for x in cards) if c is not None}))
        if len(norm) != 8:
            return None
        builtins = _builtin_deck_lookup()
        if frozenset(norm) in builtins:
            return None

        deck_id = f"{prefix}_{deck_signature(norm)}"
        # Deduplicate regardless of prefix if the exact 8-card set already exists under another id.
        existing_id = next(
            (did for did, row in self.data["decks"].items() if frozenset(row.get("cards", ())) == frozenset(norm)),
            None,
        )
        if existing_id is not None:
            deck_id = existing_id

        now = datetime.now(timezone.utc).isoformat()
        entry = self.data["decks"].get(deck_id)
        if entry is None:
            entry = {
                "id": deck_id,
                "label": friendly_deck_label(norm),
                "cards": list(norm),
                "status": status,
                "source": source,
                "play_style": infer_play_style(norm),
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "seen_count": 0,
                "first_seen": now,
                "last_seen": now,
            }
            self.data["decks"][deck_id] = entry
        else:
            if status == "verified":
                entry["status"] = "verified"
            entry["last_seen"] = now

        entry["seen_count"] = int(entry.get("seen_count", 0)) + 1
        if result in ("win", "loss", "draw"):
            key = {"win": "wins", "loss": "losses", "draw": "draws"}[result]
            entry[key] = int(entry.get(key, 0)) + 1
        if persist:
            self.save()
        return entry

    def record_live_match(
        self,
        enemy_card_counts: Counter[str],
        result: str | None,
        *,
        min_unique_cards: int = 4,
    ) -> dict | None:
        built = complete_partial_deck(enemy_card_counts, min_unique_cards=min_unique_cards)
        if built is None:
            return None
        cards, status = built
        return self.register_deck(cards, status=status, source="live", result=result, prefix="live", persist=True)

    def sync_from_live_matches(
        self,
        matches_dir: Path = Path("runs/live_matches"),
        *,
        deck: str = "xbow_30",
        min_unique_cards: int = 4,
    ) -> dict:
        """Scan recorded live JSONL matches and catalog all newly discovered opponent decks."""
        matches_dir = Path(matches_dir)
        if not matches_dir.is_dir():
            return {"scanned": 0, "added_or_updated": 0, "total_decks": len(self.data["decks"])}
        processed = set(self.data.get("processed_files") or [])
        scanned = 0
        updated = 0
        for path in sorted(matches_dir.glob("*.jsonl")):
            if path.name in processed:
                continue
            try:
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            except OSError:
                continue
            steps = [r for r in rows if r.get("kind") == "step"]
            if len(steps) < 2:
                continue
            header = next((r for r in rows if r.get("kind") == "header"), {})
            if header.get("deck") and header["deck"] != deck:
                continue
            footer = next((r for r in reversed(rows) if r.get("kind") == "footer"), {})
            result = footer.get("result") or infer_match_result(steps[-1]["frame"])
            scanned += 1
            processed.add(path.name)

            counts: Counter[str] = Counter()
            for step in steps:
                frame = step.get("frame") or {}
                for u in frame.get("units") or ():
                    if u.get("side") == "opponent" and u.get("card"):
                        counts[str(u["card"])] += 1
                for ev in frame.get("events") or ():
                    if ev.get("side") == "opponent" and ev.get("card"):
                        counts[str(ev["card"])] += 1
            built = complete_partial_deck(counts, min_unique_cards=min_unique_cards)
            if built is not None:
                cards, status = built
                if self.register_deck(cards, status=status, source="live", result=result, prefix="live", persist=False):
                    updated += 1
        self.data["processed_files"] = sorted(processed)
        if scanned > 0:
            self.save()
        return {"scanned": scanned, "added_or_updated": updated, "total_decks": len(self.data["decks"])}

    def import_from_replays(
        self,
        replays_dir: Path = Path("replays"),
        *,
        max_decks: int = 20,
        max_parts: int = 6,
    ) -> dict:
        """Harvest the most popular real-world human 8-card decks from local IL_Replay parquet partitions."""
        import pyarrow.parquet as pq

        replays_dir = Path(replays_dir)
        parts = sorted(replays_dir.glob("part-*.parquet"))[:max(1, int(max_parts))]
        deck_freq: Counter[tuple[str, ...]] = Counter()
        for part in parts:
            try:
                pf = pq.ParquetFile(part)
                for batch in pf.iter_batches(columns=["payload_json"], batch_size=512):
                    for raw in batch.column(0).to_pylist():
                        try:
                            payload = json.loads(raw)
                            battle = payload.get("battle") or {}
                            for side in ("team", "opponent"):
                                players = (battle.get(side) or {}).get("players") or []
                                if len(players) != 1:
                                    continue
                                raw_cards = players[0].get("deck") or []
                                slugs = tuple(sorted({
                                    c for c in (canonical_card_slug(x.get("card_key", "")) for x in raw_cards)
                                    if c is not None
                                }))
                                if len(slugs) == 8:
                                    deck_freq[slugs] += 1
                        except Exception:
                            continue
            except Exception:
                continue

        builtins = _builtin_deck_lookup()
        added = 0
        for cards, _freq in deck_freq.most_common():
            if added >= int(max_decks):
                break
            if frozenset(cards) in builtins:
                continue
            entry = self.register_deck(cards, status="verified", source="replay_meta", prefix="meta", persist=False)
            if entry is not None:
                added += 1
        if added > 0:
            self.save()
        return {"parts_scanned": len(parts), "unique_decks_seen": len(deck_freq), "imported": added, "total_decks": len(self.data["decks"])}

    def active_rival_decks(self, *, max_decks: int = 10) -> list[dict]:
        """Return top discovered decks prioritized by lowest live win rate (hardest first) and recency."""
        rows = list(self.data["decks"].values())

        def _priority(row: dict) -> tuple:
            w, l, d = int(row.get("wins", 0)), int(row.get("losses", 0)), int(row.get("draws", 0))
            tot = w + l + d
            is_live = 0 if row.get("source") == "live" else 1
            # Shrunk win rate: decks where we lost live sort FIRST!
            shrunk_wr = (w + 0.5 * d + 1.0) / (tot + 2.0) if tot > 0 else 0.48
            return (is_live, shrunk_wr, -l, -int(row.get("seen_count", 0)), row.get("id", ""))

        rows.sort(key=_priority)
        return rows[:max(0, int(max_decks))]

    def pfsp_seeds(self, *, max_decks: int = 10) -> dict[str, float]:
        seeds: dict[str, float] = {}
        for row in self.active_rival_decks(max_decks=max_decks):
            w, l, d = int(row.get("wins", 0)), int(row.get("losses", 0)), int(row.get("draws", 0))
            tot = w + l + d
            if tot > 0:
                shrunk = (w + 0.5 * d + 1.0) / (tot + 2.0)
                seeds[row["id"]] = round(max(0.12, min(0.85, shrunk)), 4)
            else:
                seeds[row["id"]] = 0.42
        return seeds


def load_discovered_decks(path: Path = DEFAULT_DISCOVERED_DECKS_PATH) -> dict[str, tuple[str, ...]]:
    """Lightweight reader returning `{deck_id: (card1, ..., card8)}` for xbow_environment.py."""
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        decks = raw.get("decks") or {}
        out: dict[str, tuple[str, ...]] = {}
        for did, row in decks.items():
            cards = tuple(row.get("cards") or ())
            if len(cards) == 8 and all(c in ALL_SIMULATOR_CARDS_SET for c in cards):
                out[str(did)] = cards
        return out
    except Exception:
        return {}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_DISCOVERED_DECKS_PATH)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sync = sub.add_parser("sync-live", help="Extrair decks adversários das partidas em runs/live_matches/")
    p_sync.add_argument("--matches-dir", type=Path, default=Path("runs/live_matches"))
    p_sync.add_argument("--deck", choices=DECKS, default="xbow_30")

    p_imp = sub.add_parser("import-replays", help="Importar decks populares das partições locais em replays/")
    p_imp.add_argument("--replays-dir", type=Path, default=Path("replays"))
    p_imp.add_argument("--max-decks", type=int, default=20)
    p_imp.add_argument("--max-parts", type=int, default=6)

    sub.add_parser("list", help="Listar todos os decks descobertos e seus placares")

    args = parser.parse_args()
    store = DiscoveredDeckStore(args.store)
    if args.cmd == "sync-live":
        print(json.dumps(store.sync_from_live_matches(args.matches_dir, deck=args.deck), indent=2, ensure_ascii=False))
    elif args.cmd == "import-replays":
        print(json.dumps(store.import_from_replays(args.replays_dir, max_decks=args.max_decks, max_parts=args.max_parts), indent=2, ensure_ascii=False))
    elif args.cmd == "list":
        print(json.dumps({
            "total_simulator_cards": len(ALL_SIMULATOR_CARDS),
            "total_discovered_decks": len(store.data["decks"]),
            "active_top": store.active_rival_decks(max_decks=20),
        }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
