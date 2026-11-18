"""Convert the published Cochon123 snapshot into the local replay schema."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import tarfile
from urllib.parse import parse_qs, urlsplit

import pyarrow as pa
import pyarrow.parquet as pq
from royaleimitate.replays import CARD_NAMES, base_key, match_plan

from xbow_replays import DECK_BY_CARDS, SIDES, WinningReplay, candidate_replays, parquet_parts, reconstruct


PART = "part-cochon-2026-11-01.parquet"
COLLECTION = "Cochon123/clash-royale-replays"
PERMALINK = re.compile(r"""href=["'](/replay\?[^"']+)["']""")
ABILITY_ICON = re.compile(r"/ability-([a-z0-9-]+)\.(?:png|webp)")
SEATS = {"t": 0, "o": 1, "blue": 0, "red": 1}
ALIASES = {"p-e-k-k-a": "pekka", "mini-p-e-k-k-a": "mini-pekka", "elite-barbs": "elite-barbarians"}


def card_key(value):
    slug = re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")
    base = base_key(slug)
    return ALIASES.get(base, base) + slug[len(base):]


def participant(tag):
    return tag.strip().lstrip("#").upper()


def match_details(html):
    # The archived replay's permalink carries the original seats and crowns.
    link = PERMALINK.search(html)
    if not link:
        raise ValueError("missing_result")
    query = parse_qs(urlsplit(unescape(link[1])).query)
    if not all(key in query for key in ("tag", "team_tags", "opponent_tags", "team_crowns", "opponent_crowns")):
        raise ValueError("missing_result")
    players = tuple(participant(query[f"{side}_tags"][0]) for side in SIDES)
    if any(not tag or "," in tag for tag in players):
        raise ValueError("not_one_on_one")
    crowns = tuple(int(query[f"{side}_crowns"][0]) for side in SIDES)
    if not all(0 <= value <= 3 for value in crowns) or crowns[0] == crowns[1]:
        raise ValueError("no_winning_side")
    return query["tag"][0], players, crowns


class ReplayEvents(HTMLParser):
    def __init__(self):
        super().__init__()
        self.timeline = Counter()
        self.events = []
        self.decks = [set(), set()]

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        seat = SEATS.get(attrs.get("data-s"))
        if seat is None:
            return
        slug = attrs.get("data-card") or ""
        if tag == "img" and slug and "data-t" not in attrs and slug != "_invalid":
            self.decks[seat].add(card_key(slug))
        elif tag == "img" and "replay_card" in (attrs.get("class") or "").split():
            tick = int(attrs["data-t"])
            if attrs.get("data-ability") == "1":
                icon = ABILITY_ICON.search(attrs.get("src") or "")
                self.events.append((tick, seat, None, None, None, card_key(icon[1]) if icon else None))
            elif slug and slug != "_invalid":
                self.timeline[tick, seat, base_key(card_key(slug))] += 1
        elif {"data-c", "data-t", "data-x", "data-y"} <= attrs.keys() and attrs["data-c"] != "_invalid":
            x, y = int(attrs["data-x"]), int(attrs["data-y"])
            if not (0 <= x < 18000 and 0 <= y < 32000):
                raise ValueError("coordinates_outside_arena")
            self.events.append((int(attrs["data-t"]), seat, card_key(attrs["data-c"]), x, y, None))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)


def normalized_deck(values):
    forms = defaultdict(set)
    for value in values:
        forms[base_key(value)].add(value)
    if len(forms) != 8:
        return None
    deck = []
    for base, variants in sorted(forms.items()):
        decorated = variants - {base}
        if len(decorated) > 1:
            return None
        deck.append(next(iter(decorated), base))
    return tuple(deck)


def bases(cards):
    return frozenset(base_key(card) for card in cards)


def metadata_index(stream):
    index = defaultdict(set)
    for line in stream:
        match = json.loads(line)
        if any(len(match.get(side) or ()) != 1 for side in SIDES):
            continue
        players = [match[side][0] for side in SIDES]
        players_tags = tuple(participant(player["tag"]) for player in players)
        crowns = tuple(player["crowns"] for player in players)
        decks = []
        for player in players:
            keys = []
            for card in player.get("cards") or []:
                suffix = "-hero" if card.get("is_hero") else "-ev1" if card.get("evolution_level") else ""
                if card.get("name"):
                    keys.append(card_key(card["name"]) + suffix)
            decks.append(normalized_deck(keys))
        if any(deck is None for deck in decks):
            continue
        time = (match.get("battle_time") or "")[:15]
        index[players_tags, crowns].add((tuple(decks), time))
        index[players_tags[::-1], crowns[::-1]].add((tuple(decks[::-1]), time))
    return index


def captured_before(captured_at):
    # Compare YYYYMMDDTHHMMSS; the public snapshot uses UTC in both sources.
    return captured_at.replace("-", "").replace(":", "")[:15]


def infer_decks(events, stats, matches, captured_at):
    observed = [set(cards) for cards in stats]
    for _, seat, slug, _, _, _ in events:
        if slug:
            observed[seat].add(slug)
    observed_keys = [bases(cards) for cards in observed]
    complete = tuple(normalized_deck(cards) for cards in observed)
    compatible = {
        decks for decks, played_at in matches
        if played_at and captured_at and played_at <= captured_at
        and all(seen <= bases(deck) for seen, deck in zip(observed_keys, decks, strict=True))
        and all(all(form == base_key(form) or form in deck for form in seen)
                for seen, deck in zip(observed, decks, strict=True))
    }
    if len(compatible) == 1:
        return compatible.pop()
    if all(complete):
        return complete
    raise ValueError("incomplete_deck")


def convert(raw, html, metadata):
    tag, player_tags, crowns = match_details(html)
    winner = int(crowns[1] > crowns[0])
    matches = metadata.get((player_tags, crowns), ())
    if ("x-bow" not in html or "tesla" not in html) and not any(
        bases(decks[winner]) in DECK_BY_CARDS for decks, _ in matches
    ):
        raise ValueError("other_deck")
    parser = ReplayEvents()
    parser.feed(html)
    parser.close()
    decks = infer_decks(parser.events, parser.decks, matches, captured_before(raw.get("captured_at") or ""))
    deck = DECK_BY_CARDS.get(bases(decks[winner]))
    if not deck:
        raise ValueError("other_deck")
    if not parser.timeline or parser.timeline != Counter(
        (tick, seat, base_key(slug)) for tick, seat, slug, _, _, _ in parser.events if slug
    ):
        raise ValueError("unmatched_action")

    result = "victory" if winner == 0 else "defeat"
    battle = {"result": result}
    for side, cards, score in zip(SIDES, decks, crowns, strict=True):
        battle[side] = {"players": [{"deck": [{"card_key": card} for card in cards]}], "crowns": score}
    events = []
    for index, (tick, seat, slug, x, y, ability) in enumerate(sorted(parser.events, key=lambda row: (row[0], row[1]))):
        event = {"replay_tick_20hz": tick, "source_index": index, "side": SIDES[seat]}
        if slug:
            event.update(kind="play_card", card_key=slug, coordinates={"native_world_units": {"x": x, "y": y}})
        else:
            event.update(kind="activate_ability", ability_source_candidates=[ability] if ability else [])
        events.append(event)
    public_tag = sha256(f"{COLLECTION}/{tag}".encode()).hexdigest()[:32]
    payload = {"battle": battle, "events": events,
               "source": {"collection": COLLECTION, "replay_tag": public_tag, "default_card_level": 11}}
    plan = match_plan(payload, frozenset(CARD_NAMES.values()))
    if isinstance(plan, str):
        raise ValueError(plan)
    return deck, public_tag, result, payload


def fingerprint(payload, winner):
    events = []
    for event in payload["events"]:
        side = event.get("side")
        if side not in SIDES:
            continue
        tick = event["replay_tick_20hz"]
        side = int(SIDES.index(side) != winner)
        if event["kind"] == "play_card":
            coord = event["coordinates"]["native_world_units"]
            x, y = coord["x"] // 1000, coord["y"] // 1000
            if winner == 1:
                x, y = 17 - x, 31 - y
            events.append((tick, side, base_key(event["card_key"]), x, y))
        elif event["kind"] == "activate_ability":
            events.append((tick, side, "ability", 0, 0))
    return sha256(json.dumps(sorted(events), separators=(",", ":")).encode()).digest()


def validate_candidates(candidates, seed=42):
    from xbow_environment import battle_env

    envs = {}
    kept, rejected = Counter(), Counter()
    rows = []
    try:
        for number, (deck, tag, result, payload) in enumerate(candidates, 1):
            if deck not in envs:
                envs[deck] = battle_env(deck)
            replay = WinningReplay(tag, int(result == "defeat"), payload)
            sample = reconstruct(envs[deck], replay, seed)
            if isinstance(sample, str):
                rejected[sample] += 1
            else:
                kept[deck] += 1
                rows.append({"replay_tag": tag, "result": result,
                             "payload_json": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))})
            if number % 50 == 0:
                print(f"Verified {number}/{len(candidates)} simulator replays.", flush=True)
    finally:
        for env in envs.values():
            env.close()
    return rows, kept, rejected


def import_archive(archive_path, destination):
    output = destination / PART
    known = set()
    for part in parquet_parts(destination):
        if part != output:
            for _, replay in candidate_replays(part):
                try:
                    known.add(fingerprint(replay.payload, replay.seat))
                except (KeyError, TypeError, ValueError, AttributeError):
                    continue
    print(f"Loaded {len(known)} existing replay signatures.", flush=True)

    skipped, seen = Counter(), set()
    candidates = []
    scanned = 0
    metadata = {}
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            if member.name == "data/metadata/battles.jsonl":
                with archive.extractfile(member) as data:
                    metadata = metadata_index(data)
                continue
            if not member.name.startswith("data/raw/") or not member.name.endswith(".json"):
                continue
            scanned += 1
            if member.size > 2_000_000:
                skipped["oversize"] += 1
                continue
            try:
                with archive.extractfile(member) as data:
                    raw = json.load(data)
                html = (raw.get("payload") or {}).get("html")
                if not isinstance(html, str) or "replay_card" not in html:
                    raise ValueError("unavailable_replay")
                deck, tag, result, payload = convert(raw, html, metadata)
                identity = fingerprint(payload, int(result == "defeat"))
                if tag in seen or identity in known:
                    skipped["duplicate"] += 1
                    continue
                seen.add(tag)
                known.add(identity)
            except (KeyError, TypeError, ValueError, AttributeError) as error:
                skipped[str(error) if isinstance(error, ValueError) else "malformed"] += 1
                continue
            candidates.append((deck, tag, result, payload))
    rows, added, simulator_rejections = validate_candidates(candidates)
    if rows:
        pending = output.with_suffix(".part")
        pq.write_table(pa.Table.from_pylist(rows), pending)
        pending.replace(output)
    return {"scanned": scanned, "candidates": len(candidates), "imported": dict(added),
            "excluded": dict(skipped), "simulator_rejections": dict(simulator_rejections),
            "partition": output.name if rows else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--replays", type=Path, default=Path("replays"))
    args = parser.parse_args()
    print(json.dumps(import_archive(args.archive, args.replays), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
