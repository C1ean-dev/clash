"""Interactive play session against trained X-Bow and tactical AI models."""

from collections import deque
from functools import lru_cache
from math import isfinite
from pathlib import Path
from pickle import UnpicklingError
import random
from threading import Event, Lock, RLock, Thread
import time
from uuid import uuid4

import numpy as np
import torch
from royalegym.protocol import (
    DeployStatus, HAND_SIZE, STATUS_INVISIBLE, STATUS_UNDERGROUND, slot_cost, status_of,
)
from xbow_environment import battle_env, RIVAL_DECKS
from xbow_replays import DECKS
from xbow_selfplay import scripted_move
from xbow_training import load_checkpoint, policy_from_checkpoint

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
PLAY_STEP_MS = 100
AI_STEP_MS = 500
BOT_DECKS = {"xbow": "xbow_30", "hog_push": "hog_26", "rg_patient": "royal_giant"}

CARD_PT_NAMES = {
    "Xbow": "X-Bow",
    "X-Bow": "X-Bow",
    "Tesla": "Torre Tesla",
    "Knight": "Cavaleiro",
    "Archer": "Arqueiras",
    "Archers": "Arqueiras",
    "Skeletons": "Esqueletos",
    "Fireball": "Bola de Fogo",
    "Log": "O Tronco",
    "The Log": "O Tronco",
    "ElectroSpirit": "Espírito Elétrico",
    "Electro Spirit": "Espírito Elétrico",
    "IceSpirits": "Espírito de Gelo",
    "Ice Spirit": "Espírito de Gelo",
    "IceWizard": "Mago de Gelo",
    "Ice Wizard": "Mago de Gelo",
    "Rocket": "Foguete",
    "Tornado": "Tornado",
    "HogRider": "Corredor",
    "Hog Rider": "Corredor",
    "Musketeer": "Mosqueteira",
    "Cannon": "Canhão",
    "IceGolemite": "Golem de Gelo",
    "Ice Golem": "Golem de Gelo",
    "RoyalGiant": "Gigante Real",
    "Royal Giant": "Gigante Real",
    "Hunter": "Caçador",
    "Fisherman": "Pescador",
    "Ghost": "Fantasma Real",
    "Royal Ghost": "Fantasma Real",
    "Lightning": "Relâmpago",
    "MiniPekka": "Mini P.E.K.K.A",
    "Pekka": "P.E.K.K.A",
    "Golem": "Golem",
    "Giant": "Gigante",
    "Valkyrie": "Valquíria",
    "Wizard": "Mago",
    "Princess": "Princesa",
    "Miner": "Mineiro",
    "MegaKnight": "Mega Cavaleiro",
    "Bats": "Morcegos",
    "Minions": "Servos",
    "Arrows": "Flechas",
    "Zap": "Zap",
    "Poison": "Veneno",
    "Graveyard": "Cemitério",
    "BabyDragon": "Bebê Dragão",
    "Tombstone": "Lápide",
    "BarbLog": "Barril de Bárbaro",
    "GoblinBarrel": "Barril de Goblins",
    "GoblinGang": "Gangue de Goblins",
    "InfernoTower": "Torre Inferno",
    "NightWitch": "Bruxa Sombria",
    "Lumberjack": "Lenhador",
    "MegaMinion": "Megasservo",
    "BattleRam": "Aríete de Batalha",
    "Bandit": "Bandida",
    "ElectroWizard": "Mago Elétrico",
    "MagicArcher": "Arqueiro Mágico",
}

DECK_LABELS = {
    "hog_26": "Corredor 2.6 (Ciclo Rápido)",
    "royal_giant": "Gigante Real (Controle Pesado)",
    "splashyard": "Splashyard",
    "log_bait": "Log Bait Clássico",
    "golem_beatdown": "Golem Beatdown",
    "pekka_bs": "P.E.K.K.A Bridge Spam",
    "xbow_30": "X-Bow 3.0 (Ciclo de Cerco)",
    "icebow": "IceBow (Defesa de Gelo e Foguete)",
}


def card_pt_name(name):
    return CARD_PT_NAMES.get(name) or CARD_PT_NAMES.get(name.replace("-", " ").title()) or name


PLAY_MODELS = {
    "final:xbow_30": "X-Bow 3.0 Final",
    "final:icebow": "IceBow Final",
    "rival:live": "Rival Live",
}


@lru_cache(maxsize=32)
def checkpoint_summary(path, stamp):
    """Cache metadata until the trainer replaces the checkpoint."""
    try:
        saved = torch.load(path, map_location="meta", weights_only=True)
        free = (saved.get("trainer") or {}).get("free_model") or {}
        rival = free.get("deck") == "free" and free.get("phase") == "ppo"
        return saved.get("deck"), saved.get("phase"), rival
    except (OSError, RuntimeError, ValueError, EOFError, UnpicklingError):
        return None, None, False


def latest_model_paths():
    root = RUNS.resolve()
    candidates = []
    for path in root.rglob("ppo.pt"):
        if path.parent.name not in DECKS or any(
            part.startswith(".") for part in path.relative_to(root).parts
        ):
            continue
        resolved = path.resolve()
        if resolved.is_relative_to(root):
            try:
                stat = resolved.stat()
            except OSError:
                continue
            candidates.append((stat.st_mtime_ns, stat.st_size, resolved))
    latest = {}
    for modified, size, path in sorted(candidates, reverse=True):
        deck, phase, has_rival = checkpoint_summary(path, (modified, size))
        if phase == "ppo" and deck == path.parent.name:
            latest.setdefault(f"final:{deck}", path)
            if deck == "xbow_30" and has_rival:
                latest.setdefault("rival:live", path)
        if len(latest) == len(PLAY_MODELS):
            break
    if "rival:live" not in latest:
        bootstrap = root / ".rival-live-bootstrap" / "xbow_30" / "ppo.pt"
        try:
            stat = bootstrap.stat()
        except OSError:
            return latest
        if bootstrap.resolve().is_relative_to(root) and checkpoint_summary(
            bootstrap, (stat.st_mtime_ns, stat.st_size)
        ) == ("xbow_30", "ppo", True):
            latest["rival:live"] = bootstrap
    return latest


def list_available_models():
    latest = latest_model_paths()
    return [
        {"id": model_id, "label": label, "available": model_id in latest,
         **({"decks": {deck: DECK_LABELS[deck].partition(" (")[0] for deck in RIVAL_DECKS}}
            if model_id == "rival:live" else {})}
        for model_id, label in PLAY_MODELS.items()
    ]


class PlaySession:
    """Run the interactive simulation and expose a public view to the player."""

    def __init__(self, human_deck="hog_26", ai_deck="xbow_30", model_id="scripted:xbow", speed=1.0):
        self.session_id = uuid4().hex
        self.human_deck = human_deck
        self.model_id = model_id
        self.lock = RLock()
        self.stopped = Event()
        self.paused = False
        self.finished = False
        self.result = None
        self.thread = None
        self.seq = 0
        self.next_ai_ms = 0
        self.queued_action = None
        self.command = None
        self.command_id = 0
        self.recent_moves = deque(maxlen=12)
        self.set_speed(speed)

        if not isinstance(human_deck, str) or human_deck not in DECK_LABELS:
            raise ValueError("Escolha um deck válido.")
        self.ai_deck, self.policy, self.model_label = self._load_model(ai_deck, model_id)
        obs_profile = self.policy.obs_profile if self.policy is not None else "v3"
        self.env = battle_env(
            self.human_deck, self.ai_deck, decision_ms=PLAY_STEP_MS, obs_profile=obs_profile,
        )
        self.parser = self.env.action_parser
        self.pitch = self.parser.pitch
        seed = random.randrange(1, 1 << 31)
        self.rng = np.random.default_rng(seed)
        self.obs, _ = self.env.reset(seed=seed, options={"mask_only": ("blue",)})
        self.card_by_id = {card.card_id: card for card in self.env.engine.cards()}
        self.unit_names = dict(enumerate(self.env.engine.unit_types()))

        arena = self.env.engine.arena()
        scale = arena.half_size / self.pitch
        self.arena = {
            "width": self.parser.nx, "height": self.parser.ny,
            "river": [arena.water_half_rows[0] * scale, (arena.water_half_rows[1] + 1) * scale],
            "bridges": [[left * scale, (right + 1) * scale] for left, right in arena.bridges_half_cols],
        }

    @staticmethod
    def _load_model(ai_deck, model_id):
        if not isinstance(model_id, str):
            raise ValueError("Escolha um oponente válido.")
        if model_id in PLAY_MODELS:
            path = latest_model_paths().get(model_id)
            if path is None:
                raise ValueError(f"{PLAY_MODELS[model_id]} ainda não tem um checkpoint de treino.")
            if model_id == "rival:live" and (not isinstance(ai_deck, str) or ai_deck not in RIVAL_DECKS):
                raise ValueError("Escolha um deck para o Rival Live.")
            try:
                if model_id == "rival:live":
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    policy = policy_from_checkpoint(saved["trainer"]["free_model"], "free", "cpu")
                    label = DECK_LABELS[ai_deck].partition(" (")[0]
                    return ai_deck, policy, f"Rival Live · {label}"
                deck = path.parent.name
                policy, _ = load_checkpoint(path, deck, device="cpu")
                return deck, policy, PLAY_MODELS[model_id]
            except (OSError, KeyError, ValueError, RuntimeError, EOFError, UnpicklingError) as exc:
                raise ValueError("Não foi possível carregar o checkpoint selecionado.") from exc
        if model_id.startswith("ckpt:"):
            root = RUNS.resolve()
            path = (root / model_id.removeprefix("ckpt:")).resolve()
            if (not path.is_relative_to(root) or path.name not in ("bc.pt", "ppo.pt")
                    or path.parent.name not in DECKS or not path.is_file()):
                raise ValueError("O checkpoint selecionado está indisponível.")
            deck = path.parent.name
            try:
                policy, info = load_checkpoint(path, deck, device="cpu")
            except (OSError, KeyError, ValueError, RuntimeError) as exc:
                raise ValueError("Não foi possível carregar o checkpoint selecionado.") from exc
            return deck, policy, f"{DECK_LABELS[deck].split(' (')[0]} · {info['phase'].upper()}"
        bot = model_id.removeprefix("scripted:")
        if model_id == f"scripted:{bot}" and bot in BOT_DECKS:
            deck = "icebow" if bot == "xbow" and ai_deck == "icebow" else BOT_DECKS[bot]
            label = DECK_LABELS[deck].split(" (")[0]
            return deck, None, f"{label} · Bot tático"
        raise ValueError("Escolha um oponente válido.")

    def start(self):
        self.thread = Thread(target=self._game_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
        with self.lock:
            self.finished = True
            self.env.close()

    def toggle_pause(self, paused=None):
        with self.lock:
            if not self.finished:
                self.paused = not self.paused if paused is None else paused
            return self.paused

    def set_speed(self, speed):
        if not isfinite(speed):
            raise ValueError("Escolha uma velocidade finita.")
        with self.lock:
            self.speed = max(0.25, min(2.0, speed))
            return self.speed

    def queue_human_action(self, slot, x, y, card_id=None):
        with self.lock:
            if self.finished or self.stopped.is_set():
                return {"ok": False, "error": "A partida já terminou."}
            if self.paused:
                return {"ok": False, "error": "Retome a partida para jogar."}
            if self.queued_action is not None:
                return {"ok": False, "error": "A jogada anterior ainda está na fila."}
            if not (0 <= slot < HAND_SIZE and 0 <= x < self.parser.nx and 0 <= y < self.parser.ny):
                return {"ok": False, "error": "Carta ou posição inválida."}
            action = self.parser.encode(slot, x, y)
            current_card = self.env.battle_state.players[0].hand[slot]
            if card_id is not None and card_id != current_card:
                return {"ok": False, "error": "Sua mão mudou. Selecione a carta novamente."}
            if not self.obs["blue"]["action_mask"][action]:
                return {"ok": False, "error": "Posição indisponível ou elixir insuficiente."}
            self.queued_action = action
            self.command_id += 1
            self.command = {"id": self.command_id, "status": "queued"}
            card = self.card_by_id[current_card].name
            return {"ok": True, "command_id": self.command_id, "action": action,
                    "card": card, "pt_card": card_pt_name(card), "x": x, "y": y}

    def _decide_ai_action(self):
        observation = self.obs["red"]
        if not observation["action_mask"][1:].any():
            return 0
        if self.policy is not None:
            action, _, _ = self.policy.decide(observation, "cpu", temperature=0.45)
            return action
        bot = self.model_id.removeprefix("scripted:")
        return scripted_move(self.env, 1, observation, self.rng, bot)

    def _game_loop(self):
        while not self.stopped.is_set():
            began = time.perf_counter()
            with self.lock:
                if self.finished:
                    return
                interval = 0.05 if self.paused else PLAY_STEP_MS / (1000 * self.speed)
                if not self.paused:
                    try:
                        self._step()
                    except Exception as exc:
                        print(f"Falha no modo batalha: {exc}", flush=True)
                        self.finished = True
                        self.result = "interrupted"
                        if self.queued_action is not None:
                            self.command = {"id": self.command_id, "status": "refused"}
                            self.queued_action = None
            self.stopped.wait(max(0.005, interval - (time.perf_counter() - began)))

    def _step(self):
        before = self.env.battle_state
        now_ms = before.tick * before.tick_ms
        ai_action = 0
        if now_ms >= self.next_ai_ms:
            self.next_ai_ms = now_ms + AI_STEP_MS
            ai_action = self._decide_ai_action()
        self.obs, _, term, trunc, _ = self.env.step({
            "blue": self.queued_action or 0, "red": ai_action,
        })
        self.seq += 1
        if self.queued_action is not None:
            accepted = any(move.team == 0 and move.status == DeployStatus.OK for move in self.env.last_results)
            self.command = {"id": self.command_id, "status": "played" if accepted else "refused"}
            self.queued_action = None
        for move in self.env.last_results:
            if move.status == DeployStatus.OK:
                self.recent_moves.append(self._format_move(move, before.tick_ms))

        if term["blue"] or trunc["blue"]:
            self.finished = True
            winner = self.env.battle_state.winner
            self.result = {0: "win", 1: "loss"}.get(winner, "draw")

    def _format_move(self, move, tick_ms):
        base = {"id": f"{self.seq}:{move.team}", "seat": move.team, "ms": move.tick * tick_ms}
        if move.hand_slot >= HAND_SIZE:
            return {**base, "kind": "ability", "card": "Habilidade", "pt_card": "Habilidade"}
        card = self.card_by_id[move.card_id]
        return {**base, "kind": "card", "card": card.name, "pt_card": card_pt_name(card.name),
                "card_kind": card.card_kind, "slot": move.hand_slot,
                "x": round(move.x / self.pitch, 2), "y": round(move.y / self.pitch, 2)}

    def _hand(self, player):
        mask = self.obs["blue"]["action_mask"]
        planes = mask[1:self.parser.n_tile_actions].reshape(self.parser.mask_plane_shape())
        for slot, (cid, tiles) in enumerate(zip(player.hand, planes, strict=True)):
            card = self.card_by_id.get(cid)
            cost = slot_cost(player, slot, card) if card is not None else -1
            yield {
                "slot": slot, "id": cid, **self._card_info(cid), "cost": cost,
                "can_afford": cost >= 0 and player.elixir_milli >= cost * 1000,
                "legal_tiles": np.argwhere(tiles)[:, ::-1].tolist(),
            }

    def _card_info(self, cid):
        card = self.card_by_id.get(cid)
        if card is None:
            return {"name": "-", "pt_name": "-", "cost": 0, "kind": None}
        return {"name": card.name, "pt_name": card_pt_name(card.name),
                "cost": card.elixir, "kind": card.card_kind}

    @staticmethod
    def _side_info(player, deck):
        return {"deck": deck, "deck_label": DECK_LABELS[deck], "crowns": player.crowns,
                "elixir": player.elixir_milli / 1000, "tower_hp": player.tower_hp,
                "tower_max_hp": player.tower_max_hp}

    def snapshot(self):
        with self.lock:
            state = self.env.battle_state
            ms = state.tick * state.tick_ms
            seconds = ms // 1000
            end = state.regular_ticks + (state.overtime_ticks if state.overtime else 0)
            remaining_ms = max(0, end - state.tick) * state.tick_ms
            remaining = (remaining_ms + 999) // 1000
            opening_ms = max(0, self.parser.oracle.rules.deploy_lockout_ticks - state.tick) * state.tick_ms
            blue, red = state.players

            pieces = [
                [unit.uid, unit.team, unit.kind, round(unit.x / self.pitch, 2),
                 round(unit.y / self.pitch, 2), unit.hp, unit.max_hp, unit.target_uid,
                 self.unit_names.get(unit.unit_type, "")]
                for unit in state.entities if unit.hp > 0 and (
                    unit.team == 0 or not ((status_of(unit) or 0) & (STATUS_INVISIBLE | STATUS_UNDERGROUND))
                )
            ]
            shots = [
                [p.team, round(p.x / self.pitch, 2), round(p.y / self.pitch, 2),
                 round(p.aim_x / self.pitch, 2), round(p.aim_y / self.pitch, 2), bool(p.splash)]
                for p in state.projectiles
            ]
            spells = [
                [spell.team, round(spell.x / self.pitch, 2), round(spell.y / self.pitch, 2),
                 spell.motion, self.card_by_id[spell.card_id].name]
                for spell in state.spells if spell.card_id in self.card_by_id
            ]
            return {
                "id": self.session_id, "seq": self.seq,
                "status": "finished" if self.finished else "paused" if self.paused else "running",
                "result": self.result, "speed": self.speed, "clock": f"{seconds // 60}:{seconds % 60:02d}",
                "remaining": f"{remaining // 60}:{remaining % 60:02d}", "remaining_ms": remaining_ms,
                "opening_ms": opening_ms, "elixir_rate": state.elixir_rate, "overtime": state.overtime,
                "ms": ms, "arena": self.arena, "command": self.command,
                "human": {**self._side_info(blue, self.human_deck), "hand": list(self._hand(blue)),
                          "next_card": self._card_info(blue.next_card)},
                "ai": {"deck": self.ai_deck, "deck_label": DECK_LABELS[self.ai_deck],
                       "model_id": self.model_id, "model_label": self.model_label, "crowns": red.crowns,
                       "tower_hp": red.tower_hp, "tower_max_hp": red.tower_max_hp},
                "pieces": pieces, "shots": shots, "spells": spells, "moves": list(self.recent_moves),
            }


class SessionManager:
    """Keep a single match and retire its simulation before starting another."""

    def __init__(self):
        self._session = None
        self._lock = Lock()

    def get_session(self):
        with self._lock:
            return self._session

    def start_session(self, human_deck="hog_26", ai_deck="xbow_30", model_id="scripted:xbow", speed=1.0):
        with self._lock:
            session = PlaySession(human_deck, ai_deck, model_id, speed)
            if self._session is not None:
                self._session.stop()
            session.start()
            self._session = session
            return session

    def stop_session(self, session_id=None):
        with self._lock:
            if self._session is None or (session_id is not None and self._session.session_id != session_id):
                return False
            self._session.stop()
            self._session = None
            return True


ACTIVE_SESSIONS = SessionManager()
