"""Interactive play session against trained X-Bow and tactical AI models."""

import json
from pathlib import Path
import random
import threading
from threading import Thread, RLock, Lock
import time
from uuid import uuid4

import numpy as np
import torch

from royalegym.opponents import PatientOpponent, PushOpponent
from royaleimitate.replays import CARD_NAMES
from xbow_environment import battle_env, RIVAL_DECKS
from xbow_replays import DECKS, SEATS
from xbow_selfplay import scripted_move, TACTICAL_RIVALS
from xbow_training import load_checkpoint

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"

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
}

DECK_LABELS = {
    "hog_26": "Corredor 2.6 (Ciclo Rápido)",
    "royal_giant": "Gigante Real (Controle Pesado)",
    "xbow_30": "X-Bow 3.0 (Ciclo de Cerco)",
    "icebow": "IceBow (Defesa de Gelo e Foguete)",
}


def card_pt_name(name):
    return CARD_PT_NAMES.get(name) or CARD_PT_NAMES.get(name.replace("-", " ").title()) or name


def list_available_models():
    """Discover available trained checkpoints and AI opponents."""
    models = []
    root = RUNS.resolve()

    # Pre-defined scripted bots
    models.append({
        "id": "scripted:xbow",
        "label": "Bot de Regras · X-Bow Programado",
        "deck": "xbow_30",
        "kind": "scripted",
        "desc": "Posicionamento tático programado de X-Bow e Tesla"
    })
    models.append({
        "id": "scripted:hog_push",
        "label": "Bot Tático · Push de Corredor",
        "deck": "hog_26",
        "kind": "scripted",
        "desc": "Oponente de pressão ofensiva contínua"
    })
    models.append({
        "id": "scripted:rg_patient",
        "label": "Bot Tático · Gigante Real Paciente",
        "deck": "royal_giant",
        "kind": "scripted",
        "desc": "Oponente focado em contra-ataques calculados"
    })

    if not root.is_dir():
        return models

    # Scan checkpoints in runs/
    seen = set()
    for path in sorted(root.glob("**/*.pt")):
        if path.name.startswith("."):
            continue
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            continue

        parts = rel.split("/")
        # Expected patterns: group/deck/phase.pt or group/deck/league/number.pt
        phase_or_name = path.stem
        deck_candidate = parts[-2] if len(parts) >= 2 else ""
        if deck_candidate not in DECKS:
            if len(parts) >= 3 and parts[-3] in DECKS:
                deck_candidate = parts[-3]
            else:
                continue

        group = parts[0] if len(parts) > 2 else "padrão"
        group_label = {
            "tactical": "Tático",
            "calibrated": "Calibrado",
            "public": "Observação Pública",
            "resume-check": "Retomada",
        }.get(group, group)

        if "assistido-" in group:
            group_label = "Treino Assistido"

        deck_display = "X-Bow 3.0" if deck_candidate == "xbow_30" else "IceBow"

        if phase_or_name in ("ppo", "bc"):
            label_kind = "PPO (Reforço)" if phase_or_name == "ppo" else "Imitação BC (Humana)"
            label = f"{deck_display} · {label_kind} ({group_label})"
            rec_id = f"ckpt:{rel}"
            if rec_id not in seen:
                seen.add(rec_id)
                models.append({
                    "id": rec_id,
                    "label": label,
                    "deck": deck_candidate,
                    "path": rel,
                    "kind": "checkpoint",
                    "phase": phase_or_name,
                    "desc": f"Rede neural treinada ({label_kind})"
                })

    # Put tactical ppo checkpoints at the top of checkpoints list
    def sort_key(item):
        if item["kind"] == "checkpoint":
            is_ppo = item.get("phase") == "ppo"
            is_tactical = "tactical" in item.get("path", "")
            return (0 if (is_ppo and is_tactical) else 1 if is_ppo else 2, item["label"])
        return (3, item["label"])

    models.sort(key=sort_key)
    return models


class PlaySession:
    """Manages an active real-time battle between Human and AI."""

    def __init__(self, human_deck="hog_26", ai_deck="xbow_30", model_id="scripted:xbow", speed=1.0):
        self.session_id = uuid4().hex
        self.human_deck = human_deck
        self.ai_deck = ai_deck
        self.model_id = model_id
        self.speed = float(max(0.25, min(2.0, speed)))

        # Auto-align ai_deck if a checkpoint with a specific deck is chosen
        if self.model_id.startswith("ckpt:"):
            rel = self.model_id.replace("ckpt:", "")
            if "icebow" in rel:
                self.ai_deck = "icebow"
            elif "xbow_30" in rel:
                self.ai_deck = "xbow_30"
        elif self.model_id == "scripted:hog_push":
            self.ai_deck = "hog_26"
        elif self.model_id == "scripted:rg_patient":
            self.ai_deck = "royal_giant"

        self.human_seat = 0  # Blue (bottom)
        self.ai_seat = 1     # Red (top)

        self.lock = threading.RLock()
        self.stopped = False
        self.paused = False
        self.finished = False
        self.result = None
        self.seq = 0
        self.queued_action = 0
        self.recent_moves = []
        self.thread = None

        # Build battle environment
        self.env = battle_env(self.human_deck, self.ai_deck)
        self.parser = self.env.action_parser
        self.pitch = self.parser.pitch
        self.seed = random.randint(1, 999999)
        self.rng = np.random.default_rng([self.seed, 1 - self.ai_seat])
        self.obs, _ = self.env.reset(seed=self.seed)

        # Card metadata caching
        engine_cards = self.env.engine.cards()
        self.card_by_id = {c.card_id: c for c in engine_cards}
        self.card_names = {c.card_id: c.name for c in engine_cards}
        self.unit_names = dict(enumerate(self.env.engine.unit_types()))

        # Load AI policy or scripted opponent
        self.policy = None
        self.model_label = "IA"
        self._load_model()

    def _load_model(self):
        if self.model_id.startswith("ckpt:"):
            rel_path = self.model_id.replace("ckpt:", "")
            full_path = (RUNS / rel_path).resolve()
            if full_path.is_file():
                try:
                    self.policy, info = load_checkpoint(full_path, self.ai_deck, device="cpu")
                    phase_str = "PPO" if info.get("phase") == "ppo" else "BC"
                    self.model_label = f"IA {phase_str} ({self.ai_deck})"
                    return
                except Exception as exc:
                    print(f"Error loading checkpoint {full_path}: {exc}")
        elif self.model_id == "scripted:hog_push":
            self.model_label = "Bot Corredor (Push)"
            return
        elif self.model_id == "scripted:rg_patient":
            self.model_label = "Bot Gigante Real"
            return

        # Fallback to scripted xbow
        self.model_id = "scripted:xbow"
        self.model_label = "Bot X-Bow (Scripted)"

    def start(self):
        self.thread = Thread(target=self._game_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stopped = True
        self.finished = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        try:
            self.env.close()
        except Exception:
            pass

    def pause(self):
        with self.lock:
            self.paused = True

    def resume(self):
        with self.lock:
            self.paused = False

    def toggle_pause(self):
        with self.lock:
            self.paused = not self.paused
            return self.paused

    def set_speed(self, speed):
        with self.lock:
            self.speed = float(max(0.25, min(2.0, speed)))
            return self.speed

    def queue_human_action(self, slot, x, y):
        with self.lock:
            if self.finished:
                return {"ok": False, "error": "A partida já terminou."}
            if slot < 0 or slot >= 4:
                return {"ok": False, "error": "Slot de carta inválido."}
            if x < 0 or x >= self.parser.nx or y < 0 or y >= self.parser.ny:
                return {"ok": False, "error": "Coordenada fora do campo da arena."}

            act = self.parser.encode(slot, x, y)
            mask = self.obs["blue"]["action_mask"]
            if not mask[act]:
                return {"ok": False, "error": "Jogada inválida ou elixir insuficiente para esta carta."}

            self.queued_action = act
            card_id = self.env.battle_state.players[0].hand[slot]
            c_name = self.card_names.get(card_id, "Carta")
            return {
                "ok": True,
                "action": int(act),
                "card": c_name,
                "pt_card": card_pt_name(c_name),
                "x": x,
                "y": y
            }

    def _decide_ai_action(self):
        if not self.env.agents:
            return 0

        obs_red = self.obs.get("red")
        if not obs_red:
            return 0

        if self.policy is not None:
            try:
                action, _, _ = self.policy.decide(obs_red, "cpu")
                return int(action)
            except Exception as exc:
                print(f"Policy decision error: {exc}")
                return 0
        else:
            kind = "scripted"
            if self.model_id == "scripted:hog_push":
                kind = "hog_push"
            elif self.model_id == "scripted:rg_patient":
                kind = "rg_patient"
            return scripted_move(self.env, 1, obs_red, self.rng, kind)

    def _game_loop(self):
        while not self.stopped and not self.finished:
            if self.paused:
                time.sleep(0.05)
                continue

            t_start = time.perf_counter()

            with self.lock:
                if self.finished or not self.env.agents:
                    break

                # 1. Human action
                human_act = self.queued_action
                self.queued_action = 0

                # Verify human action legality again
                if human_act != 0:
                    mask = self.obs["blue"]["action_mask"]
                    if not mask[human_act]:
                        human_act = 0

                # 2. AI action
                ai_act = self._decide_ai_action()

                # Record actions
                state = self.env.battle_state
                curr_ms = state.tick * state.tick_ms
                sides_hands = [
                    [self.card_names.get(c, "—") for c in p.hand]
                    for p in state.players
                ]
                for seat, act in enumerate((human_act, ai_act)):
                    if act:
                        move_info = self._format_action(act, seat, sides_hands[seat])
                        move_info["seat"] = seat
                        move_info["ms"] = curr_ms
                        self.recent_moves.append(move_info)
                        if len(self.recent_moves) > 10:
                            self.recent_moves.pop(0)

                # 3. Step simulation
                try:
                    self.obs, reward, term, trunc, _ = self.env.step({
                        "blue": human_act,
                        "red": ai_act
                    })
                    self.seq += 1

                    if term.get("blue", False) or trunc.get("blue", False) or not self.env.agents:
                        self.finished = True
                        winner = self.env.battle_state.winner
                        self.result = "win" if winner == 0 else ("loss" if winner == 1 else "draw")
                except Exception as exc:
                    print(f"Step simulation error: {exc}")
                    self.finished = True
                    self.result = "interrupted"

            elapsed = time.perf_counter() - t_start
            step_target = 0.5 / self.speed
            sleep_time = max(0.015, step_target - elapsed)
            time.sleep(sleep_time)

    def _format_action(self, action, seat, hand):
        if action == 0:
            return {"kind": "wait"}
        if action >= self.parser.n_tile_actions:
            return {"kind": "ability", "card": "Habilidade", "pt_card": "Habilidade"}

        slot, x, y = self.parser.decode(action)
        if seat == 1:
            x, y = self.parser.nx - 1 - x, self.parser.ny - 1 - y
        c_name = hand[slot] if 0 <= slot < len(hand) else "Carta"
        return {
            "kind": "card",
            "card": c_name,
            "pt_card": card_pt_name(c_name),
            "x": x + 0.5,
            "y": y + 0.5,
            "slot": slot
        }

    def _compute_legal_tiles_by_slot(self, mask):
        """Precompute valid (x, y) coordinates for all 4 slots in player hand."""
        tiles = {}
        for slot in range(4):
            slot_tiles = []
            for y in range(self.parser.ny):
                for x in range(self.parser.nx):
                    act = self.parser.encode(slot, x, y)
                    if mask[act]:
                        slot_tiles.append([x, y])
            tiles[slot] = slot_tiles
        return tiles

    def snapshot(self):
        with self.lock:
            state = self.env.battle_state
            tick = state.tick
            ms = tick * state.tick_ms
            seconds = ms // 1000
            clock_str = f"{seconds // 60}:{seconds % 60:02d}"

            # Calculate remaining regular match time
            reg_ticks = getattr(state, "regular_ticks", 3600)
            rem_ticks = max(0, reg_ticks - tick)
            rem_sec = rem_ticks // 20
            rem_clock = f"{rem_sec // 60}:{rem_sec % 60:02d}"

            p0 = state.players[0]
            p1 = state.players[1]

            mask_blue = self.obs["blue"]["action_mask"] if "blue" in self.obs else np.zeros(self.parser.n_actions)
            legal_tiles = self._compute_legal_tiles_by_slot(mask_blue)

            # Build human hand info
            human_hand = []
            for slot, cid in enumerate(p0.hand):
                card_obj = self.card_by_id.get(cid)
                c_name = card_obj.name if card_obj else self.card_names.get(cid, "—")
                cost = getattr(card_obj, "elixir", p0.hand_costs[slot] if hasattr(p0, "hand_costs") else 0)
                can_afford = (p0.elixir_milli >= cost * 1000)
                human_hand.append({
                    "slot": slot,
                    "id": cid,
                    "name": c_name,
                    "pt_name": card_pt_name(c_name),
                    "cost": int(cost),
                    "can_afford": bool(can_afford),
                    "legal_tiles": legal_tiles.get(slot, []),
                })

            # Next card
            next_cid = getattr(p0, "next_card", None)
            next_card_obj = self.card_by_id.get(next_cid) if next_cid is not None else None
            next_name = next_card_obj.name if next_card_obj else (self.card_names.get(next_cid) if next_cid is not None else "—")
            next_card = {
                "id": next_cid,
                "name": next_name,
                "pt_name": card_pt_name(next_name) if next_name != "—" else "—",
                "cost": int(getattr(next_card_obj, "elixir", 0)) if next_card_obj else 0
            }

            # AI hand names for display
            ai_hand = [card_pt_name(self.card_names.get(c, "—")) for c in p1.hand]

            # Board entities (pieces)
            pieces = []
            for unit in state.entities:
                if unit.hp > 0:
                    u_label = self.unit_names.get(unit.unit_type) or self.card_names.get(unit.card_id, "")
                    pieces.append([
                        unit.uid,
                        unit.team,
                        unit.kind,
                        round(unit.x / self.pitch, 2),
                        round(unit.y / self.pitch, 2),
                        unit.hp,
                        unit.max_hp,
                        unit.target_uid,
                        u_label
                    ])

            return {
                "id": self.session_id,
                "seq": self.seq,
                "status": "finished" if self.finished else ("paused" if self.paused else "running"),
                "result": self.result,
                "speed": self.speed,
                "clock": clock_str,
                "remaining": rem_clock,
                "overtime": bool(state.overtime),
                "ms": ms,
                "arena": {
                    "width": self.parser.nx,
                    "height": self.parser.ny
                },
                "human": {
                    "deck": self.human_deck,
                    "deck_label": DECK_LABELS.get(self.human_deck, self.human_deck),
                    "elixir": round(p0.elixir_milli / 1000.0, 1),
                    "crowns": int(p0.crowns),
                    "tower_hp": list(p0.tower_hp),
                    "tower_max_hp": list(p0.tower_max_hp),
                    "hand": human_hand,
                    "next_card": next_card
                },
                "ai": {
                    "deck": self.ai_deck,
                    "deck_label": DECK_LABELS.get(self.ai_deck, self.ai_deck),
                    "model_id": self.model_id,
                    "model_label": self.model_label,
                    "elixir": round(p1.elixir_milli / 1000.0, 1),
                    "crowns": int(p1.crowns),
                    "tower_hp": list(p1.tower_hp),
                    "tower_max_hp": list(p1.tower_max_hp),
                    "hand": ai_hand
                },
                "pieces": pieces,
                "moves": list(self.recent_moves)
            }


class SessionManager:
    """Singleton manager for the interactive player session."""

    def __init__(self):
        self._session = None
        self._lock = threading.Lock()

    def get_session(self):
        with self._lock:
            return self._session

    def start_session(self, human_deck="hog_26", ai_deck="xbow_30", model_id="scripted:xbow", speed=1.0):
        with self._lock:
            if self._session:
                try:
                    self._session.stop()
                except Exception:
                    pass
            session = PlaySession(human_deck, ai_deck, model_id, speed)
            session.start()
            self._session = session
            return session

    def stop_session(self):
        with self._lock:
            if self._session:
                self._session.stop()
                self._session = None


ACTIVE_SESSIONS = SessionManager()
