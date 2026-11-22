"""Click-through Windows tactical overlay and match recorder for live Clash Royale."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Event, Thread
import time
from uuid import uuid4

import numpy as np
import torch

from xbow_capture_bridge import (
    CaptureBridge, GRID, REFERENCE_HEIGHT, REFERENCE_WIDTH, capture_frames, jsonl_frames,
)
from xbow_play_session import card_pt_name
from xbow_policy_service import describe_action
from xbow_public_feed import PublicFeed
from xbow_replays import DECKS
from xbow_training import load_checkpoint

DEFAULT_WINDOW_KEYWORDS = ("clash royale", "google play games", "scrcpy", "mumu", "bluestacks", "ldplayer")
TRANSPARENT_KEY = "#010101"


def tile_to_window_px(tile_x: float, tile_y: float, width: int, height: int) -> tuple[float, float]:
    """Project arena tile center (x in 0..17, y in 0..31) onto emulator window pixels."""
    x0, y0, x1, y1 = GRID
    ref_x = x0 + ((tile_x + 0.5) / 18.0) * (x1 - x0)
    ref_y = y1 - ((tile_y + 0.5) / 32.0) * (y1 - y0)
    return ref_x * width / REFERENCE_WIDTH, ref_y * height / REFERENCE_HEIGHT


def infer_match_result(frame: dict | None) -> str:
    """Infer match outcome from the last observed tower crowns and HP."""
    if not frame:
        return "unknown"
    own = frame.get("own_towers") or {}
    opp = frame.get("opponent_towers") or {}
    own_crowns = int(own.get("crowns", 0))
    opp_crowns = int(opp.get("crowns", 0))
    if own_crowns != opp_crowns:
        return "win" if own_crowns > opp_crowns else "loss"
    own_hp = [int(h) for h in own.get("hp", ()) if int(h) > 0]
    opp_hp = [int(h) for h in opp.get("hp", ()) if int(h) > 0]
    if own_hp and opp_hp:
        own_min, opp_min = min(own_hp), min(opp_hp)
        if own_min != opp_min:
            return "win" if own_min > opp_min else "loss"
        own_sum, opp_sum = sum(own_hp), sum(opp_hp)
        if own_sum != opp_sum:
            return "win" if own_sum > opp_sum else "loss"
    return "draw"


class LiveMatchRecorder:
    """Record live captured matches (frames + AI advice + outcome) for offline continual learning."""

    def __init__(self, directory: Path | None, deck: str, checkpoint: str):
        self.directory = Path(directory) if directory is not None else None
        self.deck = deck
        self.checkpoint = str(checkpoint)
        self._stream = None
        self._path: Path | None = None
        self._last_frame: dict | None = None
        self._count = 0

    def record(self, frame: dict, suggestion: dict, hud: dict):
        if self.directory is None:
            return
        if frame.get("reset") or self._stream is None:
            self.finish()
            self.directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self._path = self.directory / f"live-{self.deck}-{stamp}-{uuid4().hex[:6]}.jsonl"
            self._stream = self._path.open("w", encoding="utf-8")
            self._write({
                "kind": "header",
                "deck": self.deck,
                "checkpoint": self.checkpoint,
                "started": datetime.now(timezone.utc).isoformat(),
            })
            self._count = 0
            self._last_frame = None
        entry = {
            "kind": "step",
            "frame": frame,
            "suggestion": suggestion,
            "hud": hud,
        }
        if isinstance(suggestion, dict) and suggestion.get("played_action") is not None:
            entry["played_action"] = int(suggestion["played_action"])
        self._write(entry)
        self._last_frame = frame
        self._count += 1

    def finish(self, explicit_result: str | None = None) -> Path | None:
        if self._stream is None:
            return None
        result = explicit_result or infer_match_result(self._last_frame)
        self._write({
            "kind": "footer",
            "result": result,
            "frames": self._count,
            "ended": datetime.now(timezone.utc).isoformat(),
        })
        self._stream.close()
        finished_path = self._path
        self._stream = None
        self._path = None
        self._last_frame = None
        self._count = 0
        return finished_path

    def _write(self, obj: dict):
        self._stream.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._stream.flush()


def find_emulator_window(keywords=DEFAULT_WINDOW_KEYWORDS) -> tuple[int, int, int, int] | None:
    """Locate the Clash Royale / Google Play Games 9:16 portrait area in screen pixels on Windows."""
    if os.name != "nt":
        return None
    try:
        from xbow_screen_vision import find_game_window_rect
        return find_game_window_rect(keywords=tuple(keywords))
    except Exception:
        pass
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    found: list[tuple[int, int, int, int]] = []
    lowered = tuple(k.lower() for k in keywords)

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.lower()
        if any(key in title for key in lowered):
            rect = wintypes.RECT()
            if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                w, h = rect.right - rect.left, rect.bottom - rect.top
                if w >= 200 and h >= 300:
                    found.append((rect.left, rect.top, w, h))
                    return False
        return True

    user32.EnumWindows(callback, 0)
    return found[0] if found else None


def _make_click_through(root):
    if os.name != "nt":
        return
    import ctypes

    user32 = ctypes.windll.user32
    hwnd = int(user32.GetParent(root.winfo_id()) or root.winfo_id())
    gwl_exstyle = -20
    ws_ex_layered = 0x00080000
    ws_ex_transparent = 0x00000020
    ws_ex_topmost = 0x00000008
    ws_ex_toolwindow = 0x00000080
    ws_ex_noactivate = 0x08000000
    style = user32.GetWindowLongW(hwnd, gwl_exstyle)
    user32.SetWindowLongW(
        hwnd, gwl_exstyle,
        style | ws_ex_layered | ws_ex_transparent | ws_ex_topmost | ws_ex_toolwindow | ws_ex_noactivate,
    )
    # Exclude overlay from screen capture (mss / BitBlt) on Windows 10 2004+ / 11 so YOLO never sees its own boxes!
    try:
        WDA_EXCLUDEFROMCAPTURE = 0x00000011
        user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
    except Exception:
        pass
    return hwnd


def group_overlay_detections(detections: list[dict], *, merge_dist_px: float = 34.0) -> list[dict]:
    """Cluster nearby same-team/same-class detections so swarms render cleanly (e.g. 'skeletons x2')."""
    grouped: list[dict] = []
    used = [False] * len(detections)
    for i, det in enumerate(detections):
        if used[i]:
            continue
        used[i] = True
        team = det.get("team", "enemy")
        cls = str(det.get("class_name") or "unit")
        src = det.get("source") or "yolo"
        cx, cy = [float(v) for v in (det.get("center_px") or (0.0, 0.0))]
        bbox = list(det["bbox_px"]) if det.get("bbox_px") else None
        conf = det.get("conf")
        hp = float(det.get("estimated_hp", 1.0))
        count = 1

        if bbox is not None and src != "hp_bar_fallback":
            for j in range(i + 1, len(detections)):
                if used[j]:
                    continue
                other = detections[j]
                if other.get("team") != team or str(other.get("class_name") or "") != cls:
                    continue
                if not other.get("bbox_px") or other.get("source") == "hp_bar_fallback":
                    continue
                ocx, ocy = [float(v) for v in (other.get("center_px") or (0.0, 0.0))]
                if ((cx - ocx) ** 2 + (cy - ocy) ** 2) <= merge_dist_px ** 2:
                    used[j] = True
                    ob = other["bbox_px"]
                    bbox = [min(bbox[0], ob[0]), min(bbox[1], ob[1]), max(bbox[2], ob[2]), max(bbox[3], ob[3])]
                    count += 1
                    if other.get("conf") is not None:
                        conf = max(float(conf or 0.0), float(other["conf"]))
                    hp = min(hp, float(other.get("estimated_hp", 1.0)))

        grouped.append({
            "team": team,
            "class_name": cls,
            "count": count,
            "center_px": [cx, cy],
            "bbox_px": bbox,
            "conf": conf,
            "estimated_hp": hp,
            "source": src,
        })
    return grouped


class TacticalOverlay:
    """Transparent click-through HUD displaying live YOLO object boxes, AI decisions, and enemy cycle."""

    def __init__(self, updates: Queue, stop_event: Event, *,
                 region: tuple[int, int, int, int] | None = None,
                 window_title: str | None = None):
        import tkinter as tk

        if os.name == "nt":
            try:
                from xbow_screen_vision import _ensure_dpi_aware
                _ensure_dpi_aware()
            except Exception:
                pass

        self.updates = updates
        self.stop_event = stop_event
        self.fixed_region = region
        self.keywords = (window_title,) if window_title else DEFAULT_WINDOW_KEYWORDS
        self.rect = region or find_emulator_window(self.keywords) or (120, 60, 480, 820)
        self.latest: dict | None = None
        self._last_play_visual: dict | None = None
        self._hwnd: int | None = None

        self.root = tk.Tk()
        self.root.title("X-Bow 3.0 Tactical Overlay")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        if os.name == "nt":
            self.root.attributes("-transparentcolor", TRANSPARENT_KEY)
        self.root.configure(bg=TRANSPARENT_KEY)
        self.canvas = tk.Canvas(
            self.root, bg=TRANSPARENT_KEY, highlightthickness=0, bd=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.root.update_idletasks()
        self._hwnd = _make_click_through(self.root)
        self._apply_geometry(self.rect)

    def _apply_geometry(self, rect: tuple[int, int, int, int]):
        x, y, w, h = [int(v) for v in rect]
        self.rect = (x, y, w, h)
        if os.name == "nt" and self._hwnd:
            import ctypes
            HWND_TOPMOST = -1
            SWP_NOACTIVATE = 0x0010
            SWP_SHOWWINDOW = 0x0040
            ctypes.windll.user32.SetWindowPos(
                self._hwnd, HWND_TOPMOST, x, y, w, h, SWP_NOACTIVATE | SWP_SHOWWINDOW
            )
        else:
            self.root.geometry(f"{w}x{h}{x:+d}{y:+d}")

    def _viewport_box(self) -> tuple[float, float, float, float]:
        """Return (left_trim, top_trim, trimmed_w, trimmed_h) inside the overlay window."""
        _, _, w, h = self.rect
        trim = (self.latest or {}).get("viewport_trim")
        if trim and len(trim) == 4 and trim[2] > 40 and trim[3] > 60:
            return float(trim[0]), float(trim[1]), float(trim[2]), float(trim[3])
        return 0.0, 0.0, float(w), float(h)

    def _ref_to_px(self, rx: float, ry: float) -> tuple[float, float]:
        lt, tt, tw, th = self._viewport_box()
        return lt + rx * (tw / REFERENCE_WIDTH), tt + ry * (th / REFERENCE_HEIGHT)

    def _tile_to_px(self, tile_x: float, tile_y: float) -> tuple[float, float]:
        try:
            from xbow_screen_vision import LIVE_ARENA_GRID_PX
            gx0, gy0, gx1, gy1 = LIVE_ARENA_GRID_PX
            ref_x = gx0 + ((tile_x + 0.5) / 18.0) * (gx1 - gx0)
            ref_y = gy1 - ((tile_y + 0.5) / 32.0) * (gy1 - gy0)
            return self._ref_to_px(ref_x, ref_y)
        except Exception:
            lt, tt, tw, th = self._viewport_box()
            px, py = tile_to_window_px(tile_x, tile_y, int(tw), int(th))
            return lt + px, tt + py

    def _draw_detections(self, detections: list[dict]):
        """Draw crisp red (enemy) & blue (ally) YOLO bounding boxes with filled label pills."""
        clustered = group_overlay_detections(detections)
        for det in clustered:
            is_enemy = det["team"] == "enemy"
            box_color = "#ef4444" if is_enemy else "#3b82f6"
            pill_bg = "#dc2626" if is_enemy else "#2563eb"
            name_txt = det["class_name"].replace("-", " ")
            if det["count"] > 1:
                name_txt = f"{name_txt} x{det['count']}"
            elif det.get("source") == "hp_bar_fallback":
                name_txt = "inimigo?" if is_enemy else "aliado?"

            bbox = det.get("bbox_px")
            if bbox and len(bbox) == 4:
                x1, y1 = self._ref_to_px(float(bbox[0]), float(bbox[1]))
                x2, y2 = self._ref_to_px(float(bbox[2]), float(bbox[3]))
                # Outer crisp box matching reference style
                self.canvas.create_rectangle(x1, y1, x2, y2, outline=box_color, width=2)
                # Label pill anchored cleanly on top edge
                pill_w = max(46, len(name_txt) * 6.5 + 10)
                pill_h = 15
                px1 = max(2.0, x1)
                py2 = max(pill_h + 2.0, y1)
                py1 = py2 - pill_h
                self.canvas.create_rectangle(px1, py1, px1 + pill_w, py2, fill=pill_bg, outline=box_color, width=1)
                self.canvas.create_text(
                    px1 + 5, (py1 + py2) * 0.5, anchor="w", text=name_txt,
                    fill="#ffffff", font=("Segoe UI", 8, "bold"),
                )
                # Mini HP bar along bottom edge when unit is damaged
                hp = float(det.get("estimated_hp", 1.0))
                if 0.0 < hp < 0.99 and (x2 - x1) >= 12:
                    bw = (x2 - x1) * max(0.05, min(1.0, hp))
                    self.canvas.create_rectangle(x1, y2 + 1, x2, y2 + 4, fill="#111827", outline="")
                    self.canvas.create_rectangle(x1, y2 + 1, x1 + bw, y2 + 4, fill=box_color, outline="")
            else:
                cx, cy = self._ref_to_px(float(det["center_px"][0]), float(det["center_px"][1]))
                self.canvas.create_oval(cx - 10, cy - 10, cx + 10, cy + 10, outline=box_color, width=2, dash=(3, 2))
                self.canvas.create_text(
                    cx, cy - 15, text=name_txt, fill=box_color, font=("Segoe UI", 8, "bold"),
                )

    def _draw_decision(self, sug: dict, *, lt: float, tt: float, tw: float, th: float):
        now = time.monotonic()
        kind = sug.get("kind")
        if kind == "play_card" and sug.get("x") is not None and sug.get("y") is not None:
            self._last_play_visual = {**sug, "_ts": now}

        active_play = None
        if kind == "play_card":
            active_play = sug
        elif self._last_play_visual and (now - self._last_play_visual["_ts"]) <= 2.0:
            active_play = self._last_play_visual

        if active_play:
            tx, ty = float(active_play["x"]), float(active_play["y"])
            px, py = self._tile_to_px(tx, ty)
            card_label = card_pt_name(active_play.get("card") or "")
            prob = float(active_play.get("probability", 0.0))
            slot = int(active_play.get("slot", 0))
            executed = bool(active_play.get("executed"))

            accent = "#22c55e" if executed else "#34d399"
            tag_bg = "#052e16" if executed else "#064e3b"
            prefix = "✓ JOGOU:" if executed else "▶ DECISÃO:"

            # Hand slot highlight (aligned to ROI_HAND_SLOTS inside trimmed viewport)
            slot_centers = (0.312, 0.502, 0.692, 0.882)
            scx = lt + tw * slot_centers[max(0, min(3, slot))]
            scy = tt + th * 0.894
            sw, sh = tw * 0.088, th * 0.056
            self.canvas.create_rectangle(scx - sw, scy - sh, scx + sw, scy + sh, outline=accent, width=3)

            # Trajectory arrow from card slot to target tile
            self.canvas.create_line(scx, scy - sh, px, py, fill=accent, width=2, dash=(6, 4))

            # Target crosshair
            r = max(12, int(tw * 0.036))
            self.canvas.create_oval(px - r, py - r, px + r, py + r, outline=accent, width=3)
            self.canvas.create_line(px - r - 6, py, px + r + 6, py, fill=accent, width=2)
            self.canvas.create_line(px, py - r - 6, px, py + r + 6, fill=accent, width=2)

            label = f"{prefix} {card_label} ({int(round(tx))},{int(round(ty))}) {int(round(prob * 100))}%"
            tag_w = max(110, len(label) * 6.2 + 12)
            tag_x = max(lt + tag_w / 2 + 4, min(lt + tw - tag_w / 2 - 4, px))
            tag_y = max(tt + 66, py - r - 16)
            self.canvas.create_rectangle(
                tag_x - tag_w / 2, tag_y - 11, tag_x + tag_w / 2, tag_y + 11,
                fill=tag_bg, outline=accent, width=1,
            )
            self.canvas.create_text(
                tag_x, tag_y, text=label, fill="#bbf7d0", font=("Segoe UI", 8, "bold"),
            )
        elif kind == "ability":
            prob = float(sug.get("probability", 0.0))
            cx, cy = lt + tw * 0.5, tt + th * 0.74
            self.canvas.create_rectangle(cx - 90, cy - 13, cx + 90, cy + 13, fill="#451a03", outline="#f59e0b", width=2)
            self.canvas.create_text(
                cx, cy, text=f"ATIVAR HABILIDADE ({int(round(prob * 100))}%)",
                fill="#fde68a", font=("Segoe UI", 9, "bold"),
            )

    def _redraw(self):
        self.canvas.delete("all")
        lt, tt, tw, th = self._viewport_box()
        # Subtle frame around the active game viewport
        self.canvas.create_rectangle(lt + 1, tt + 1, lt + tw - 1, tt + th - 1, outline="#38bdf8", width=1)

        if not self.latest or self.latest.get("lobby"):
            status_txt = (self.latest or {}).get("status_text") or "X-Bow AI · No Lobby (aguardando partida…)"
            self.canvas.create_rectangle(lt + 6, tt + 6, lt + tw - 6, tt + 34, fill="#0f172a", outline="#38bdf8")
            self.canvas.create_text(
                lt + tw * 0.5, tt + 20, text=status_txt,
                fill="#e2e8f0", font=("Segoe UI", 8, "bold"),
            )
            return

        # 1. Draw all live YOLO / HP-bar object detections directly over the arena
        detections = self.latest.get("detections") or []
        if detections:
            self._draw_detections(detections)

        # 2. Draw AI decision (card highlight, trajectory line, target tile crosshair)
        sug = self.latest.get("suggestion") or {}
        self._draw_decision(sug, lt=lt, tt=tt, tw=tw, th=th)

        # 3. Compact top tactical HUD (elixir, decision, enemy cycle, vision count)
        hud = self.latest.get("hud") or {}
        enemy_elix = float(hud.get("enemy_elixir", 0.0))
        own_elix = float(hud.get("own_elixir", 0.0))
        queue = [card_pt_name(c) for c in (hud.get("enemy_queue") or [])[:4]]
        queue_txt = " · ".join(queue) if queue else "lendo ciclo…"
        val = sug.get("value")
        val_txt = f" · Val {val:+.2f}" if isinstance(val, (int, float)) else ""
        if sug.get("kind") == "play_card":
            dec_txt = f"Jogar {card_pt_name(sug.get('card') or '')} ({int(round(float(sug.get('probability', 0.0)) * 100))}%)"
        else:
            dec_txt = f"Aguardando ({int(round(float(sug.get('probability', 0.0)) * 100))}%){val_txt}"
        extra = self.latest.get("status_text")
        header = extra or f"Elixir: {own_elix:.1f}  |  Rival: ~{enemy_elix:.1f}  |  Obj: {len(detections)}"

        self.canvas.create_rectangle(lt + 6, tt + 4, lt + tw - 6, tt + 52, fill="#0f172a", outline="#38bdf8", width=1)
        self.canvas.create_text(
            lt + 13, tt + 15, anchor="w", text=header,
            fill="#f472b6", font=("Segoe UI", 8, "bold"),
        )
        self.canvas.create_text(
            lt + 13, tt + 29, anchor="w", text=f"IA: {dec_txt}",
            fill="#86efac", font=("Segoe UI", 8, "bold"),
        )
        self.canvas.create_text(
            lt + 13, tt + 43, anchor="w", text=f"Ciclo Rival: {queue_txt}",
            fill="#cbd5e1", font=("Segoe UI", 7),
        )

        warning = self.latest.get("deck_warning")
        if warning:
            self.canvas.create_rectangle(lt + 6, tt + 55, lt + tw - 6, tt + 77, fill="#450a0a", outline="#f87171", width=1)
            self.canvas.create_text(
                lt + tw * 0.5, tt + 66, text=f"⚠ {warning}",
                fill="#fecaca", font=("Segoe UI", 8, "bold"),
            )

    def _poll(self):
        if self.stop_event.is_set():
            self.root.destroy()
            return
        changed = False
        while True:
            try:
                self.latest = self.updates.get_nowait()
                changed = True
            except Empty:
                break
        target_rect = (self.latest or {}).get("window_rect")
        if target_rect and len(target_rect) == 4:
            tr = tuple(int(v) for v in target_rect)
            if tr != self.rect:
                self._apply_geometry(tr)
                changed = True
        elif self.fixed_region is None:
            found = find_emulator_window(self.keywords)
            if found and found != self.rect:
                self._apply_geometry(found)
                changed = True
        if changed or self.latest is None or self._last_play_visual is not None:
            self._redraw()
        self.root.after(90, self._poll)

    def run(self):
        self._redraw()
        self.root.after(90, self._poll)
        self.root.mainloop()


class OverlayBridge:
    """Thread-safe non-blocking launcher embedding TacticalOverlay inside LiveAutoTrainer."""

    def __init__(self, *, window_title: str | None = None):
        self.updates: Queue = Queue(maxsize=4)
        self.stop_event = Event()
        self.window_title = window_title
        self.thread: Thread | None = None
        self.started = False
        self.error: str | None = None

    def start(self) -> bool:
        if self.started:
            return True
        ready = Event()

        def _worker():
            try:
                ov = TacticalOverlay(self.updates, self.stop_event, window_title=self.window_title)
                self.started = True
                ready.set()
                ov.run()
            except Exception as exc:
                self.error = str(exc)
                ready.set()

        self.thread = Thread(target=_worker, daemon=True, name="xbow-live-overlay")
        self.thread.start()
        ready.wait(timeout=3.0)
        return self.started

    def push(self, payload: dict) -> None:
        if not self.started or self.stop_event.is_set():
            return
        while self.updates.full():
            try:
                self.updates.get_nowait()
            except Empty:
                break
        try:
            self.updates.put_nowait(payload)
        except Exception:
            pass

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.5)


def _demo_frames(deck: str, stop_event: Event):
    """Synthesize live public frames from a local RoyaleSim match to preview the overlay."""
    from royalegym.protocol import DeployStatus, EMPTY_CARD, EntityKind
    from xbow_environment import battle_env

    env = battle_env(deck, "hog_26", obs_profile="v2")
    feed_keys = PublicFeed(deck, obs_profile="v2")
    cards = {cid: key for key, cid in feed_keys.keys.items()}
    cards[EMPTY_CARD] = None
    scale = feed_keys.scale
    feed_keys.close()
    try:
        obs, _ = env.reset(seed=77)
        events = []
        first = True
        while env.agents and not stop_event.is_set():
            state = env.battle_state
            own = state.players[0]
            towers = [
                {"hp": list(p.tower_hp), "max_hp": list(p.tower_max_hp),
                 "crowns": p.crowns, "king_active": p.king_active}
                for p in state.players
            ]
            yield {
                "tick": state.tick, "reset": first,
                "hand": [cards[c] for c in own.hand], "next_card": cards[own.next_card],
                "own_elixir_milli": own.elixir_milli,
                "own_towers": towers[0], "opponent_towers": towers[1],
                "events": events,
                "units": [{
                    "side": ("own", "opponent")[u.team], "card": cards.get(u.card_id),
                    "position": [u.x / scale, u.y / scale], "hp": u.hp, "max_hp": u.max_hp,
                    "radius": u.radius / scale,
                    "kind": "building" if u.kind == EntityKind.BUILDING else "air" if u.flying else "ground",
                    "deploying": u.deploy_ticks > 0, "stunned": u.stun_ticks > 0,
                } for u in state.entities if u.tower_slot < 0 and u.hp > 0],
                "spells": [{
                    "side": ("own", "opponent")[s.team], "card": cards.get(s.card_id),
                    "position": [s.x / scale, s.y / scale],
                    "own_aim": [s.aim_x / scale, s.aim_y / scale] if s.team == 0 else None,
                } for s in state.spells],
            }
            first = False
            legal_red = np.flatnonzero(obs["red"]["action_mask"][1:])
            red_act = int(legal_red[0] + 1) if legal_red.size and state.tick % 40 == 0 else 0
            obs, *_ = env.step({"blue": 0, "red": red_act})
            events = [
                {"tick": r.tick, "side": ("own", "opponent")[r.team], "card": cards.get(r.card_id)}
                for r in env.last_results if r.status == DeployStatus.OK
            ]
            time.sleep(0.45)
    finally:
        env.close()


def _local_vision_frames(args, stop_event: Event):
    from xbow_screen_vision import LiveVisionServer

    region = tuple(args.region) if args.region else None
    vision = LiveVisionServer(
        args.deck, window_title=args.window_title, region=region, fps=args.fps,
        own_max=args.own_max, opponent_max=args.opponent_max,
        templates_dir=args.templates_dir, yolo_weights=args.yolo_model,
    )
    vision.start()
    cursor = -1
    try:
        while not stop_event.is_set():
            batch = vision.query_frames(cursor, limit=50)
            for item in batch:
                cursor = max(cursor, item["frame_index"])
                yield item
            if not batch:
                stop_event.wait(0.1)
    finally:
        vision.stop()


def _inference_worker(args, updates: Queue, stop_event: Event):
    torch.set_num_threads(2)
    policy, _ = load_checkpoint(args.checkpoint, args.deck, "cpu")
    feed = PublicFeed(args.deck, obs_profile=policy.obs_profile)
    recorder = LiveMatchRecorder(args.record_dir, args.deck, args.checkpoint)
    bridge = None if args.demo else CaptureBridge(
        args.deck, args.own_max, args.opponent_max, allow_mid_match=bool(args.vision_local),
    )
    try:
        if args.demo:
            source = _demo_frames(args.deck, stop_event)
        else:
            if args.vision_local:
                raw_source = _local_vision_frames(args, stop_event)
            elif args.capture_url:
                raw_source = capture_frames(args.capture_url)
            else:
                raw_source = jsonl_frames()
            def _guarded_source():
                was_in_game = False
                for captured in raw_source:
                    if stop_event.is_set():
                        break
                    if not captured.get("in_game"):
                        if was_in_game:
                            updates.put({"lobby": True})
                            was_in_game = False
                        bridge.push(captured)
                        continue
                    was_in_game = True
                    vs = ((captured.get("record") or {}).get("visual_state") or {})
                    hand_conf = vs.get("hand_confidence")
                    pub = bridge.push(captured)
                    if pub is not None:
                        pub["_detections"] = captured.get("detections") or []
                        if hand_conf is not None and float(hand_conf) < 0.22:
                            pub["_deck_warning"] = f"Deck suspeito (conf {hand_conf:.2f}) — confirme {args.deck}"
                        yield pub
            source = _guarded_source()
        for frame in source:
            if stop_event.is_set():
                break
            try:
                deck_warning = frame.pop("_deck_warning", None)
                detections = frame.pop("_detections", [])
                obs = feed.update(frame)
                action, logp, value = policy.decide(
                    obs, "cpu", deterministic=args.deterministic, temperature=args.temperature,
                )
                suggestion = {
                    "action": action, **describe_action(action),
                    "probability": float(np.exp(logp)), "value": round(float(value), 3),
                }
                if suggestion["kind"] == "play_card":
                    suggestion["card"] = frame["hand"][suggestion["slot"]]
                hud = feed.public_summary()
                if not deck_warning:
                    recorder.record(frame, suggestion, hud)
                updates.put({
                    "suggestion": suggestion, "hud": hud, "tick": feed.tick,
                    "detections": detections, "deck_warning": deck_warning,
                })
            except (KeyError, TypeError, ValueError) as exc:
                print(f"Quadro ignorado no overlay: {exc}", file=sys.stderr, flush=True)
    finally:
        saved = recorder.finish()
        if saved is not None:
            print(f"Partida ao vivo gravada em {saved}", flush=True)
        feed.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", choices=DECKS, default="xbow_30")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint BC/PPO treinado")
    parser.add_argument("--capture-url", help="URL do servidor de visao (ex.: http://127.0.0.1:8000)")
    parser.add_argument("--vision-local", action="store_true",
                        help="Captura diretamente a janela do Clash Royale (Google Play Games / scrcpy) sem servidor externo")
    parser.add_argument("--fps", type=float, default=10.0, help="Taxa de captura no modo --vision-local")
    parser.add_argument("--templates-dir", type=Path, default=Path("calibration/templates"))
    parser.add_argument("--yolo-model", type=Path, help="Pesos YOLO opcionais no modo --vision-local")
    parser.add_argument("--ensure-deck", action="store_true",
                        help="Antes de iniciar o overlay, vai na aba Decks por cliques e equipa o slot com o deck correto")
    parser.add_argument("--deck-slot", type=int, choices=range(1, 6), metavar="1..5",
                        help="Com --ensure-deck, slot salvo (1..5) onde o deck esta montado")
    parser.add_argument("--demo", action="store_true", help="Pre-visualiza o overlay com partida simulada")
    parser.add_argument("--window-title", help="Trecho do titulo da janela do jogo (padrao: Clash Royale / Google Play Games)")
    parser.add_argument("--region", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                        help="Regiao fixa da tela em pixels (sobrescreve deteccao de janela)")
    parser.add_argument("--own-max", nargs=3, type=int, default=(4824, 3052, 3052), metavar=("REI", "ESQ", "DIR"))
    parser.add_argument("--opponent-max", nargs=3, type=int, default=(4824, 3052, 3052), metavar=("REI", "ESQ", "DIR"))
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--record-dir", type=Path, default=Path("runs/live_matches"),
                        help="Pasta para salvar partidas reais para aprendizado continuo")
    parser.add_argument("--no-record", action="store_true", help="Desativa a gravacao das partidas ao vivo")
    args = parser.parse_args()
    if args.no_record:
        args.record_dir = None

    if args.ensure_deck and not args.demo:
        from xbow_screen_vision import LiveVisionServer, ensure_deck_by_clicks

        vision = LiveVisionServer(
            args.deck, window_title=args.window_title,
            region=tuple(args.region) if args.region else None,
            fps=args.fps, own_max=args.own_max, opponent_max=args.opponent_max,
            templates_dir=args.templates_dir, yolo_weights=args.yolo_model,
        )
        res = ensure_deck_by_clicks(
            vision, args.deck, deck_slot=args.deck_slot, templates_dir=args.templates_dir,
        )
        if res["ok"]:
            print(f"Deck {args.deck} confirmado (slot={res.get('slot_used')}). Iniciando overlay...", flush=True)
        elif res.get("already_in_game"):
            print("Partida em andamento detectada! Pulando verificacao de deck e conectando o Overlay direto a partida...", flush=True)
        else:
            print(f"Aviso na verificacao de deck: {res.get('error')}", file=sys.stderr, flush=True)

    updates: Queue = Queue()
    stop_event = Event()
    worker = Thread(target=_inference_worker, args=(args, updates, stop_event), daemon=True)
    worker.start()
    try:
        overlay = TacticalOverlay(
            updates, stop_event,
            region=tuple(args.region) if args.region else None,
            window_title=args.window_title,
        )
        overlay.run()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        worker.join(timeout=2.0)


if __name__ == "__main__":
    main()
