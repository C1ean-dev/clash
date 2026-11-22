"""Local Windows screen capture & vision server for Clash Royale (Google Play Games / scrcpy).

Captures the live Clash Royale window at ~10-15 FPS, extracts the 9:16 portrait arena
(even when Google Play Games runs in widescreen/fullscreen), reads:
  - Own elixir bar (0.0 .. 10.0)
  - Own 4-card hand + next card (via template matching with built-in deck portraits or custom templates)
  - 6 tower HP bars (monotonic damage tracking)
  - On-arena allied/enemy units (via optional YOLO detector or native OpenCV HP-bar tracker)

Serves `/api/frames?since=<idx>&limit=100` compatible with `xbow_capture_bridge.py` and
`xbow_live_overlay.py`, and also provides `--calibrate` to save an annotated debug screenshot.
"""

from __future__ import annotations

import argparse
import base64
from collections import deque
import ctypes
from ctypes import wintypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import sys
from threading import Event, Lock, Thread
import time
from urllib.parse import parse_qs, urlsplit
import webbrowser

import cv2
import mss
import numpy as np
from PIL import Image

from xbow_capture_bridge import GRID, REFERENCE_HEIGHT, REFERENCE_WIDTH
from xbow_replays import DECKS
from xbow_vision_yolo import (
    DEFAULT_VISION_MODELS_DIR,
    HybridVisionDetector,
    UltralyticsYoloDetector,
    VisionAutoCollector,
    build_vision_detector,
)

DEFAULT_WINDOW_KEYWORDS = ("clash royale", "google play games", "scrcpy", "mumu", "bluestacks")
PORTRAIT_ASPECT = REFERENCE_WIDTH / REFERENCE_HEIGHT  # 568 / 896 ≈ 0.6339

# Atlas index mapping matching dashboard/arena.js (6x6 grid in dashboard/dashboard.css).
ATLAS_CARD_INDEX = {
    "x-bow": 0, "tesla": 1, "knight": 2, "archers": 3, "skeletons": 4,
    "fireball": 5, "the-log": 6, "electro-spirit": 7, "ice-spirit": 8,
    "ice-wizard": 9, "rocket": 10, "tornado": 11,
    "hog-rider": 12, "musketeer": 13, "cannon": 14, "ice-golem": 15,
    "royal-giant": 16, "hunter": 17, "fisherman": 18, "royal-ghost": 19,
    "lightning": 20, "graveyard": 21, "poison": 22, "baby-dragon": 23,
    "tombstone": 24, "barbarian-barrel": 25, "mega-minion": 26,
    "elite-barbarians": 27, "goblins": 28, "valkyrie": 29, "balloon": 30,
    "inferno-tower": 31, "bats": 32, "miner": 33, "mortar": 34, "witch": 35,
}

# Official Supercell card IDs for Clash Royale's built-in 1-click copyDeck deep link.
OFFICIAL_CARD_IDS = {
    "knight": "26000000",
    "archers": "26000001",
    "skeletons": "26000010",
    "ice-wizard": "26000023",
    "electro-spirit": "26000084",
    "tesla": "27000006",
    "x-bow": "27000008",
    "fireball": "28000000",
    "rocket": "28000003",
    "the-log": "28000011",
    "tornado": "28000012",
}

# Normalized ROIs for the 4x2 deck builder screen in Clash Royale 9:16 portrait layout
# (after trimming any dark side toolbar from Google Play Games / emulator windows).
ROI_DECK_SCREEN_SLOTS = (
    # Top row (slots 1..4)
    (0.025, 0.255, 0.245, 0.395),
    (0.265, 0.255, 0.485, 0.395),
    (0.505, 0.255, 0.725, 0.395),
    (0.745, 0.255, 0.965, 0.395),
    # Bottom row (slots 5..8)
    (0.025, 0.480, 0.245, 0.620),
    (0.265, 0.480, 0.485, 0.620),
    (0.505, 0.480, 0.725, 0.620),
    (0.745, 0.480, 0.965, 0.620),
)


def trim_viewport_bounds(frame_bgr: np.ndarray, threshold: float = 42.0) -> tuple[np.ndarray, int, int, int, int]:
    """Strip dark side toolbars & pure-black top/bottom letterboxing (Google Play Games / PrintWindow).

    Returns (cropped_bgr, left_trim_px, top_trim_px, cropped_width, cropped_height).
    """
    h, w = frame_bgr.shape[:2]
    if w < 160 or h < 200:
        return frame_bgr, 0, 0, w, h
    # Use median across rows/cols so bright icons (e.g. white mute icon on dark sidebar) or
    # a 2-3px overlay border never defeat dark-bar detection.
    mid = frame_bgr[int(h * 0.14):int(h * 0.86), :, :]
    col_med = np.median(mid, axis=(0, 2))
    min_bar = int(w * 0.035)
    max_trim = int(w * 0.28)
    border_skip = min(5, max(0, min_bar - 1))

    # Walk past dark sidebar; tolerate a thin <=4px overlay border line embedded between two dark bands.
    left = border_skip
    while left < max_trim:
        if col_med[left] < threshold:
            left += 1
        elif left + 5 < max_trim and float(np.median(col_med[left + 4:left + 12])) < threshold:
            left += 4
        else:
            break
    if (left - border_skip) < min_bar:
        left = 0

    right = w - 1 - border_skip
    while right > w - 1 - max_trim:
        if col_med[right] < threshold:
            right -= 1
        elif right - 5 > w - 1 - max_trim and float(np.median(col_med[right - 12:right - 4])) < threshold:
            right -= 4
        else:
            break
    if ((w - 1 - border_skip) - right) < min_bar:
        right = w - 1

    cropped = frame_bgr[:, left:right + 1] if (left > 0 or right < w - 1) else frame_bgr
    if cropped.shape[1] < int(w * 0.65):
        cropped, left = frame_bgr, 0

    ch, cw = cropped.shape[:2]
    inner_x0, inner_x1 = int(cw * 0.15), int(cw * 0.85)
    row_med = np.median(cropped[:, inner_x0:inner_x1, :], axis=(1, 2))

    # Trim contiguous pure-black top bar (Google Play Games custom titlebar + top letterbox in PrintWindow).
    # Requires contiguous median < 10.0 from y=0 so a dark HUD overlay (median ~17..26) is NEVER trimmed.
    top = 0
    max_top_trim = int(ch * 0.14)
    while top < max_top_trim and row_med[top] < 10.0:
        top += 1
    if top < max(6, int(ch * 0.012)):
        top = 0

    # Trim pure-black bottom letterboxing (supports both clean PrintWindow and screenshots with taskbar below).
    max_bottom_trim = int(ch * 0.12)
    bottom = ch - 1
    dark_band = [y for y in range(ch - 1, max(top, ch - max_bottom_trim), -1) if row_med[y] < 14.0]
    if len(dark_band) >= max(8, int(ch * 0.015)):
        top_of_dark = min(dark_band)
        bottom = top_of_dark
        while bottom > max(top + int(ch * 0.70), int(ch * 0.84)) and row_med[bottom] < 14.0:
            bottom -= 1
        if bottom >= ch - 6:
            bottom = ch - 1

    if top > 0 or bottom < ch - 1:
        cropped = cropped[top:bottom + 1, :]

    return cropped, left, top, cropped.shape[1], cropped.shape[0]


def trim_dark_sidebars_bounds(frame_bgr: np.ndarray, threshold: float = 42.0) -> tuple[np.ndarray, int, int]:
    """Strip dark side toolbars & top/bottom letterboxing. Returns (cropped_bgr, left_offset_px, cropped_width)."""
    cropped, left, _top, cw, _ch = trim_viewport_bounds(frame_bgr, threshold=threshold)
    return cropped, left, cw


def trim_dark_sidebars(frame_bgr: np.ndarray, threshold: float = 42.0) -> np.ndarray:
    """Strip dark side toolbars / letterboxing (e.g. Google Play Games left bar) from the game frame."""
    return trim_viewport_bounds(frame_bgr, threshold=threshold)[0]


# Normalized UI anchor points (in the trimmed 9:16 Clash Royale layout) for deck navigation.
UI_TAB_COLLECTION = (0.35, 0.95)
UI_SUBTAB_DECKS = (0.30, 0.13)
UI_TAB_BATTLE = (0.58, 0.95)
UI_DECK_SWITCHER_BTN = (0.93, 0.20)
# Standard horizontal deck slot bar positions (slots 1..5 in row 1, 6..10 in row 2 if expanded).
UI_DECK_SLOT_BAR = (
    (0.18, 0.20), (0.31, 0.20), (0.44, 0.20), (0.57, 0.20), (0.70, 0.20),
)


def norm_to_screen_px(rect: tuple[int, int, int, int], left_trim: int, trimmed_w: int,
                      norm_x: float, norm_y: float, *,
                      top_trim: int = 0,
                      trimmed_h: int | None = None) -> tuple[int, int]:
    """Convert normalized (0..1, 0..1) trimmed-game coordinates into Windows screen pixels."""
    win_left, win_top, _win_w, win_h = rect
    eff_h = trimmed_h if trimmed_h is not None else max(1, win_h - top_trim)
    sx = win_left + left_trim + int(round(norm_x * trimmed_w))
    sy = win_top + top_trim + int(round(norm_y * eff_h))
    return sx, sy


_DPI_INITIALIZED = False


def _ensure_dpi_aware() -> None:
    """Enable Per-Monitor V2 DPI awareness so multi-monitor coordinates (e.g. negative X) stay exact."""
    global _DPI_INITIALIZED
    if _DPI_INITIALIZED or sys.platform != "win32":
        return
    _DPI_INITIALIZED = True
    try:
        user32 = ctypes.windll.user32
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = (HANDLE)-4
        user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        user32.SetProcessDpiAwarenessContext.restype = wintypes.BOOL
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def _deepest_child_at(top_hwnd: int, sx: int, sy: int) -> int:
    """Find the deepest interactive child HWND at screen point (sx, sy), skipping transparent overlays."""
    user32 = ctypes.windll.user32
    CWP_SKIPINVISIBLE = 0x0001
    CWP_SKIPDISABLED = 0x0002
    CWP_SKIPTRANSPARENT = 0x0004
    flags = CWP_SKIPINVISIBLE | CWP_SKIPDISABLED | CWP_SKIPTRANSPARENT

    # Primary: Win32 Z-order hit-test (skips invisible, disabled, and WS_EX_TRANSPARENT overlay children).
    cur = int(top_hwnd)
    for _ in range(12):
        pt = wintypes.POINT(int(sx), int(sy))
        if not user32.ScreenToClient(cur, ctypes.byref(pt)):
            break
        nxt = int(user32.ChildWindowFromPointEx(cur, pt, flags) or 0)
        if not nxt or nxt == cur:
            break
        cur = nxt
    if cur != int(top_hwnd):
        return cur

    # Fallback: smallest visible, enabled, non-transparent child containing (sx, sy).
    GWL_STYLE = -16
    GWL_EXSTYLE = -20
    WS_DISABLED = 0x08000000
    WS_EX_TRANSPARENT = 0x00000020
    candidates: list[tuple[int, int]] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def _enum_child(ch, _lp):
        if not user32.IsWindowVisible(ch):
            return True
        if user32.GetWindowLongW(ch, GWL_STYLE) & WS_DISABLED:
            return True
        if user32.GetWindowLongW(ch, GWL_EXSTYLE) & WS_EX_TRANSPARENT:
            return True
        r = wintypes.RECT()
        if user32.GetWindowRect(ch, ctypes.byref(r)):
            if r.left <= sx <= r.right and r.top <= sy <= r.bottom:
                area = max(1, (r.right - r.left) * (r.bottom - r.top))
                candidates.append((area, int(ch)))
        return True

    user32.EnumChildWindows(int(top_hwnd), _enum_child, 0)
    if not candidates:
        return int(top_hwnd)
    candidates.sort()
    return candidates[0][1]


def click_window_background(hwnd: int | None, sx: int, sy: int, *, hold_s: float = 0.04):
    """Send a background mouse click via Win32 PostMessageW WITHOUT moving the user's cursor."""
    if sys.platform != "win32":
        raise RuntimeError("Clique em segundo plano suportado apenas no Windows.")
    if not hwnd:
        raise RuntimeError("HWND da janela do jogo nao disponivel para clique em segundo plano.")
    _ensure_dpi_aware()
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    WM_MOUSEMOVE = 0x0200
    WM_LBUTTONDOWN = 0x0201
    WM_LBUTTONUP = 0x0202
    MK_LBUTTON = 0x0001

    target = _deepest_child_at(hwnd, int(sx), int(sy))
    pt = wintypes.POINT(int(sx), int(sy))
    user32.ScreenToClient(target, ctypes.byref(pt))
    lparam = ((pt.y & 0xFFFF) << 16) | (pt.x & 0xFFFF)
    ok_move = user32.PostMessageW(target, WM_MOUSEMOVE, 0, lparam)
    err = ctypes.get_last_error() if not ok_move else 0
    if not ok_move and err == 5:
        raise PermissionError(
            "Windows bloqueou PostMessageW (UIPI / Acesso Negado): o Google Play Games está "
            "rodando elevado (Administrador) enquanto o painel está normal."
        )
    time.sleep(0.015)
    user32.PostMessageW(target, WM_LBUTTONDOWN, MK_LBUTTON, lparam)
    time.sleep(hold_s)
    user32.PostMessageW(target, WM_LBUTTONUP, 0, lparam)


def click_with_cursor_restore(sx: int, sy: int, *, hold_s: float = 0.035):
    """Explicit opt-in fallback: click and immediately restore the user's previous cursor position."""
    if sys.platform != "win32":
        raise RuntimeError("Clique assistido suportado apenas no Windows.")
    user32 = ctypes.windll.user32
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    prev = wintypes.POINT()
    had_prev = bool(user32.GetCursorPos(ctypes.byref(prev)))
    try:
        user32.SetCursorPos(int(sx), int(sy))
        time.sleep(0.015)
        user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(hold_s)
        user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    finally:
        if had_prev:
            time.sleep(0.01)
            user32.SetCursorPos(int(prev.x), int(prev.y))


def dispatch_click(hwnd: int | None, sx: int, sy: int, *, mode: str = "background",
                   hold_s: float = 0.04):
    """Dispatch click using background PostMessageW (default) or opt-in cursor_restore."""
    if mode == "cursor_restore":
        click_with_cursor_restore(sx, sy, hold_s=hold_s)
    else:
        click_window_background(hwnd, sx, sy, hold_s=hold_s)


# Exact 18x32 playable Clash Royale arena grid in the trimmed 568x896 9:16 portrait frame
# (measured directly from Clash Royale's red enemy deployment border + floor tile seams in PrintWindow).
LIVE_ARENA_GRID_PX = (42.0, 91.0, 526.0, 682.0)


def _live_arena_crop_for_bridge() -> list[float]:
    """Compute CaptureBridge `arena_px` so arena_position() maps LIVE_ARENA_GRID_PX -> (0..18, 0..32)."""
    gx0, gy0, gx1, gy1 = GRID
    ax0, ay0, ax1, ay1 = LIVE_ARENA_GRID_PX
    w = (ax1 - ax0) * REFERENCE_WIDTH / (gx1 - gx0)
    h = (ay1 - ay0) * REFERENCE_HEIGHT / (gy1 - gy0)
    left = ax0 - gx0 * w / REFERENCE_WIDTH
    top = ay0 - gy0 * h / REFERENCE_HEIGHT
    return [round(left, 3), round(top, 3), round(w, 3), round(h, 3)]


LIVE_ARENA_PX = _live_arena_crop_for_bridge()


def tile_to_trimmed_norm(tile_x: float, tile_y: float) -> tuple[float, float]:
    """Project arena tile center (x in 0..17, y in 0..31) onto normalized [0..1] trimmed game coords."""
    x0, y0, x1, y1 = LIVE_ARENA_GRID_PX
    ref_x = x0 + ((tile_x + 0.5) / 18.0) * (x1 - x0)
    ref_y = y1 - ((tile_y + 0.5) / 32.0) * (y1 - y0)
    return ref_x / REFERENCE_WIDTH, ref_y / REFERENCE_HEIGHT


def play_card_on_window(hwnd: int | None, rect: tuple[int, int, int, int],
                        frame_bgr: np.ndarray, slot: int, tile_x: float, tile_y: float, *,
                        click_mode: str = "background", select_delay_s: float = 0.08,
                        click_fn=None) -> tuple[tuple[int, int], tuple[int, int]]:
    """Click hand slot `slot` (0..3) and place it on arena tile `(tile_x, tile_y)`."""
    if not 0 <= int(slot) < len(ROI_HAND_SLOTS):
        raise ValueError(f"Slot de carta invalido: {slot}")
    _cleaned, left_trim, top_trim, trimmed_w, trimmed_h = trim_viewport_bounds(frame_bgr)
    s_roi = ROI_HAND_SLOTS[int(slot)]
    slot_nx, slot_ny = (s_roi[0] + s_roi[2]) * 0.5, (s_roi[1] + s_roi[3]) * 0.5
    tile_nx, tile_ny = tile_to_trimmed_norm(float(tile_x), float(tile_y))
    slot_px = norm_to_screen_px(rect, left_trim, trimmed_w, slot_nx, slot_ny, top_trim=top_trim, trimmed_h=trimmed_h)
    tile_px = norm_to_screen_px(rect, left_trim, trimmed_w, tile_nx, tile_ny, top_trim=top_trim, trimmed_h=trimmed_h)

    def _do(pt: tuple[int, int]):
        if click_fn is not None:
            click_fn(pt[0], pt[1])
        else:
            dispatch_click(hwnd, pt[0], pt[1], mode=click_mode)

    _do(slot_px)
    if select_delay_s > 0:
        time.sleep(select_delay_s)
    _do(tile_px)
    return slot_px, tile_px


# Standard UI button coordinates in trimmed 9:16 Clash Royale lobby / post-match screens.
UI_BATTLE_BUTTON = (0.51, 0.765)
UI_MODE_SELECTOR_BUTTON = (0.84, 0.765)
UI_POST_MATCH_OK_BUTTON = (0.50, 0.86)


def _has_post_match_ok_button(portrait_bgr: np.ndarray) -> tuple[bool, tuple[float, float]]:
    """Detect Clash Royale's post-match bright blue 'OK' button at bottom-center (never in active match/lobby)."""
    ok_roi = (0.37, 0.82, 0.63, 0.93)
    ok_patch = _crop_norm(portrait_bgr, ok_roi)
    ok_hsv = cv2.cvtColor(ok_patch, cv2.COLOR_BGR2HSV)
    blue_mask = (ok_hsv[:, :, 0] >= 92) & (ok_hsv[:, :, 0] <= 116) & (ok_hsv[:, :, 1] >= 120) & (ok_hsv[:, :, 2] >= 145)
    ok_blue = float(blue_mask.mean())
    ok_white = float(((ok_hsv[:, :, 1] <= 55) & (ok_hsv[:, :, 2] >= 205)).mean())
    if ok_blue < 0.28 or ok_white < 0.012:
        return False, UI_POST_MATCH_OK_BUTTON
    # Exclude lobby (which has a large bright yellow/orange Battle button above).
    battle_patch = _crop_norm(portrait_bgr, (0.18, 0.70, 0.68, 0.85))
    b_hsv = cv2.cvtColor(battle_patch, cv2.COLOR_BGR2HSV)
    battle_yellow = float(((b_hsv[:, :, 0] >= 14) & (b_hsv[:, :, 0] <= 34) & (b_hsv[:, :, 1] >= 135) & (b_hsv[:, :, 2] >= 145)).mean())
    if battle_yellow >= 0.16:
        return False, UI_POST_MATCH_OK_BUTTON
    # Post-match OK button is isolated on a dark arena floor: flanks left & right are NOT bright blue tabs,
    # and the very bottom strip (below OK) is dark (no 5-tab Lobby/Decks navigation bar).
    for flank_roi in ((0.22, 0.85, 0.34, 0.93), (0.66, 0.85, 0.78, 0.93)):
        f_hsv = cv2.cvtColor(_crop_norm(portrait_bgr, flank_roi), cv2.COLOR_BGR2HSV)
        f_blue = float(((f_hsv[:, :, 0] >= 92) & (f_hsv[:, :, 0] <= 116) & (f_hsv[:, :, 1] >= 120) & (f_hsv[:, :, 2] >= 145)).mean())
        if f_blue >= 0.08:
            return False, UI_POST_MATCH_OK_BUTTON
    bot_strip = _crop_norm(portrait_bgr, (0.10, 0.94, 0.90, 0.99))
    if float(cv2.cvtColor(bot_strip, cv2.COLOR_BGR2HSV)[:, :, 2].mean()) >= 85.0:
        return False, UI_POST_MATCH_OK_BUTTON
    ys, xs = np.nonzero(blue_mask)
    if ys.size >= 20:
        cx = ok_roi[0] + (float(xs.mean()) / max(1, ok_patch.shape[1])) * (ok_roi[2] - ok_roi[0])
        cy = ok_roi[1] + (float(ys.mean()) / max(1, ok_patch.shape[0])) * (ok_roi[3] - ok_roi[1])
        return True, (round(cx, 3), round(cy, 3))
    return True, UI_POST_MATCH_OK_BUTTON


def find_ui_template(portrait_bgr: np.ndarray, template_name: str, *,
                     region: tuple[float, float, float, float] | None = None,
                     scales: tuple[float, ...] = (0.65, 0.80, 0.95, 1.10, 1.25),
                     templates_dir: Path | None = None) -> tuple[float, float, float] | None:
    """Multi-scale template search in portrait_bgr returning (norm_cx, norm_cy, score) or None."""
    base = (Path(templates_dir) if templates_dir else Path(__file__).resolve().parent / "calibration" / "templates")
    tmpl_path = base / "ui" / template_name
    if not tmpl_path.is_file():
        return None
    tmpl = cv2.imread(str(tmpl_path), cv2.IMREAD_COLOR)
    if tmpl is None or tmpl.size == 0:
        return None
    h, w = portrait_bgr.shape[:2]
    if region is not None:
        rx0, ry0 = max(0, int(round(region[0] * w))), max(0, int(round(region[1] * h)))
        rx1, ry1 = min(w, int(round(region[2] * w))), min(h, int(round(region[3] * h)))
        search = portrait_bgr[ry0:ry1, rx0:rx1]
    else:
        rx0, ry0 = 0, 0
        search = portrait_bgr
    sh, sw = search.shape[:2]
    if sh < 12 or sw < 12:
        return None
    best: tuple[float, float, float] | None = None
    for sc in scales:
        tw = max(8, int(round(tmpl.shape[1] * sc)))
        th = max(8, int(round(tmpl.shape[0] * sc)))
        if tw >= sw or th >= sh:
            continue
        resized = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(search, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if best is None or float(max_val) > best[2]:
            cx = (rx0 + max_loc[0] + tw * 0.5) / w
            cy = (ry0 + max_loc[1] + th * 0.5) / h
            best = (float(cx), float(cy), float(max_val))
    return best


def detect_screen_state(frame_bgr: np.ndarray, *, templates_dir: Path | None = None) -> dict:
    """Classify the current Clash Royale screen and verify whether Classic 1v1 (unranked) is active."""
    cleaned = trim_dark_sidebars(frame_bgr)
    portrait = cv2.resize(cleaned, (REFERENCE_WIDTH, REFERENCE_HEIGHT), interpolation=cv2.INTER_AREA)

    # 1. Magenta elixir bar is an unambiguous signal of an active live match.
    elixir, has_bar = read_elixir(portrait)
    if has_bar:
        return {
            "state": "in_match",
            "elixir": elixir,
            "classic_1v1_selected": False,
            "classic_1v1_card": None,
            "ok_button": None,
            "battle_button": None,
        }

    # 2. Check Post-Match bright blue "OK" button BEFORE 0-elixir fallback (winner/loser banners + emote
    # bubble remain visible on post-match screen, so OK must take precedence when no elixir bar is shown).
    is_post_match, ok_center = _has_post_match_ok_button(portrait)
    if is_post_match:
        return {
            "state": "post_match",
            "elixir": 0.0,
            "classic_1v1_selected": False,
            "classic_1v1_card": None,
            "ok_button": ok_center,
            "battle_button": None,
        }

    # 3. Secondary 0-elixir active match check (e.g. restarted mid-match right after spending to 0 elixir).
    elixir, in_game = detect_match_state(portrait)
    if in_game:
        return {
            "state": "in_match",
            "elixir": elixir,
            "classic_1v1_selected": False,
            "classic_1v1_card": None,
            "ok_button": None,
            "battle_button": None,
        }

    # 4. Check if the "Modos clássicos" picker screen is open (shows the "1x1 Clássico" banner card).
    card_hit = find_ui_template(
        portrait, "classic_1v1_card.png", region=(0.0, 0.10, 1.0, 0.88),
        scales=(0.75, 0.90, 1.0, 1.12), templates_dir=templates_dir,
    )
    if card_hit is not None and card_hit[2] >= 0.58:
        return {
            "state": "mode_picker",
            "elixir": 0.0,
            "classic_1v1_selected": False,
            "classic_1v1_card": card_hit,
            "ok_button": None,
            "battle_button": None,
        }

    # 5. Battle lobby has bright yellow/orange Battle button in lower center-left (x=0.18..0.68, y=0.72..0.85).
    battle_patch = _crop_norm(portrait, (0.18, 0.72, 0.68, 0.85))
    b_hsv = cv2.cvtColor(battle_patch, cv2.COLOR_BGR2HSV)
    battle_yellow = float(((b_hsv[:, :, 0] >= 14) & (b_hsv[:, :, 0] <= 34) & (b_hsv[:, :, 1] >= 135) & (b_hsv[:, :, 2] >= 145)).mean())
    shield_hit = find_ui_template(
        portrait, "classic_1v1_shield.png", region=(0.56, 0.64, 0.99, 0.92),
        scales=(0.55, 0.70, 0.85, 1.0, 1.15), templates_dir=templates_dir,
    )
    classic_selected = bool(shield_hit is not None and shield_hit[2] >= 0.52)

    if battle_yellow >= 0.22:
        return {
            "state": "lobby",
            "elixir": 0.0,
            "classic_1v1_selected": classic_selected,
            "classic_1v1_shield_conf": round(shield_hit[2], 3) if shield_hit else 0.0,
            "classic_1v1_card": None,
            "ok_button": None,
            "battle_button": UI_BATTLE_BUTTON,
        }

    return {
        "state": "unknown",
        "elixir": 0.0,
        "classic_1v1_selected": classic_selected,
        "classic_1v1_shield_conf": round(shield_hit[2], 3) if shield_hit else 0.0,
        "classic_1v1_card": None,
        "ok_button": None,
        "battle_button": None,
    }


def save_deck_templates(frame_bgr: np.ndarray, deck: str, templates_dir: Path,
                        card_order: list[str] | None = None) -> list[Path]:
    """Capture the 8 real card artworks from the Clash Royale deck screen as PNG templates."""
    cleaned = trim_dark_sidebars(frame_bgr)
    portrait = cv2.resize(cleaned, (REFERENCE_WIDTH, REFERENCE_HEIGHT), interpolation=cv2.INTER_AREA)
    order = list(card_order) if card_order else list(DECKS[deck])
    if len(order) != 8:
        raise ValueError(f"Esperadas 8 cartas para salvar templates, recebido {len(order)}.")
    out_folder = Path(templates_dir) / deck
    out_folder.mkdir(parents=True, exist_ok=True)
    saved = []
    for roi, card in zip(ROI_DECK_SCREEN_SLOTS, order, strict=True):
        patch = _crop_norm(portrait, roi)
        ph, pw = patch.shape[:2]
        inner = patch[int(ph * 0.12):int(ph * 0.82), int(pw * 0.10):int(pw * 0.90)]
        resized = cv2.resize(inner, (48, 56))
        path = out_folder / f"{card}.png"
        cv2.imwrite(str(path), resized)
        saved.append(path)
    return saved


def ensure_deck_by_clicks(vision: "LiveVisionServer", deck: str, *,
                          deck_slot: int | None = None,
                          max_slots: int = 5,
                          templates_dir: Path | None = None,
                          delay: float = 0.55,
                          return_to_battle: bool = True,
                          click_fn=None) -> dict:
    """Navigate to the Decks tab, verify the equipped deck, and cycle saved deck slots if needed.

    Uses background Win32 PostMessageW clicks so the user's physical mouse cursor is NEVER moved.
    Safety guards:
      - Never clicks if an active match is detected (elixir bar visible).
      - Only clicks lobby navigation & saved deck slots (never card upgrades in Collection).
    """
    bgr, rect = vision.grab_once()
    if bgr is None or rect is None:
        return {"ok": False, "error": "Janela do Clash Royale / Google Play Games nao encontrada."}
    cleaned, left_trim, top_trim, trimmed_w, trimmed_h = trim_viewport_bounds(bgr)
    portrait = cv2.resize(cleaned, (REFERENCE_WIDTH, REFERENCE_HEIGHT), interpolation=cv2.INTER_AREA)
    _, in_game = detect_match_state(portrait)
    if in_game:
        return {
            "ok": False, "already_in_game": True,
            "error": "Partida em andamento detectada; troca de deck pulada automaticamente.",
        }

    def _click(point: tuple[float, float]):
        sx, sy = norm_to_screen_px(
            rect, left_trim, trimmed_w, point[0], point[1], top_trim=top_trim, trimmed_h=trimmed_h
        )
        if click_fn is not None:
            click_fn(sx, sy)
        else:
            hwnd = getattr(vision, "current_hwnd", None)
            if not hwnd:
                found = find_game_window(getattr(vision, "window_title", None))
                hwnd = found[0] if found else None
            click_window_background(hwnd, sx, sy)
        time.sleep(delay)

    # 1. Open Collection -> Decks tab
    _click(UI_TAB_COLLECTION)
    _click(UI_SUBTAB_DECKS)

    # 2. Check currently active deck
    bgr, rect = vision.grab_once()
    if bgr is None or rect is None:
        return {"ok": False, "error": "Janela perdida durante a verificacao."}
    cleaned, left_trim, top_trim, trimmed_w, trimmed_h = trim_viewport_bounds(bgr)
    check = verify_deck_screen(bgr, deck, templates_dir=templates_dir)
    if check["valid"]:
        if return_to_battle:
            _click(UI_TAB_BATTLE)
        return {"ok": True, "switched": False, "slot_used": "atual", "check": check}

    # 3. If a specific saved slot was requested, click it first; otherwise scan saved deck slots 1..N
    slots_to_try = [deck_slot] if deck_slot is not None else list(range(1, min(max_slots, len(UI_DECK_SLOT_BAR)) + 1))
    # First click the deck switcher / slot bar to ensure deck slot numbers are visible.
    for slot_num in slots_to_try:
        idx = max(0, min(len(UI_DECK_SLOT_BAR) - 1, slot_num - 1))
        _click(UI_DECK_SLOT_BAR[idx])
        bgr, rect = vision.grab_once()
        if bgr is None or rect is None:
            break
        cleaned, left_trim, top_trim, trimmed_w, trimmed_h = trim_viewport_bounds(bgr)
        check = verify_deck_screen(bgr, deck, templates_dir=templates_dir)
        if check["valid"]:
            if return_to_battle:
                _click(UI_TAB_BATTLE)
            return {"ok": True, "switched": True, "slot_used": slot_num, "check": check}

    # Also try clicking the deck-switcher button on the right (mode deck selector) + re-check.
    _click(UI_DECK_SWITCHER_BTN)
    bgr, rect = vision.grab_once()
    if bgr is not None:
        check = verify_deck_screen(bgr, deck, templates_dir=templates_dir)
        if check["valid"]:
            if return_to_battle:
                _click(UI_TAB_BATTLE)
            return {"ok": True, "switched": True, "slot_used": "switcher", "check": check}

    out_path = Path("calibration/deck_check.png")
    if bgr is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotate_deck_check(bgr, check))
    return {
        "ok": False,
        "switched": False,
        "error": f"Nenhum slot verificado continha todas as 8 cartas de {deck}.",
        "check": check,
        "preview": str(out_path),
    }


def deck_copy_url(deck: str) -> str:
    """Build the official Supercell Clash Royale copyDeck URL for 1-click deck import."""
    ids = [OFFICIAL_CARD_IDS[card] for card in DECKS[deck] if card in OFFICIAL_CARD_IDS]
    return f"https://link.clashroyale.com/deck/en?deck={';'.join(ids)}"

# Normalized ROIs in the trimmed 9:16 Clash Royale match layout (calibrated on Google Play Games PC).
ROI_ELIXIR_BAR = (0.310, 0.968, 0.980, 0.989)
ROI_HAND_SLOTS = (
    (0.222, 0.838, 0.402, 0.950),
    (0.412, 0.838, 0.592, 0.950),
    (0.602, 0.838, 0.782, 0.950),
    (0.792, 0.838, 0.972, 0.950),
)
ROI_NEXT_CARD = (0.052, 0.928, 0.142, 0.992)
ROI_ARENA_SEARCH = (0.04, 0.085, 0.96, 0.805)

# Tower HP bar ROIs (left_princess, king, right_princess) for enemy (top) and self (bottom),
# calibrated on the clean 9:16 PrintWindow viewport.
ROI_ENEMY_TOWER_BARS = (
    (0.190, 0.130, 0.340, 0.160),  # enemy left princess
    (0.430, 0.055, 0.570, 0.085),  # enemy king
    (0.700, 0.130, 0.855, 0.160),  # enemy right princess
)
ROI_SELF_TOWER_BARS = (
    (0.190, 0.595, 0.340, 0.625),  # own left princess
    (0.430, 0.720, 0.570, 0.752),  # own king
    (0.700, 0.595, 0.855, 0.625),  # own right princess
)


def fit_portrait_rect(rect: tuple[int, int, int, int],
                      aspect: float = PORTRAIT_ASPECT) -> tuple[int, int, int, int]:
    """Center-crop a widescreen window (e.g. Google Play Games 16:9) to the 9:16 game area."""
    left, top, width, height = rect
    if width <= 0 or height <= 0:
        return rect
    if width / height > aspect * 1.12:
        game_w = max(1, round(height * aspect))
        pad_x = (width - game_w) // 2
        return (left + pad_x, top, game_w, height)
    if height / width > (1.0 / aspect) * 1.15:
        game_h = max(1, round(width / aspect))
        pad_y = (height - game_h) // 2
        return (left, top + pad_y, width, game_h)
    return rect


def find_game_window(title_hint: str | None = None,
                     keywords: tuple[str, ...] = DEFAULT_WINDOW_KEYWORDS
                     ) -> tuple[int, tuple[int, int, int, int]] | None:
    """Locate the Clash Royale / Google Play Games (hwnd, portrait_rect) on Windows."""
    if sys.platform != "win32":
        return None
    _ensure_dpi_aware()
    user32 = ctypes.windll.user32
    needles = (title_hint.lower(),) if title_hint else tuple(k.lower() for k in keywords)
    browser_markers = ("google chrome", "microsoft edge", "mozilla firefox", "opera", "brave", "x-bow control")
    matches: list[tuple[int, int, int, tuple[int, int, int, int]]] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def _enum(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.lower()
        if any(b in title for b in browser_markers):
            return True
        for rank, needle in enumerate(needles):
            if needle in title:
                crect = wintypes.RECT()
                if not user32.GetClientRect(hwnd, ctypes.byref(crect)):
                    break
                w, h = crect.right - crect.left, crect.bottom - crect.top
                if w < 240 or h < 360:
                    break
                pt = wintypes.POINT(0, 0)
                user32.ClientToScreen(hwnd, ctypes.byref(pt))
                matches.append((rank, -(w * h), int(hwnd), (pt.x, pt.y, w, h)))
                break
        return True

    user32.EnumWindows(_enum, 0)
    if not matches:
        return None
    matches.sort()
    return matches[0][2], fit_portrait_rect(matches[0][3])


def find_game_window_rect(title_hint: str | None = None,
                          keywords: tuple[str, ...] = DEFAULT_WINDOW_KEYWORDS
                          ) -> tuple[int, int, int, int] | None:
    """Locate the Clash Royale / Google Play Games client rectangle in screen pixels (Windows)."""
    found = find_game_window(title_hint=title_hint, keywords=keywords)
    return found[1] if found is not None else None


def _crop_norm(img: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = max(0, min(w - 1, int(round(roi[0] * w))))
    y0 = max(0, min(h - 1, int(round(roi[1] * h))))
    x1 = max(x0 + 1, min(w, int(round(roi[2] * w))))
    y1 = max(y0 + 1, min(h, int(round(roi[3] * h))))
    return img[y0:y1, x0:x1]


def load_atlas_templates(deck: str | None = None, css_path: Path | None = None) -> dict[str, np.ndarray]:
    """Extract BGR card portrait templates from dashboard/dashboard.css (deck subset or all 36 cards)."""
    if css_path is None:
        css_path = Path(__file__).resolve().parent / "dashboard" / "dashboard.css"
    if not css_path.is_file():
        return {}
    text = css_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r'--battle-card-portraits:\s*url\("data:image/webp;base64,([^"]+)"\)', text)
    if not m:
        return {}
    raw = base64.b64decode(m.group(1))
    atlas = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    atlas_bgr = cv2.cvtColor(atlas, cv2.COLOR_RGB2BGR)
    cols = 6
    rows = 6
    ch = atlas_bgr.shape[0] // rows
    cw = atlas_bgr.shape[1] // cols
    wanted = DECKS[deck] if deck is not None else tuple(ATLAS_CARD_INDEX.keys())
    templates = {}
    for card in wanted:
        idx = ATLAS_CARD_INDEX.get(card)
        if idx is None:
            continue
        r, c = divmod(idx, cols)
        patch = atlas_bgr[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
        # Trim outer border to focus on inner card artwork.
        pad_y, pad_x = max(1, ch // 10), max(1, cw // 10)
        templates[card] = cv2.resize(patch[pad_y:ch - pad_y, pad_x:cw - pad_x], (48, 56))
    return templates


def verify_deck_screen(frame_bgr: np.ndarray, deck: str, *,
                       templates_dir: Path | None = None,
                       threshold: float = 0.34) -> dict:
    """Inspect the 4x2 Clash Royale deck screen and check whether all 8 deck cards are equipped."""
    cleaned = trim_dark_sidebars(frame_bgr)
    portrait = cv2.resize(cleaned, (REFERENCE_WIDTH, REFERENCE_HEIGHT), interpolation=cv2.INTER_AREA)
    all_templates: dict[str, list[np.ndarray]] = {k: [v] for k, v in load_atlas_templates(deck=None).items()}
    if templates_dir is not None:
        for card, variants in load_custom_templates(deck, templates_dir).items():
            all_templates[card] = variants
    expected = list(DECKS[deck])
    expected_set = set(expected)

    slot_results = []
    detected_expected: set[str] = set()
    for slot_idx, roi in enumerate(ROI_DECK_SCREEN_SLOTS, 1):
        patch = _crop_norm(portrait, roi)
        ph, pw = patch.shape[:2]
        inner = patch[int(ph * 0.12):int(ph * 0.82), int(pw * 0.10):int(pw * 0.90)]
        resized = cv2.resize(inner, (48, 56)) if inner.size else np.zeros((56, 48, 3), dtype=np.uint8)
        best_card, best_score = None, -1.0
        for card_name, variants in all_templates.items():
            score = max(
                float(cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0])
                for tmpl in variants
            )
            if score > best_score:
                best_card, best_score = card_name, score
        in_deck = bool(best_card in expected_set and best_score >= threshold and best_card not in detected_expected)
        if in_deck and best_card is not None:
            detected_expected.add(best_card)
        slot_results.append({
            "slot": slot_idx,
            "detected": best_card if best_score >= threshold else None,
            "confidence": round(best_score, 3),
            "ok": in_deck,
        })

    missing = [card for card in expected if card not in detected_expected]
    wrong_slots = [row for row in slot_results if not row["ok"]]
    return {
        "deck": deck,
        "valid": len(missing) == 0 and len(wrong_slots) == 0,
        "matched": sorted(detected_expected),
        "missing": missing,
        "wrong_slots": wrong_slots,
        "slots": slot_results,
        "copy_url": deck_copy_url(deck),
    }


def annotate_deck_check(frame_bgr: np.ndarray, check: dict) -> np.ndarray:
    """Draw green/red boxes over the 8 deck screen slots showing which cards match or need replacing."""
    cleaned = trim_dark_sidebars(frame_bgr)
    canvas = cv2.resize(cleaned, (REFERENCE_WIDTH, REFERENCE_HEIGHT), interpolation=cv2.INTER_AREA)
    h, w = canvas.shape[:2]
    for roi, info in zip(ROI_DECK_SCREEN_SLOTS, check["slots"], strict=True):
        x0, y0, x1, y1 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)
        color = (80, 225, 130) if info["ok"] else (70, 80, 250)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 3)
        label = f"#{info['slot']} {info['detected'] or '?'} ({info['confidence']:.2f})"
        cv2.putText(canvas, label, (x0 + 2, max(16, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
    banner_color = (60, 210, 120) if check["valid"] else (60, 80, 245)
    status_txt = f"DECK OK: {check['deck']}" if check["valid"] else f"FALTANDO: {', '.join(check['missing'][:4])}"
    cv2.rectangle(canvas, (10, 10), (w - 10, 44), (20, 24, 34), -1)
    cv2.putText(canvas, status_txt, (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.52, banner_color, 2, cv2.LINE_AA)
    return canvas


def load_custom_templates(deck: str, directory: Path) -> dict[str, list[np.ndarray]]:
    """Load user-captured PNG templates (including variants like tesla.png + tesla_hand.png / _evo.png)."""
    folder = directory / deck
    templates: dict[str, list[np.ndarray]] = {}
    if not folder.is_dir():
        return templates
    for card in DECKS[deck]:
        variants = []
        for path in sorted(folder.glob(f"{card}*.png")):
            stem = path.stem
            if stem != card and not stem.startswith(f"{card}_"):
                continue
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None and img.size > 0:
                variants.append(cv2.resize(img, (48, 56)))
        if variants:
            templates[card] = variants
    return templates


class HandRecognizer:
    """Match the 4 hand cards + next card uniquely against the 8-card deck (supports Evo + Normal variants)."""

    def __init__(self, deck: str, templates_dir: Path | None = None):
        self.deck = deck
        self.cards = list(DECKS[deck])
        atlas = load_atlas_templates(deck)
        self.templates: dict[str, list[np.ndarray]] = {k: [v] for k, v in atlas.items()}
        if templates_dir is not None:
            for card, variants in load_custom_templates(deck, templates_dir).items():
                self.templates[card] = variants
        self._last_hand: list[str] = list(self.cards[:4])
        self._last_next: str | None = self.cards[4]

    def recognize(self, portrait_bgr: np.ndarray) -> tuple[list[str], str | None, float]:
        if not self.templates:
            return list(self._last_hand), self._last_next, 0.0
        rois = [*ROI_HAND_SLOTS, ROI_NEXT_CARD]
        score_mat = np.full((len(rois), len(self.cards)), -1.0, dtype=np.float32)
        for slot_idx, roi in enumerate(rois):
            patch = _crop_norm(portrait_bgr, roi)
            ph, pw = patch.shape[:2]
            # Focus on upper/middle artwork (avoiding the bottom elixir droplet on hand cards).
            inner = patch[int(ph * 0.08):int(ph * 0.78), int(pw * 0.10):int(pw * 0.90)]
            if inner.size == 0:
                continue
            resized = cv2.resize(inner, (48, 56))
            for card_idx, card in enumerate(self.cards):
                variants = self.templates.get(card)
                if not variants:
                    continue
                best_v = max(
                    float(cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0])
                    for tmpl in variants
                )
                score_mat[slot_idx, card_idx] = best_v

        # Greedy unique assignment by highest confidence across the 5 slots (4 hand + next).
        assigned: list[str | None] = [None] * len(rois)
        used_cards: set[int] = set()
        used_slots: set[int] = set()
        confidences: list[float] = []
        flat_order = np.argsort(score_mat, axis=None)[::-1]
        for flat_idx in flat_order:
            s_idx, c_idx = divmod(int(flat_idx), len(self.cards))
            if s_idx in used_slots or c_idx in used_cards:
                continue
            val = float(score_mat[s_idx, c_idx])
            if val < 0.05:
                break
            assigned[s_idx] = self.cards[c_idx]
            used_slots.add(s_idx)
            used_cards.add(c_idx)
            confidences.append(val)
            if len(used_slots) == len(rois):
                break

        # Fill any low-confidence slots from remaining deck cards / previous hand.
        remaining = [c for i, c in enumerate(self.cards) if i not in used_cards]
        for i in range(4):
            if assigned[i] is None:
                prev = self._last_hand[i]
                if prev in remaining:
                    assigned[i] = prev
                    remaining.remove(prev)
                elif remaining:
                    assigned[i] = remaining.pop(0)
                else:
                    assigned[i] = self.cards[i]
        next_card = assigned[4] or (remaining[0] if remaining else self._last_next)
        hand = [str(c) for c in assigned[:4]]
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        self._last_hand = hand
        self._last_next = next_card
        return hand, next_card, mean_conf


def read_elixir(portrait_bgr: np.ndarray) -> tuple[float, bool]:
    """Read own elixir (0.0..10.0) from the magenta elixir bar at the bottom of the screen."""
    bar = _crop_norm(portrait_bgr, ROI_ELIXIR_BAR)
    hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Clash Royale elixir bar has a tight magenta/purple hue (~140..158; 132..166 with lighting).
    # Excludes red lobby badges (H>168 or H<12) and blue UI chrome (H~105).
    magenta = ((h >= 132) & (h <= 166) & (s >= 75) & (v >= 95))
    dark_bg = (v <= 85) & (s <= 115)
    col_fill = magenta.mean(axis=0)
    col_track = (magenta | dark_bg).mean(axis=0)
    filled_cols = np.flatnonzero(col_fill >= 0.25)
    head_span = max(3, bar.shape[1] // 14)
    starts_at_left = float(col_fill[:head_span].mean()) >= 0.22
    has_magenta_bar = starts_at_left and filled_cols.size >= max(4, bar.shape[1] // 28)
    in_game_bar = bool(has_magenta_bar and float(col_track.mean()) >= 0.38)
    if not has_magenta_bar:
        return 0.0, False
    rightmost = int(filled_cols[-1]) + 1
    elixir = round(10.0 * rightmost / max(1, bar.shape[1]), 2)
    return float(max(0.0, min(10.0, elixir))), in_game_bar


def _bar_fraction(patch_bgr: np.ndarray, enemy: bool) -> float | None:
    """Estimate HP bar fill ratio in [0..1], or None if no HP bar is visible in patch."""
    if patch_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    if enemy:
        color = (((h <= 12) | (h >= 165)) & (s >= 110) & (v >= 100))
    else:
        color = ((h >= 88) & (h <= 125) & (s >= 100) & (v >= 100))
    col_ratio = color.mean(axis=0)
    cols = np.flatnonzero(col_ratio >= 0.22)
    if cols.size < 3:
        return None
    # Span relative to inner bar width (~80% of ROI width).
    span = (int(cols[-1]) - int(cols[0]) + 1) / max(1.0, patch_bgr.shape[1] * 0.78)
    return float(max(0.02, min(1.0, span)))


def detect_match_state(portrait_bgr: np.ndarray) -> tuple[float, bool]:
    """Detect whether a live match is active even if elixir is momentarily 0.0."""
    elixir, has_bar = read_elixir(portrait_bgr)
    if has_bar:
        return elixir, True
    # Post-match screen ("Vencedor! ... OK") keeps the emote bubble & red/blue banners -> never in_match.
    if _has_post_match_ok_button(portrait_bgr)[0]:
        return 0.0, False
    # Secondary check (handles restarting overlay mid-match right after spending down to 0 elixir):
    # Requires BOTH an enemy RED tower bar at the top AND an allied BLUE tower bar at the bottom
    # PLUS the white emote speech bubble on the bottom-left (never present in Lobby/Decks).
    enemy_towers = sum(
        1 for roi in ROI_ENEMY_TOWER_BARS if _bar_fraction(_crop_norm(portrait_bgr, roi), enemy=True) is not None
    )
    self_towers = sum(
        1 for roi in ROI_SELF_TOWER_BARS if _bar_fraction(_crop_norm(portrait_bgr, roi), enemy=False) is not None
    )
    if enemy_towers >= 1 and self_towers >= 1:
        emote = _crop_norm(portrait_bgr, (0.04, 0.84, 0.16, 0.90))
        hsv = cv2.cvtColor(emote, cv2.COLOR_BGR2HSV)
        white_bubble = ((hsv[:, :, 1] <= 45) & (hsv[:, :, 2] >= 200)).mean()
        if float(white_bubble) >= 0.25:
            return elixir, True
    return 0.0, False


def detect_post_match_result(frame_bgr: np.ndarray) -> str | None:
    """Read gold crown bars on Clash Royale's post-match screen to classify exact win / loss / draw."""
    cleaned = trim_dark_sidebars(frame_bgr)
    portrait = cv2.resize(cleaned, (REFERENCE_WIDTH, REFERENCE_HEIGHT), interpolation=cv2.INTER_AREA)
    if not _has_post_match_ok_button(portrait)[0]:
        return None
    # Enemy crowns sit in top banner (y=0.15..0.29), own crowns in lower banner (y=0.44..0.58).
    def _gold_ratio(roi):
        hsv = cv2.cvtColor(_crop_norm(portrait, roi), cv2.COLOR_BGR2HSV)
        gold = (hsv[:, :, 0] >= 14) & (hsv[:, :, 0] <= 34) & (hsv[:, :, 1] >= 120) & (hsv[:, :, 2] >= 145)
        return float(gold.mean())

    enemy_gold = _gold_ratio((0.18, 0.15, 0.82, 0.29))
    own_gold = _gold_ratio((0.18, 0.44, 0.82, 0.58))
    if abs(own_gold - enemy_gold) < 0.025:
        return "draw"
    return "win" if own_gold > enemy_gold else "loss"


def detect_hp_bar_units(portrait_bgr: np.ndarray) -> list[dict]:
    """Native OpenCV fallback: locate real allied (blue) and enemy (crimson-pink) units via HP bars."""
    from xbow_capture_bridge import arena_position

    h, w = portrait_bgr.shape[:2]
    ax0, ay0, ax1, ay1 = [int(round(v)) for v in LIVE_ARENA_GRID_PX]
    ax0, ay0 = max(0, ax0), max(0, ay0)
    ax1, ay1 = min(w, ax1), min(h, ay1)
    arena = portrait_bgr[ay0:ay1, ax0:ax1]
    if arena.size == 0:
        return []
    hsv = cv2.cvtColor(arena, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Exclude all 6 tower visual footprints + outer rim walls so static tower skins never spawn fake units.
    def _in_tower_footprint(tx: float, ty: float) -> bool:
        if 6.2 <= tx <= 11.8 and (ty <= 5.2 or ty >= 26.2):
            return True
        if (1.5 <= tx <= 5.8 or 12.2 <= tx <= 16.5) and (4.2 <= ty <= 9.2 or 22.8 <= ty <= 28.2):
            return True
        if tx < 0.5 or tx > 17.5 or ty < 0.3 or ty > 31.0:
            return True
        return False

    detections = []
    # Enemy troop HP bars are crisp crimson-pink/magenta-red (H >= 164 or H <= 2), EXCLUDING orange-red carpets (H=4..12).
    # Allied troop HP bars are bright sky/cyan blue (H 95..112, high saturation & value).
    masks = [
        ("enemy", (((hue <= 2) | (hue >= 164)) & (sat >= 135) & (val >= 150))),
        ("ally", ((hue >= 95) & (hue <= 112) & (sat >= 140) & (val >= 155))),
    ]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    ah, aw = arena.shape[:2]
    for team, mask in masks:
        clean = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Real troop HP bars are crisp horizontal strips.
            if bw < 10 or bw > int(w * 0.12) or bh < 2 or bh > 9 or (bw / max(1, bh)) < 2.0:
                continue
            if float(mask[by:by + bh, bx:bx + bw].mean()) < 0.50:
                continue
            # Require the dark HP-bar outline directly above or below the colored strip.
            above = val[max(0, by - 3):max(1, by), bx:bx + bw]
            below = val[min(ah - 1, by + bh):min(ah, by + bh + 3), bx:bx + bw]
            has_dark_outline = (
                (above.size > 0 and float((above < 100).mean()) >= 0.30)
                or (below.size > 0 and float((below < 100).mean()) >= 0.30)
            )
            if not has_dark_outline:
                continue
            cx = ax0 + bx + bw * 0.5
            cy = ay0 + by + bh + 14.0  # Unit feet sit slightly below its floating HP bar.
            pos = arena_position((cx, cy), LIVE_ARENA_PX)
            if pos is None or _in_tower_footprint(pos[0], pos[1]):
                continue
            est_hp = min(1.0, max(0.08, bw / max(18.0, w * 0.068)))
            detections.append({
                "class_name": "knight",
                "team": team,
                "center_px": [round(float(cx), 1), round(float(cy), 1)],
                "estimated_hp": round(float(est_hp), 3),
                "source": "hp_bar",
            })
    return detections


class OptionalYoloDetector(UltralyticsYoloDetector):
    """Backwards-compatible alias for Ultralytics YOLO unit detector."""


class ScreenVisionAnalyzer:
    """Stateful analyzer converting BGR portrait screenshots into CaptureBridge-compatible frames."""

    def __init__(self, deck: str, own_max=(4824, 3052, 3052), opponent_max=(4824, 3052, 3052),
                 templates_dir: Path | None = None, yolo_weights: str | Path | None = "auto",
                 collect_vision_dir: Path | None = None, models_dir: Path = DEFAULT_VISION_MODELS_DIR):
        self.deck = deck
        self.own_max = tuple(own_max)           # (king, left, right)
        self.opponent_max = tuple(opponent_max) # (king, left, right)
        self.hand_rec = HandRecognizer(deck, templates_dir=templates_dir)
        self.yolo: HybridVisionDetector | None = build_vision_detector(
            yolo_weights,
            deck=deck,
            models_dir=models_dir,
            hp_bar_fn=detect_hp_bar_units,
            auto_download=False,
        )
        self.collector: VisionAutoCollector | None = (
            VisionAutoCollector(collect_vision_dir) if collect_vision_dir is not None else None
        )
        self.reset_match()

    @property
    def vision_backend(self) -> str:
        return self.yolo.backend_name if self.yolo is not None else "hp_bar_only"

    def record_own_play(self, card: str, tile_x: float, tile_y: float, timestamp_s: float | None = None) -> None:
        if self.collector is not None:
            self.collector.record_own_play(card, tile_x, tile_y, timestamp_s=timestamp_s)

    def reload_vision_model(self) -> bool:
        return bool(self.yolo is not None and self.yolo.reload_if_newer())

    def reset_match(self):
        self.in_match = False
        self.match_start_s: float | None = None
        self._missed_bar_frames = 0
        # Stored in CaptureBridge visual_state order: [left, king, right].
        self.self_tower_hp = [self.own_max[1], self.own_max[0], self.own_max[2]]
        self.enemy_tower_hp = [self.opponent_max[1], self.opponent_max[0], self.opponent_max[2]]
        self._self_tower_hist: list[deque[int]] = [deque(maxlen=3) for _ in range(3)]
        self._enemy_tower_hist: list[deque[int]] = [deque(maxlen=3) for _ in range(3)]
        self._self_tower_miss = [0, 0, 0]
        self._enemy_tower_miss = [0, 0, 0]
        self._prev_enemy_tiles: list[tuple[float, float]] = []
        self._last_enemy_spawn_s = -10.0
        self.last_hand: list[str] | None = None
        self.last_elixir: float | None = None
        self.frame_index = 0
        if self.collector is not None:
            self.collector.reset_match()

    def _update_towers(self, portrait_bgr: np.ndarray, rois, current_hp: list[int],
                       max_lkr: tuple[int, int, int], enemy: bool):
        hists = self._enemy_tower_hist if enemy else self._self_tower_hist
        misses = self._enemy_tower_miss if enemy else self._self_tower_miss
        for idx, roi in enumerate(rois):
            frac = _bar_fraction(_crop_norm(portrait_bgr, roi), enemy=enemy)
            if frac is not None:
                misses[idx] = 0
                est = max(1, min(max_lkr[idx], round(frac * max_lkr[idx])))
                hists[idx].append(est)
                # Debounce over 3 consecutive readings so a single occluded/flash frame NEVER
                # permanently zeroes a tower and falsely unlocks enemy-pocket tiles.
                if len(hists[idx]) >= 3:
                    confirmed = max(hists[idx])
                    if confirmed < current_hp[idx]:
                        current_hp[idx] = confirmed
            else:
                misses[idx] += 1
                if current_hp[idx] < int(max_lkr[idx] * 0.25) and misses[idx] >= 5:
                    current_hp[idx] = 0

    def analyze(self, frame_bgr: np.ndarray, timestamp_s: float | None = None) -> dict:
        ts = time.time() if timestamp_s is None else float(timestamp_s)
        # Strip dark sidebars (e.g. Google Play Games toolbar) and normalize to 568x896.
        cleaned = trim_dark_sidebars(frame_bgr)
        portrait = cv2.resize(cleaned, (REFERENCE_WIDTH, REFERENCE_HEIGHT), interpolation=cv2.INTER_AREA)
        elixir, in_game = detect_match_state(portrait)
        self.frame_index += 1
        if not in_game:
            # During an active match, tolerate brief 0.0-elixir moments (~1.5s) before ending match.
            if self.in_match and self._missed_bar_frames < 15:
                self._missed_bar_frames += 1
                in_game = True
            else:
                if self.in_match:
                    self.reset_match()
                return {
                    "frame_index": self.frame_index,
                    "timestamp_s": ts,
                    "in_game": False,
                    "detections": [],
                    "own_actions": [],
                    "enemy_plays": [],
                    "record": {},
                }
        else:
            self._missed_bar_frames = 0

        if not self.in_match:
            self.in_match = True
            self.match_start_s = ts
            self.self_tower_hp = [self.own_max[1], self.own_max[0], self.own_max[2]]
            self.enemy_tower_hp = [self.opponent_max[1], self.opponent_max[0], self.opponent_max[2]]

        hand, next_card, hand_conf = self.hand_rec.recognize(portrait)
        own_actions = []
        if self.last_hand is not None and self.last_elixir is not None:
            elixir_drop = self.last_elixir - elixir
            if elixir_drop >= 0.8:
                for prev_card in self.last_hand:
                    if prev_card not in hand:
                        own_actions.append({"card": prev_card, "frame_confirmed": True})
                        break
        self.last_hand = list(hand)
        self.last_elixir = elixir

        self._update_towers(
            portrait, ROI_SELF_TOWER_BARS, self.self_tower_hp,
            (self.own_max[1], self.own_max[0], self.own_max[2]), enemy=False,
        )
        self._update_towers(
            portrait, ROI_ENEMY_TOWER_BARS, self.enemy_tower_hp,
            (self.opponent_max[1], self.opponent_max[0], self.opponent_max[2]), enemy=True,
        )

        detections = self.yolo.detect(portrait) if self.yolo is not None else detect_hp_bar_units(portrait)
        # Estimate enemy spawns when a genuinely new enemy unit appears so enemy_elixir tracks actual cards.
        from xbow_capture_bridge import arena_position
        enemy_plays = []
        cur_enemy_entries: list[tuple[float, float, str, bool]] = []
        for det in detections:
            if det.get("team") == "enemy" and det.get("center_px"):
                pos = arena_position(det["center_px"], LIVE_ARENA_PX)
                if pos is not None:
                    is_yolo = det.get("source") in ("public_onnx_yolo", "custom_yolo")
                    card_slug = det.get("class_name") or "knight"
                    cur_enemy_entries.append((float(pos[0]), float(pos[1]), str(card_slug), is_yolo))
        if cur_enemy_entries and (ts - self._last_enemy_spawn_s) >= 1.8:
            for tx, ty, card_slug, is_yolo in cur_enemy_entries:
                is_new = all(((tx - px) ** 2 + (ty - py) ** 2) > 9.0 for px, py in self._prev_enemy_tiles)
                if is_new:
                    enemy_plays.append({"card": card_slug, "frame_confirmed": is_yolo})
                    self._last_enemy_spawn_s = ts
                    break
        self._prev_enemy_tiles = [(tx, ty) for tx, ty, _, _ in cur_enemy_entries]

        elapsed = max(0.0, ts - (self.match_start_s or ts))
        time_left = max(0.0, 180.0 - elapsed)
        overtime = elapsed > 180.0

        if self.collector is not None:
            unmatched = self.yolo.last_unmatched_hp_bars if self.yolo is not None else []
            self.collector.observe_frame(
                portrait, detections, unmatched_hp_bars=unmatched,
                elapsed_s=elapsed, timestamp_s=ts,
            )

        return {
            "frame_index": self.frame_index,
            "timestamp_s": ts,
            "in_game": True,
            "detections": detections,
            "own_actions": own_actions,
            "enemy_plays": enemy_plays,
            "record": {
                "visual_state": {
                    "time_left_s": round(time_left, 2),
                    "overtime": overtime,
                    "hand": hand,
                    "next_card": next_card,
                    "hand_confidence": round(hand_conf, 3),
                    "elixir": elixir,
                    "tower_hp_self": list(self.self_tower_hp),
                    "tower_hp_enemy": list(self.enemy_tower_hp),
                    "arena_px": list(LIVE_ARENA_PX),
                },
            },
        }


def annotate_calibration(frame_bgr: np.ndarray, analyzed: dict) -> np.ndarray:
    """Draw all ROIs, arena grid, elixir, hand labels and unit detections for visual verification."""
    cleaned = trim_dark_sidebars(frame_bgr)
    canvas = cv2.resize(cleaned, (REFERENCE_WIDTH, REFERENCE_HEIGHT), interpolation=cv2.INTER_AREA)
    h, w = canvas.shape[:2]

    def _box(roi, color, label=""):
        x0, y0, x1, y1 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        if label:
            cv2.putText(canvas, label, (x0 + 2, max(14, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    # Arena outer 18x32 grid
    gx0, gy0, gx1, gy1 = [int(round(v)) for v in LIVE_ARENA_GRID_PX]
    cv2.rectangle(canvas, (max(0, gx0), gy0), (min(w - 1, gx1), gy1), (200, 220, 90), 2)
    vs = (analyzed.get("record") or {}).get("visual_state") or {}
    _box(ROI_ELIXIR_BAR, (255, 80, 240), f"Elixir: {vs.get('elixir', '?')}")
    hand = vs.get("hand") or ["?", "?", "?", "?"]
    for idx, roi in enumerate(ROI_HAND_SLOTS):
        _box(roi, (90, 230, 190), f"#{idx + 1} {hand[idx]}")
    _box(ROI_NEXT_CARD, (170, 190, 255), f"Next: {vs.get('next_card', '?')}")
    for idx, roi in enumerate(ROI_ENEMY_TOWER_BARS):
        hp = (vs.get("tower_hp_enemy") or ["?", "?", "?"])[idx]
        _box(roi, (80, 90, 255), str(hp))
    for idx, roi in enumerate(ROI_SELF_TOWER_BARS):
        hp = (vs.get("tower_hp_self") or ["?", "?", "?"])[idx]
        _box(roi, (255, 190, 70), str(hp))
    for det in analyzed.get("detections") or []:
        cx, cy = [int(round(v)) for v in det["center_px"]]
        color = (70, 90, 255) if det["team"] == "enemy" else (255, 210, 80)
        bbox = det.get("bbox_px")
        conf = det.get("conf")
        label = det["class_name"] + (f" {int(round(conf * 100))}%" if conf is not None else "")
        if bbox and len(bbox) == 4:
            bx1, by1, bx2, by2 = [int(round(v)) for v in bbox]
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, 2)
            cv2.putText(canvas, label, (bx1, max(14, by1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (cx, cy), 8, color, 2)
            cv2.putText(canvas, label, (cx - 18, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
    return canvas


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", _BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


def grab_hwnd_bgr(hwnd: int | None, target_rect: tuple[int, int, int, int]) -> np.ndarray | None:
    """Capture a window's own framebuffer via Win32 PrintWindow (works even behind Chrome/other windows!)."""
    if sys.platform != "win32" or not hwnd:
        return None
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    wrect = wintypes.RECT()
    if not user32.GetWindowRect(int(hwnd), ctypes.byref(wrect)):
        return None
    ww, wh = wrect.right - wrect.left, wrect.bottom - wrect.top
    if ww < 120 or wh < 160:
        return None

    hwnd_dc = user32.GetWindowDC(int(hwnd))
    if not hwnd_dc:
        return None
    mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
    bmp = gdi32.CreateCompatibleBitmap(hwnd_dc, ww, wh)
    old = gdi32.SelectObject(mem_dc, bmp)
    try:
        PW_RENDERFULLCONTENT = 0x00000002
        ok = user32.PrintWindow(int(hwnd), mem_dc, PW_RENDERFULLCONTENT)
        if not ok:
            ok = user32.PrintWindow(int(hwnd), mem_dc, 0)
        if not ok:
            return None
        bmi = _BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = ww
        bmi.bmiHeader.biHeight = -wh  # top-down DIB
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = 0  # BI_RGB
        buf = ctypes.create_string_buffer(ww * wh * 4)
        rows = gdi32.GetDIBits(mem_dc, bmp, 0, wh, buf, ctypes.byref(bmi), 0)
        if rows == 0:
            return None
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((wh, ww, 4))
        bgr_full = arr[:, :, :3].copy()
    finally:
        gdi32.SelectObject(mem_dc, old)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(int(hwnd), hwnd_dc)

    # Crop captured window bitmap to target_rect (client/portrait area)
    tl, tt, tw, th = target_rect
    ox = max(0, tl - wrect.left)
    oy = max(0, tt - wrect.top)
    ex = min(ww, ox + tw)
    ey = min(wh, oy + th)
    cropped = bgr_full[oy:ey, ox:ex] if (ex - ox >= 120 and ey - oy >= 160) else bgr_full
    # Guard against blank/black PrintWindow output (e.g. minimized window or blocked surface).
    if float(cropped.mean()) < 5.0:
        return None
    return cropped


class LiveVisionServer:
    """Background capture loop + `/api/frames` HTTP server compatible with CaptureBridge."""

    def __init__(self, deck: str, *, window_title: str | None = None,
                 region: tuple[int, int, int, int] | None = None,
                 fps: float = 10.0, own_max=(4824, 3052, 3052), opponent_max=(4824, 3052, 3052),
                 templates_dir: Path | None = None, yolo_weights: str | Path | None = "auto",
                 collect_vision_dir: Path | None = None):
        self.window_title = window_title
        self.region = region
        self.fps = max(1.0, float(fps))
        self.analyzer = ScreenVisionAnalyzer(
            deck, own_max=own_max, opponent_max=opponent_max,
            templates_dir=templates_dir, yolo_weights=yolo_weights,
            collect_vision_dir=collect_vision_dir,
        )
        self.lock = Lock()
        self.frames: deque[dict] = deque(maxlen=240)
        self.latest_jpeg: bytes | None = None
        self.current_rect: tuple[int, int, int, int] | None = region
        self.current_hwnd: int | None = None
        self.stop_event = Event()
        self.thread: Thread | None = None

    def _capture_rect(self, sct, hwnd: int | None, rect: tuple[int, int, int, int]) -> np.ndarray:
        if self.region is None and hwnd:
            pw_bgr = grab_hwnd_bgr(hwnd, rect)
            if pw_bgr is not None:
                return pw_bgr
        left, top, width, height = rect
        monitor = {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
        raw = np.asarray(sct.grab(monitor))
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

    def grab_once(self) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
        found = find_game_window(self.window_title)
        self.current_hwnd = found[0] if found else None
        rect = self.region or (found[1] if found else None)
        self.current_rect = rect
        if rect is None:
            return None, None
        with mss.mss() as sct:
            return self._capture_rect(sct, self.current_hwnd, rect), rect

    def _loop(self):
        interval = 1.0 / self.fps
        with mss.mss() as sct:
            while not self.stop_event.is_set():
                began = time.perf_counter()
                found = find_game_window(self.window_title)
                self.current_hwnd = found[0] if found else None
                rect = self.region or (found[1] if found else None)
                self.current_rect = rect
                if rect is not None:
                    try:
                        bgr = self._capture_rect(sct, self.current_hwnd, rect)
                        frame = self.analyzer.analyze(bgr)
                        with self.lock:
                            self.frames.append(frame)
                            if frame["frame_index"] % 5 == 1:
                                annotated = annotate_calibration(bgr, frame)
                                ok, enc = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                                if ok:
                                    self.latest_jpeg = enc.tobytes()
                    except Exception as exc:
                        print(f"Erro na captura: {exc}", file=sys.stderr, flush=True)
                wait = interval - (time.perf_counter() - began)
                if wait > 0:
                    self.stop_event.wait(wait)

    def start(self):
        self.thread = Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def query_frames(self, since: int, limit: int = 100) -> list[dict]:
        with self.lock:
            return [f for f in self.frames if f["frame_index"] > since][:max(1, min(limit, 200))]


def make_handler(server_state: LiveVisionServer):
    class VisionHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == "/api/frames":
                qs = parse_qs(url.query)
                since = int(qs.get("since", ["-1"])[0])
                limit = int(qs.get("limit", ["100"])[0])
                payload = json.dumps({"frames": server_state.query_frames(since, limit)}).encode("utf-8")
                self._send(200, "application/json", payload)
            elif url.path == "/api/health":
                last = server_state.frames[-1] if server_state.frames else None
                payload = json.dumps({
                    "ok": True,
                    "vision_backend": server_state.analyzer.vision_backend,
                    "window_rect": server_state.current_rect,
                    "frames_captured": server_state.analyzer.frame_index,
                    "in_game": bool(last and last.get("in_game")),
                    "elixir": ((last or {}).get("record") or {}).get("visual_state", {}).get("elixir"),
                    "hand": ((last or {}).get("record") or {}).get("visual_state", {}).get("hand"),
                }, ensure_ascii=False).encode("utf-8")
                self._send(200, "application/json", payload)
            elif url.path == "/api/preview.jpg":
                with server_state.lock:
                    jpg = server_state.latest_jpeg
                if jpg is None:
                    self._send(404, "text/plain", b"Aguardando janela do Clash Royale.")
                else:
                    self._send(200, "image/jpeg", jpg)
            else:
                self._send(404, "text/plain", b"Not found")

        def _send(self, code: int, content_type: str, body: bytes):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    return VisionHandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", choices=DECKS, default="xbow_30")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--window-title", help="Trecho do titulo da janela (padrao: Clash Royale / Google Play Games)")
    parser.add_argument("--region", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                        help="Regiao fixa da tela em pixels (sobrescreve deteccao de janela)")
    parser.add_argument("--own-max", nargs=3, type=int, default=(4824, 3052, 3052), metavar=("REI", "ESQ", "DIR"))
    parser.add_argument("--opponent-max", nargs=3, type=int, default=(4824, 3052, 3052), metavar=("REI", "ESQ", "DIR"))
    parser.add_argument("--templates-dir", type=Path, default=Path("calibration/templates"),
                        help="Pasta opcional com recortes PNG das 8 cartas capturadas do seu PC")
    parser.add_argument("--yolo-model", default="auto",
                        help="Modelo YOLO: 'auto' (custom_yolo.pt > public ONNX > HP), 'public', 'none', ou caminho .pt/.onnx")
    parser.add_argument("--collect-vision-dir", type=Path, default=None,
                        help="Pasta opcional para auto-coleta de quadros/sprites YOLO ao vivo (ex.: runs/vision_dataset)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Captura 1 quadro da janela do jogo, salva calibration/preview.png e encerra")
    parser.add_argument("--check-deck", action="store_true",
                        help="Valida os 8 slots da tela de Decks do jogo contra o deck selecionado")
    parser.add_argument("--ensure-deck", action="store_true",
                        help="Vai para a aba Decks por cliques, valida e troca para o slot salvo com o deck correto")
    parser.add_argument("--deck-slot", type=int, choices=range(1, 6), metavar="1..5",
                        help="Com --ensure-deck, testa diretamente o slot de deck salvo (1 a 5)")
    parser.add_argument("--save-deck-templates", action="store_true",
                        help="Salva os 8 recortes reais da tela de Decks atual em calibration/templates/<deck>/")
    parser.add_argument("--cards", nargs=8, metavar="CARTA",
                        help="Com --save-deck-templates, ordem exata das 8 cartas nos slots #1..#8 (padrao: ordem do deck)")
    parser.add_argument("--open-copy-link", action="store_true",
                        help="Com --check-deck, abre o link oficial copyDeck do Clash Royale se o deck estiver diferente")
    args = parser.parse_args()

    region = tuple(args.region) if args.region else None
    vision = LiveVisionServer(
        args.deck, window_title=args.window_title, region=region, fps=args.fps,
        own_max=args.own_max, opponent_max=args.opponent_max,
        templates_dir=args.templates_dir, yolo_weights=args.yolo_model,
        collect_vision_dir=args.collect_vision_dir,
    )

    if args.save_deck_templates:
        bgr, rect = vision.grab_once()
        if bgr is None:
            print("Janela do Clash Royale / Google Play Games nao encontrada. Abra a aba de Decks e tente novamente.",
                  file=sys.stderr)
            sys.exit(1)
        paths = save_deck_templates(bgr, args.deck, args.templates_dir, card_order=args.cards)
        print(f"8 templates reais salvos em {args.templates_dir / args.deck} ({len(paths)} arquivos)!")
        return

    if args.ensure_deck:
        res = ensure_deck_by_clicks(
            vision, args.deck, deck_slot=args.deck_slot, templates_dir=args.templates_dir,
        )
        if res["ok"]:
            acao = f"trocado para slot {res['slot_used']}" if res.get("switched") else "ja estava equipado"
            print(f"DECK PRONTO ({args.deck}): {acao}! Voltou para a aba Batalha.")
            return
        print(f"NAO FOI POSSIVEL EQUIPAR {args.deck}: {res.get('error')}", file=sys.stderr)
        chk = res.get("check") or {}
        if chk.get("missing"):
            print(f"  Faltando: {', '.join(chk['missing'])}", file=sys.stderr)
        if res.get("preview"):
            print(f"  Preview salvo em: {res['preview']}", file=sys.stderr)
        sys.exit(2)

    if args.check_deck:
        bgr, rect = vision.grab_once()
        if bgr is None:
            print("Janela do Clash Royale / Google Play Games nao encontrada. Abra a aba de Decks e tente novamente.",
                  file=sys.stderr)
            sys.exit(1)
        check = verify_deck_screen(bgr, args.deck, templates_dir=args.templates_dir)
        out_path = Path("calibration/deck_check.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotate_deck_check(bgr, check))
        if check["valid"]:
            print(f"DECK VALIDADO ({args.deck})! Todas as 8 cartas conferem. Preview: {out_path}")
            return
        print(f"DECK DIFERENTE DE {args.deck}!")
        print(f"  Cartas faltando: {', '.join(check['missing'])}")
        for row in check["wrong_slots"]:
            print(f"  Slot #{row['slot']}: detectado={row['detected'] or 'desconhecido'} (conf={row['confidence']})")
        print(f"  Preview salvo em: {out_path}")
        print(f"  Link oficial 1-clique para equipar {args.deck}: {check['copy_url']}")
        if args.open_copy_link:
            print("Abrindo link oficial de copia de deck no navegador...")
            webbrowser.open(check["copy_url"])
        sys.exit(2)

    if args.calibrate:
        bgr, rect = vision.grab_once()
        if bgr is None:
            print("Janela do Clash Royale / Google Play Games nao encontrada. Abra o jogo e tente novamente.",
                  file=sys.stderr)
            sys.exit(1)
        analyzed = vision.analyzer.analyze(bgr)
        out_path = Path("calibration/preview.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        annotated = annotate_calibration(bgr, analyzed)
        cv2.imwrite(str(out_path), annotated)
        vs = (analyzed.get("record") or {}).get("visual_state") or {}
        print(f"Calibracao salva em {out_path} (janela={rect}, in_game={analyzed['in_game']}, "
              f"elixir={vs.get('elixir')}, hand={vs.get('hand')}, deteccoes={len(analyzed['detections'])})")
        return

    vision.start()
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(vision))
    print(f"Servidor de Visao X-Bow rodando em http://{args.host}:{args.port} "
          f"(preview ao vivo: http://{args.host}:{args.port}/api/preview.jpg)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        vision.stop()
        httpd.server_close()


if __name__ == "__main__":
    main()
