"""Web-panel controller for live Clash Royale capture, deck switching, overlay and AWR learning."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import subprocess
import sys
from threading import Lock, Thread

import cv2

from xbow_replays import DECKS

TOWER_LEVEL_PRESETS = {
    "11": (4824, 3052, 3052),
    "12": (5304, 3356, 3356),
    "13": (5832, 3668, 3668),
    "14": (6408, 4032, 4032),
    "15": (7032, 4424, 4424),
}


class LiveGameController:
    """Thread-safe backend manager powering the 'Jogo real & Overlay' dashboard tab."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.runs = self.root / "runs"
        self.calibration = self.root / "calibration"
        self.templates_dir = self.calibration / "templates"
        self.lock = Lock()
        self._overlay_proc: subprocess.Popen | None = None
        self._overlay_meta: dict = {}
        self._overlay_log: deque[str] = deque(maxlen=80)
        self._learn_proc: subprocess.Popen | None = None
        self._learn_exit: int | None = None
        self._learn_log: deque[str] = deque(maxlen=80)
        self._auto_proc: subprocess.Popen | None = None
        self._auto_exit: int | None = None
        self._auto_log: deque[str] = deque(maxlen=120)
        self._auto_last_payload: dict | None = None
        self._auto_state_file = self.runs / ".portal" / "auto_trainer_state.json"
        self._last_result: dict | None = None

    def _poll_proc(self, attr: str) -> bool:
        proc: subprocess.Popen | None = getattr(self, attr)
        if proc is None:
            return False
        code = proc.poll()
        if code is not None:
            if attr == "_learn_proc":
                self._learn_exit = code
            elif attr == "_auto_proc":
                self._auto_exit = code
            setattr(self, attr, None)
            return False
        return True

    def _pipe_reader(self, proc: subprocess.Popen, buf: deque[str]):
        if proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                text = line.rstrip()
                if text:
                    with self.lock:
                        buf.append(text)
        except OSError:
            pass

    def _templates_count(self, deck: str) -> int:
        folder = self.templates_dir / deck
        if not folder.is_dir():
            return 0
        return sum(1 for card in DECKS[deck] if (folder / f"{card}.png").is_file())

    def _file_mtime(self, path: Path) -> int:
        try:
            return int(path.stat().st_mtime)
        except OSError:
            return 0

    def status(self, deck: str = "xbow_30") -> dict:
        if deck not in DECKS:
            deck = "xbow_30"
        window_rect = None
        try:
            from xbow_screen_vision import find_game_window_rect
            rect = find_game_window_rect()
            window_rect = list(rect) if rect else None
        except Exception:
            window_rect = None
        auto_snapshot = None
        try:
            if self._auto_state_file.is_file():
                auto_snapshot = json.loads(self._auto_state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            auto_snapshot = None
        with self.lock:
            ov_running = self._poll_proc("_overlay_proc")
            lr_running = self._poll_proc("_learn_proc")
            au_running = self._poll_proc("_auto_proc")
            ov_pid = self._overlay_proc.pid if ov_running and self._overlay_proc else None
            au_pid = self._auto_proc.pid if au_running and self._auto_proc else None
            ov_meta = dict(self._overlay_meta) if ov_running else {}
            ov_log = list(self._overlay_log)
            lr_log = list(self._learn_log)
            au_log = list(self._auto_log)
            lr_exit = self._learn_exit
            au_exit = self._auto_exit
            last_res = dict(self._last_result) if self._last_result else None
        return {
            "deck": deck,
            "window_rect": window_rect,
            "templates_saved": self._templates_count(deck),
            "templates_total": len(DECKS[deck]),
            "previews": {
                "deck": self._file_mtime(self.calibration / "deck_check.png"),
                "arena": self._file_mtime(self.calibration / "preview.png"),
            },
            "overlay": {
                "running": ov_running,
                "pid": ov_pid,
                "mode": ov_meta.get("mode"),
                "deck": ov_meta.get("deck"),
                "checkpoint": ov_meta.get("checkpoint"),
                "log": ov_log,
            },
            "learner": {
                "running": lr_running,
                "exit_code": lr_exit,
                "log": lr_log,
            },
            "auto_trainer": {
                "running": au_running,
                "pid": au_pid,
                "exit_code": au_exit,
                "snapshot": auto_snapshot,
                "log": au_log,
            },
            "last_result": last_res,
        }

    def preview_bytes(self, kind: str) -> bytes | None:
        name = "deck_check.png" if kind == "deck" else "preview.png" if kind == "arena" else None
        if name is None:
            return None
        path = self.calibration / name
        try:
            return path.read_bytes()
        except OSError:
            return None

    def _resolve_checkpoint(self, run_id: str, deck: str, stage: str = "auto") -> Path:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("Selecione um experimento treinado antes de iniciar.")
        run_dir = (self.runs / run_id).resolve()
        if not run_dir.is_relative_to(self.runs) or run_dir.name != deck:
            raise ValueError("Experimento invalido para o deck selecionado.")
        candidates = ("ppo.pt", "bc.pt") if stage == "auto" else (f"{stage}.pt",)
        for filename in candidates:
            ckpt = run_dir / filename
            if ckpt.is_file():
                return ckpt
        raise ValueError(f"Checkpoint nao encontrado em {run_id}.")

    def _tower_max(self, level_key) -> tuple[int, int, int]:
        return TOWER_LEVEL_PRESETS.get(str(level_key or "11"), TOWER_LEVEL_PRESETS["11"])

    def _make_vision(self, deck: str, own_max=(4824, 3052, 3052)):
        from xbow_screen_vision import LiveVisionServer
        return LiveVisionServer(
            deck, own_max=own_max, opponent_max=own_max, templates_dir=self.templates_dir,
        )

    def stop_overlay(self) -> dict:
        with self.lock:
            proc = self._overlay_proc
            self._overlay_proc = None
            self._overlay_meta = {}
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            with self.lock:
                self._overlay_log.append("Overlay encerrado pelo painel.")
            return {"ok": True, "message": "Overlay encerrado."}
        return {"ok": True, "message": "Overlay ja estava parado."}

    def stop_auto_train(self) -> dict:
        with self.lock:
            proc = self._auto_proc
            self._auto_proc = None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            with self.lock:
                self._auto_log.append("Treino real autonomo parado pelo painel.")
            return {"ok": True, "message": "Treino real autonomo encerrado."}
        return {"ok": True, "message": "Treino real autonomo ja estava parado."}

    def _launch_auto_train(self, deck: str, payload: dict, *, click_mode: str = "background") -> dict:
        ckpt = self._resolve_checkpoint(payload.get("run_id", ""), deck, stage="auto")
        ref_ckpt = ckpt.parent / "bc.pt"
        out_ckpt = ckpt.parent / "ppo_live.pt"
        mode = str(payload.get("auto_mode") or "classic_1v1")
        if mode not in ("classic_1v1", "trainer_camp"):
            raise ValueError("Modo de treino real invalido (use classic_1v1 ou trainer_camp).")
        max_games = max(1, min(100, int(payload.get("max_games", 20))))
        learn_every = max(1, min(50, int(payload.get("learn_every", 5))))
        own_max = self._tower_max(payload.get("tower_level"))
        dry_run = bool(payload.get("dry_run", False))
        show_overlay = bool(payload.get("show_overlay", True))
        self.stop_auto_train()
        self.stop_overlay()
        self._auto_state_file.parent.mkdir(parents=True, exist_ok=True)
        if self._auto_state_file.exists():
            try:
                self._auto_state_file.unlink()
            except OSError:
                pass
        cmd = [
            sys.executable, "-u", str(self.root / "xbow_live_trainer.py"),
            "--checkpoint", str(ckpt),
            "--output", str(out_ckpt),
            "--deck", deck,
            "--mode", mode,
            "--max-games", str(max_games),
            "--learn-every", str(learn_every),
            "--click-mode", click_mode,
            "--own-max", *(str(v) for v in own_max),
            "--opponent-max", *(str(v) for v in own_max),
            "--record-dir", str(self.runs / "live_matches"),
            "--templates-dir", str(self.templates_dir),
            "--state-file", str(self._auto_state_file),
            "--overlay" if show_overlay else "--no-overlay",
        ]
        if ref_ckpt.is_file():
            cmd += ["--reference", str(ref_ckpt)]
        if dry_run:
            cmd.append("--dry-run")
        if not payload.get("ensure_deck", True):
            cmd.append("--no-ensure-deck")
        raw_slot = payload.get("deck_slot")
        if raw_slot not in (None, "", 0, "auto"):
            cmd += ["--deck-slot", str(int(raw_slot))]
        proc = subprocess.Popen(
            cmd, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        with self.lock:
            self._auto_proc = proc
            self._auto_exit = None
            self._auto_last_payload = {**payload, "deck": deck}
            self._auto_log.clear()
            self._auto_log.append(
                f"Treino real 1v1 iniciado (PID {proc.pid} · modo={mode} · cliques={click_mode}"
                + (" · SIMULACAO" if dry_run else "") + ")"
            )
        Thread(target=self._pipe_reader, args=(proc, self._auto_log), daemon=True).start()
        return {
            "ok": True, "action": "start_auto_train", "pid": proc.pid,
            "message": f"Treino real autônomo iniciado (PID {proc.pid} · {max_games} jogos, AWR a cada {learn_every}).",
        }

    def action(self, kind: str, payload: dict) -> dict:
        deck = payload.get("deck", "xbow_30")
        if deck not in DECKS:
            raise ValueError("Deck invalido.")

        if kind == "stop_overlay":
            return self.stop_overlay()

        if kind == "stop_auto_train":
            return self.stop_auto_train()

        if kind == "start_auto_train":
            return self._launch_auto_train(deck, payload, click_mode="background")

        if kind == "confirm_cursor_fallback":
            with self.lock:
                saved_payload = dict(self._auto_last_payload or payload)
            return self._launch_auto_train(saved_payload.get("deck", deck), saved_payload, click_mode="cursor_restore")

        if kind == "check_deck":
            from xbow_screen_vision import (
                REFERENCE_HEIGHT, REFERENCE_WIDTH, annotate_calibration, annotate_deck_check,
                detect_match_state, trim_dark_sidebars, verify_deck_screen,
            )

            vision = self._make_vision(deck)
            bgr, rect = vision.grab_once()
            if bgr is None:
                raise ValueError("Janela do Clash Royale / Google Play Games nao encontrada. Abra o jogo na aba Decks.")
            portrait = cv2.resize(trim_dark_sidebars(bgr), (REFERENCE_WIDTH, REFERENCE_HEIGHT), interpolation=cv2.INTER_AREA)
            _, in_game = detect_match_state(portrait)
            if in_game:
                analyzed = vision.analyzer.analyze(bgr)
                self.calibration.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(self.calibration / "preview.png"), annotate_calibration(bgr, analyzed))
                res = {
                    "ok": True, "action": kind, "already_in_game": True, "preview": "arena",
                    "message": "Partida em andamento detectada! Verificacao da tela de Decks pulada; exibindo captura da arena.",
                }
                with self.lock:
                    self._last_result = res
                return res
            check = verify_deck_screen(bgr, deck, templates_dir=self.templates_dir)
            self.calibration.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.calibration / "deck_check.png"), annotate_deck_check(bgr, check))
            msg = (
                f"Deck {deck} validado! Todas as 8 cartas conferem."
                if check["valid"]
                else f"Deck diferente de {deck}. Faltando: {', '.join(check['missing'])}."
            )
            res = {"ok": bool(check["valid"]), "action": kind, "message": msg, "check": check, "preview": "deck"}
            with self.lock:
                self._last_result = res
            return res

        if kind == "save_templates":
            from xbow_screen_vision import annotate_deck_check, save_deck_templates, verify_deck_screen

            card_order = payload.get("cards")
            if isinstance(card_order, str):
                card_order = [c.strip() for c in card_order.replace(",", " ").split() if c.strip()]
            if card_order:
                if len(card_order) != 8 or set(card_order) != set(DECKS[deck]):
                    raise ValueError(f"Informe exatamente as 8 cartas de {deck} na ordem dos slots #1..#8.")
            else:
                card_order = None
            vision = self._make_vision(deck)
            bgr, _ = vision.grab_once()
            if bgr is None:
                raise ValueError("Janela do Clash Royale / Google Play Games nao encontrada. Abra a aba Decks com o deck equipado.")
            paths = save_deck_templates(bgr, deck, self.templates_dir, card_order=card_order)
            check = verify_deck_screen(bgr, deck, templates_dir=self.templates_dir)
            self.calibration.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.calibration / "deck_check.png"), annotate_deck_check(bgr, check))
            res = {
                "ok": True, "action": kind,
                "message": f"{len(paths)} templates reais salvos em calibration/templates/{deck}/!",
                "check": check, "preview": "deck",
            }
            with self.lock:
                self._last_result = res
            return res

        if kind == "ensure_deck":
            from xbow_screen_vision import annotate_deck_check, ensure_deck_by_clicks

            raw_slot = payload.get("deck_slot")
            deck_slot = int(raw_slot) if raw_slot not in (None, "", 0, "auto") else None
            if deck_slot is not None and not 1 <= deck_slot <= 5:
                raise ValueError("Slot de deck deve estar entre 1 e 5.")
            vision = self._make_vision(deck)
            result = ensure_deck_by_clicks(
                vision, deck, deck_slot=deck_slot, templates_dir=self.templates_dir,
                return_to_battle=bool(payload.get("return_to_battle", True)),
            )
            bgr, _ = vision.grab_once()
            if bgr is not None and result.get("check"):
                self.calibration.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(self.calibration / "deck_check.png"), annotate_deck_check(bgr, result["check"]))
            if result["ok"]:
                acao = f"trocado para slot {result['slot_used']}" if result.get("switched") else "ja estava equipado"
                msg = f"Deck {deck} confirmado ({acao})!"
                ok_flag = True
            elif result.get("already_in_game"):
                msg = "Partida em andamento detectada! Troca de deck pulada automaticamente — pode continuar jogando."
                ok_flag = True
            else:
                msg = result.get("error") or "Nao foi possivel confirmar o deck."
                ok_flag = False
            res = {
                "ok": ok_flag, "action": kind, "message": msg,
                "check": result.get("check"), "preview": "arena" if result.get("already_in_game") else "deck",
            }
            with self.lock:
                self._last_result = res
            return res

        if kind == "calibrate":
            from xbow_screen_vision import annotate_calibration

            own_max = self._tower_max(payload.get("tower_level"))
            vision = self._make_vision(deck, own_max=own_max)
            bgr, rect = vision.grab_once()
            if bgr is None:
                raise ValueError("Janela do Clash Royale / Google Play Games nao encontrada. Abra uma partida ou Campo de Treino.")
            analyzed = vision.analyzer.analyze(bgr)
            self.calibration.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.calibration / "preview.png"), annotate_calibration(bgr, analyzed))
            vs = (analyzed.get("record") or {}).get("visual_state") or {}
            msg = (
                f"Arena calibrada ({rect[2]}x{rect[3]}) · Elixir: {vs.get('elixir', '—')} · "
                f"Mao: {', '.join(vs.get('hand') or [])} · Tropas: {len(analyzed.get('detections') or [])}"
                if analyzed["in_game"]
                else f"Janela capturada ({rect[2]}x{rect[3]}), mas nenhuma partida ativa detectada (abra o Campo de Treino)."
            )
            res = {"ok": True, "action": kind, "message": msg, "in_game": analyzed["in_game"], "preview": "arena"}
            with self.lock:
                self._last_result = res
            return res

        if kind in ("start_overlay", "start_demo"):
            demo = kind == "start_demo"
            ckpt = self._resolve_checkpoint(payload.get("run_id", ""), deck, stage="auto")
            own_max = self._tower_max(payload.get("tower_level"))
            self.stop_overlay()
            cmd = [
                sys.executable, "-u", str(self.root / "xbow_live_overlay.py"),
                "--checkpoint", str(ckpt), "--deck", deck,
                "--own-max", *(str(v) for v in own_max),
                "--opponent-max", *(str(v) for v in own_max),
            ]
            if demo:
                cmd += ["--demo", "--no-record"]
            else:
                cmd += ["--vision-local"]
                if payload.get("record", True):
                    cmd += ["--record-dir", str(self.runs / "live_matches")]
                else:
                    cmd += ["--no-record"]
                if payload.get("ensure_deck"):
                    cmd += ["--ensure-deck"]
                    raw_slot = payload.get("deck_slot")
                    if raw_slot not in (None, "", 0, "auto"):
                        cmd += ["--deck-slot", str(int(raw_slot))]
            proc = subprocess.Popen(
                cmd, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            with self.lock:
                self._overlay_proc = proc
                self._overlay_meta = {
                    "mode": "demo" if demo else "live",
                    "deck": deck,
                    "checkpoint": ckpt.relative_to(self.runs).as_posix(),
                }
                self._overlay_log.clear()
                self._overlay_log.append(f"Iniciado ({'Demo' if demo else 'Google Play Games ao vivo'}) · PID {proc.pid}")
            Thread(target=self._pipe_reader, args=(proc, self._overlay_log), daemon=True).start()
            return {"ok": True, "action": kind, "pid": proc.pid, "message": f"Overlay iniciado (PID {proc.pid})."}

        if kind == "start_learn":
            with self.lock:
                if self._poll_proc("_learn_proc"):
                    raise ValueError("O aprendizado continuo (AWR) ja esta em andamento.")
            ckpt = self._resolve_checkpoint(payload.get("run_id", ""), deck, stage="auto")
            ref_ckpt = self._resolve_checkpoint(payload.get("run_id", ""), deck, stage="bc")
            matches_dir = self.runs / "live_matches"
            if not matches_dir.is_dir() or not list(matches_dir.glob("*.jsonl")):
                raise ValueError("Nenhuma partida gravada em runs/live_matches/ ainda. Jogue com o Overlay gravando primeiro.")
            out_ckpt = ckpt.parent / "ppo_live.pt"
            cmd = [
                sys.executable, "-u", str(self.root / "xbow_live_learn.py"),
                "--checkpoint", str(ckpt), "--reference", str(ref_ckpt),
                "--matches-dir", str(matches_dir), "--output", str(out_ckpt), "--deck", deck,
            ]
            proc = subprocess.Popen(
                cmd, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            with self.lock:
                self._learn_proc = proc
                self._learn_exit = None
                self._learn_log.clear()
                self._learn_log.append(f"Aprendizado continuo AWR iniciado (PID {proc.pid}) -> {out_ckpt.name}")
            Thread(target=self._pipe_reader, args=(proc, self._learn_log), daemon=True).start()
            return {"ok": True, "action": kind, "pid": proc.pid, "message": "Treino continuo AWR iniciado em segundo plano."}

        raise ValueError(f"Acao desconhecida: {kind}")
