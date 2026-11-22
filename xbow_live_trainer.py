"""Autonomous live Clash Royale trainer for unranked Classic 1v1 / Trainer Camp.

Executes the full closed loop on Google Play Games PC:
  LOBBY -> VERIFY CLASSIC 1V1 (hard anti-trophy guard) -> QUEUING -> IN_MATCH (play + record)
        -> POST_MATCH (click OK) -> PERIODIC AWR CONTINUAL LEARNING (hot-reload) -> NEXT MATCH

Safety & UX guarantees:
  - Uses Win32 PostMessageW background clicks by default (NEVER moves the user's cursor).
  - If the game window ignores background clicks (3 consecutive no-op clicks), stops cleanly
    with status `needs_cursor_permission` so the user can explicitly confirm cursor-restore mode.
  - Hard anti-trophy guard: refuses to click Battle unless Classic 1v1 (blue crossed-swords shield)
    is visually verified, or switches into Classic 1v1 via the mode picker first.
  - Records `played_action` explicitly for exact AWR fine-tuning anchored to `bc.pt` via Reverse KL.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from threading import Event
import time

import cv2
import numpy as np
import torch

from xbow_capture_bridge import CaptureBridge
from xbow_live_learn import finetune_from_live, load_live_matches
from xbow_live_overlay import LiveMatchRecorder, OverlayBridge, infer_match_result
from xbow_play_session import card_pt_name
from xbow_policy_service import describe_action
from xbow_public_feed import PublicFeed
from xbow_replays import DECKS
from xbow_screen_vision import (
    LiveVisionServer,
    UI_BATTLE_BUTTON,
    UI_MODE_SELECTOR_BUTTON,
    UI_POST_MATCH_OK_BUTTON,
    UI_TAB_BATTLE,
    annotate_calibration,
    detect_post_match_result,
    detect_screen_state,
    dispatch_click,
    ensure_deck_by_clicks,
    norm_to_screen_px,
    play_card_on_window,
    trim_viewport_bounds,
)
from xbow_training import MirrorSpec, load_checkpoint, pack_id_planes, save_checkpoint


def _frame_changed(before_bgr: np.ndarray, after_bgr: np.ndarray, threshold: float = 3.2) -> bool:
    """Return True if the trimmed game viewport changed meaningfully after a UI click."""
    a = cv2.resize(before_bgr, (142, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(after_bgr, (142, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
    return float(np.abs(a - b).mean()) >= threshold


class LiveAutoTrainer:
    """State-machine autonomous player & continual AWR learner for Classic 1v1 / Trainer Camp."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        deck: str = "xbow_30",
        reference: Path | None = None,
        output: Path | None = None,
        mode: str = "classic_1v1",
        max_games: int = 20,
        learn_every: int = 5,
        temperature: float = 0.45,
        dry_run: bool = False,
        click_mode: str = "background",
        ensure_deck: bool = True,
        deck_slot: int | None = None,
        own_max: tuple[int, int, int] = (4824, 3052, 3052),
        opponent_max: tuple[int, int, int] = (4824, 3052, 3052),
        record_dir: Path = Path("runs/live_matches"),
        templates_dir: Path = Path("calibration/templates"),
        yolo_model: str | Path = "auto",
        collect_vision_dir: Path | None = Path("runs/vision_dataset"),
        vision_retrain_every: int = 0,
        sim_reinforce_every: int = 0,
        sim_ppo_updates: int = 2,
        sim_rollout_games: int = 8,
        show_overlay: bool = False,
        state_file: Path | None = None,
        play_cooldown_s: float = 0.90,
        loop_sleep_s: float = 0.45,
        low_elixir_hold: float = 6.0,
        low_elixir_min_play_prob: float = 0.30,
        threads: int = 6,
        vision=None,
        click_fn=None,
        stop_event: Event | None = None,
    ):
        self.checkpoint = Path(checkpoint)
        self.reference = Path(reference) if reference is not None else (self.checkpoint.parent / "bc.pt")
        self.output = Path(output) if output is not None else (self.checkpoint.parent / "ppo_live.pt")
        self.deck = deck
        self.mode = mode
        self.max_games = max(1, int(max_games))
        self.learn_every = max(1, int(learn_every))
        self.temperature = float(temperature)
        self.dry_run = bool(dry_run)
        self.click_mode = click_mode if click_mode in ("background", "cursor_restore") else "background"
        self.ensure_deck = bool(ensure_deck)
        self.deck_slot = deck_slot
        self.own_max = tuple(own_max)
        self.opponent_max = tuple(opponent_max)
        self.record_dir = Path(record_dir)
        self.templates_dir = Path(templates_dir)
        self.yolo_model = yolo_model
        self.collect_vision_dir = Path(collect_vision_dir) if collect_vision_dir is not None else None
        self.vision_retrain_every = max(0, int(vision_retrain_every))
        self.sim_reinforce_every = max(0, int(sim_reinforce_every))
        self.sim_ppo_updates = max(1, int(sim_ppo_updates))
        self.sim_rollout_games = max(2, int(sim_rollout_games))
        self.show_overlay = bool(show_overlay)
        self.overlay: OverlayBridge | None = None
        self.state_file = Path(state_file) if state_file is not None else None
        self.play_cooldown_s = float(play_cooldown_s)
        self.loop_sleep_s = float(loop_sleep_s)
        self.low_elixir_hold = float(low_elixir_hold)
        self.low_elixir_min_play_prob = float(low_elixir_min_play_prob)
        self.threads = max(1, int(threads))
        self.stop_event = stop_event or Event()
        self.click_fn = click_fn

        self.vision = vision or LiveVisionServer(
            deck, own_max=self.own_max, opponent_max=self.opponent_max,
            templates_dir=self.templates_dir, yolo_weights=self.yolo_model,
            collect_vision_dir=self.collect_vision_dir,
        )
        torch.set_num_threads(2)
        self.policy, self.saved_ckpt = load_checkpoint(self.checkpoint, self.deck, "cpu")
        self.anchor_policy = None
        if self.reference.is_file():
            self.anchor_policy, _ = load_checkpoint(self.reference, self.deck, "cpu")
        self.feed = PublicFeed(self.deck, obs_profile=self.policy.obs_profile)
        self.mirror_spec = MirrorSpec.from_env(self.feed.env)
        self.bridge = CaptureBridge(self.deck, self.own_max, self.opponent_max, allow_mid_match=True)
        self.recorder = LiveMatchRecorder(self.record_dir, self.deck, str(self.checkpoint))

        self.status = "starting"
        self.state = "LOBBY"
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.awr_updates = 0
        self.sim_updates = 0
        self._policy_mtime = self.output.stat().st_mtime if self.output.is_file() else (
            self.checkpoint.stat().st_mtime if self.checkpoint.is_file() else 0.0
        )
        self.last_action = ""
        self.message = "Iniciando sessão de treino real…"
        self.needs_cursor_permission = False
        self._deck_checked = False
        self._in_match_seen = False
        self._last_match_frame: dict | None = None
        self._last_play_ts = 0.0
        self._noop_clicks = 0
        self._bg_click_verified = False
        self._mode_attempts = 0
        self._unknown_ticks = 0
        self._recorded_paths: list[Path] = []
        self._match_enemy_cards: Counter[str] = Counter()

    def snapshot(self) -> dict:
        analyzer = getattr(self.vision, "analyzer", None)
        collector = getattr(analyzer, "collector", None) if analyzer is not None else None
        return {
            "status": self.status,
            "state": self.state,
            "mode": self.mode,
            "deck": self.deck,
            "dry_run": self.dry_run,
            "click_mode": self.click_mode,
            "vision_backend": getattr(analyzer, "vision_backend", "hp_bar_only"),
            "vision_collected": dict(collector.stats) if collector is not None else None,
            "games_played": self.games_played,
            "max_games": self.max_games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "awr_updates": self.awr_updates,
            "sim_updates": self.sim_updates,
            "learn_every": self.learn_every,
            "last_action": self.last_action,
            "message": self.message,
            "needs_cursor_permission": self.needs_cursor_permission,
            "updated": datetime.now(timezone.utc).isoformat(),
        }

    def _publish(self, msg: str | None = None):
        if msg is not None:
            self.message = msg
            print(f"[{self.state}] {msg}", flush=True)
        if self.state_file is None:
            return
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.state_file)
        except OSError:
            pass

    def _click_norm(self, rect: tuple[int, int, int, int], frame_bgr: np.ndarray,
                    norm_x: float, norm_y: float) -> tuple[int, int]:
        _cleaned, left_trim, top_trim, trimmed_w, trimmed_h = trim_viewport_bounds(frame_bgr)
        sx, sy = norm_to_screen_px(
            rect, left_trim, trimmed_w, norm_x, norm_y, top_trim=top_trim, trimmed_h=trimmed_h
        )
        if self.click_fn is not None:
            self.click_fn(sx, sy)
        else:
            hwnd = getattr(self.vision, "current_hwnd", None)
            try:
                dispatch_click(hwnd, sx, sy, mode=self.click_mode)
            except PermissionError as exc:
                self._publish(str(exc))
        return sx, sy

    def _register_click_effect(self, effective: bool, *, is_card_play: bool = False) -> bool:
        """Track background click effectiveness; request cursor permission after 3 consecutive no-ops."""
        if effective:
            self._noop_clicks = 0
            if not self.dry_run:
                self._bg_click_verified = True
            return True
        if self.dry_run or self.click_mode == "cursor_restore":
            return True
        # Once background clicks are already proven working in this session, a rejected card placement
        # (e.g. blocked tower tile or brief elixir lag) should never abort an active match.
        if is_card_play and self._bg_click_verified:
            return True
        self._noop_clicks += 1
        if self._noop_clicks >= 3:
            self.needs_cursor_permission = True
            self.status = "needs_cursor_permission"
            self.state = "PAUSED_INPUT"
            self._publish(
                "O Google Play Games ignorou 3 cliques em segundo plano (PostMessageW). "
                "Confirme no painel se deseja autorizar cliques rápidos com restauração do cursor."
            )
            return False
        return True

    def _finish_match_and_maybe_learn(self, post_match_bgr: np.ndarray | None = None):
        crown_res = detect_post_match_result(post_match_bgr) if post_match_bgr is not None else None
        result = crown_res or infer_match_result(self._last_match_frame)
        saved_path = self.recorder.finish(explicit_result=result)
        self.games_played += 1
        if result == "win":
            self.wins += 1
        elif result == "loss":
            self.losses += 1
        else:
            self.draws += 1
        if saved_path is not None:
            self._recorded_paths.append(saved_path)
        discovered_note = ""
        if self._match_enemy_cards:
            try:
                from xbow_deck_discovery import DiscoveredDeckStore
                entry = DiscoveredDeckStore().record_live_match(self._match_enemy_cards, result)
                if entry is not None:
                    discovered_note = f" · Deck rival salvo: {entry['id']} ({entry['label']})"
            except Exception:
                pass
            self._match_enemy_cards.clear()
        self._in_match_seen = False
        self._last_match_frame = None
        self.bridge.reset()
        self.vision.analyzer.reset_match()
        self._publish(
            f"Partida {self.games_played}/{self.max_games} encerrada ({result}) · "
            f"Placar sessão: {self.wins}V-{self.losses}D-{self.draws}E{discovered_note}."
        )
        ran_awr = False
        if not self.dry_run and self.games_played % self.learn_every == 0:
            self._run_continual_awr()
            ran_awr = True
        if (
            not self.dry_run
            and self.sim_reinforce_every > 0
            and self.games_played % self.sim_reinforce_every == 0
        ):
            self._run_hybrid_sim_reinforce(skip_awr=ran_awr)
        if (
            not self.dry_run
            and self.vision_retrain_every > 0
            and self.collect_vision_dir is not None
            and self.games_played % self.vision_retrain_every == 0
        ):
            self._run_continual_vision()
        self._maybe_reload_policy_from_disk()

    def _maybe_reload_policy_from_disk(self) -> bool:
        """Hot-reload policy weights if an external hybrid loop promoted a newer `output` checkpoint."""
        if not self.output.is_file():
            return False
        try:
            mtime = self.output.stat().st_mtime
            if mtime <= self._policy_mtime + 1e-3:
                return False
            new_policy, new_ckpt = load_checkpoint(self.output, self.deck, "cpu")
            if new_policy.obs_profile != self.policy.obs_profile:
                return False
            self.policy = new_policy
            self.saved_ckpt = new_ckpt
            self._policy_mtime = mtime
            self._publish(f"Hot-reload automático: novos pesos carregados de {self.output.name}!")
            return True
        except Exception:
            return False

    def _run_hybrid_sim_reinforce(self, *, skip_awr: bool = False):
        self.state = "LEARNING_SIM"
        self._publish(
            f"Reforço sintético híbrido no RoyaleSim ({self.sim_ppo_updates} updates × "
            f"{self.sim_rollout_games} jogos focados nos decks difíceis ao vivo)…"
        )
        try:
            from xbow_hybrid_loop import run_hybrid_cycle
            src = self.output if self.output.is_file() else self.checkpoint
            report = run_hybrid_cycle(
                deck=self.deck,
                checkpoint=src,
                bc_checkpoint=self.reference if self.reference.is_file() else None,
                output=self.output,
                matches_dir=self.record_dir,
                skip_awr=skip_awr,
                ppo_updates=self.sim_ppo_updates,
                rollout_games=self.sim_rollout_games,
                eval_games_per_rival=2,
                threads=self.threads,
            )
            ab = report["ab_eval"]
            if report.get("promoted"):
                self.sim_updates += 1
                self._maybe_reload_policy_from_disk()
            self._publish(
                f"Ciclo híbrido #{self.sim_updates} concluído: promovido={report['promoted']} · "
                f"{ab['reason']} ({report['elapsed_seconds']}s)."
            )
        except Exception as exc:
            self._publish(f"Aviso no reforço sintético híbrido (mantendo pesos atuais): {exc}")
        finally:
            torch.set_num_threads(2)

    def _run_continual_vision(self):
        self.state = "LEARNING_VISION"
        self._publish("Atualizando modelo YOLO com quadros coletados + arena sintética…")
        try:
            from xbow_vision_train import generate_synthetic_dataset, train_vision_yolo
            generate_synthetic_dataset(self.collect_vision_dir, count=120, deck=self.deck)
            report = train_vision_yolo(self.collect_vision_dir, epochs=4, imgsz=480, batch=8)
            reloaded = False
            analyzer = getattr(self.vision, "analyzer", None)
            if report.get("promoted") and analyzer is not None and hasattr(analyzer, "reload_vision_model"):
                reloaded = analyzer.reload_vision_model()
            self._publish(
                f"YOLO atualizado (mAP50={report['map50']:.3f}, promovido={report['promoted']}, "
                f"hot-reload={reloaded})."
            )
        except Exception as exc:
            self._publish(f"Aviso no treino contínuo de visão (mantendo detector atual): {exc}")

    def _run_continual_awr(self):
        self.state = "LEARNING_AWR"
        paths = sorted(self.record_dir.glob("*.jsonl")) if self.record_dir.is_dir() else []
        if not paths:
            self._publish("Sem arquivos gravados para AWR; continuando sessão.")
            return
        self._publish(f"Executando aprendizado contínuo AWR com {len(paths)} partidas gravadas…")
        try:
            torch.set_num_threads(self.threads)
            samples, data_summary = load_live_matches(paths[-40:], self.deck, self.policy.obs_profile)
            if len(samples) < 8:
                self._publish("Amostras insuficientes para AWR (<8 quadros); pulando atualização.")
                return
            if data_summary.get("wins", 0) == 0 and all(s.ret <= 0.05 for s in samples):
                self._publish("AWR adiado: aguardando ao menos 1 vitória ou lance positivo para não reforçar erros.")
                return
            if self.output.is_file():
                shutil.copy2(self.output, self.output.with_suffix(".bak.pt"))
            train_metrics = finetune_from_live(
                self.policy, samples, mirror_spec=self.mirror_spec,
                anchor_policy=self.anchor_policy, epochs=2, batch_size=64,
                lr=5e-5, kl_coef=0.15, awr_beta=0.5,
            )
            merged_metrics = {
                **(self.saved_ckpt.get("metrics") or {}),
                "live_finetune": {**data_summary, **train_metrics},
            }
            save_checkpoint(
                self.output, self.policy, self.deck, self.saved_ckpt.get("phase", "ppo"),
                merged_metrics, self.saved_ckpt.get("trainer"),
            )
            if self.output.is_file():
                self._policy_mtime = self.output.stat().st_mtime
            self.awr_updates += 1
            self._publish(
                f"AWR #{self.awr_updates} concluído (loss={train_metrics['policy_loss']}, "
                f"KL={train_metrics['kl_to_previous']}) · pesos atualizados em memória!"
            )
        except Exception as exc:
            self._publish(f"Aviso no AWR (continuando com pesos atuais): {exc}")
        finally:
            torch.set_num_threads(2)

    def step_once(self) -> bool:
        """Execute one tick of the state machine. Returns False when session should stop."""
        if self.stop_event.is_set() or self.games_played >= self.max_games:
            return False

        bgr, rect = self.vision.grab_once()
        if bgr is None or rect is None:
            self.state = "WAITING_WINDOW"
            self._publish("Aguardando janela do Clash Royale / Google Play Games…")
            return True

        screen = detect_screen_state(bgr, templates_dir=self.templates_dir)
        kind = screen["state"]
        # Refresh calibration/preview.png periodically so the web panel can show what the trainer sees.
        now_pv = time.monotonic()
        if not hasattr(self, "_last_preview_ts") or (now_pv - self._last_preview_ts) >= 2.5:
            self._last_preview_ts = now_pv
            try:
                pv = Path("calibration/preview.png")
                pv.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(pv), bgr)
            except OSError:
                pass

        if self.overlay is not None and kind != "in_match":
            _, lt, tt, tw, th = trim_viewport_bounds(bgr)
            self.overlay.push({
                "lobby": True,
                "window_rect": rect,
                "viewport_trim": (lt, tt, tw, th),
                "status_text": f"X-Bow AI · {self.state} ({self.games_played}/{self.max_games} · {self.wins}V-{self.losses}D)",
            })

        # ---------- IN_MATCH ----------
        if kind == "in_match":
            self.state = "IN_MATCH"
            self._in_match_seen = True
            self._mode_attempts = 0
            self._unknown_ticks = 0
            captured = self.vision.analyzer.analyze(bgr)
            if not captured.get("in_game"):
                return True
            for det in captured.get("detections") or ():
                if det.get("team") == "enemy" and det.get("source") in ("public_onnx_yolo", "custom_yolo"):
                    cname = det.get("class_name")
                    if cname:
                        self._match_enemy_cards[str(cname)] += 1
            if not hasattr(self, "_last_annot_ts") or (time.monotonic() - self._last_annot_ts) >= 4.0:
                self._last_annot_ts = time.monotonic()
                try:
                    ap = Path("calibration/preview_match_annotated.png")
                    ap.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(ap), annotate_calibration(bgr, captured))
                except OSError:
                    pass
            try:
                pub = self.bridge.push(captured)
            except (KeyError, TypeError, ValueError) as exc:
                self._publish(f"Quadro ignorado: {exc}")
                return True
            if pub is None:
                return True
            self._last_match_frame = pub
            obs = self.feed.update(pub)
            action, logp, value = self.policy.decide(obs, "cpu", temperature=self.temperature)
            # Elixir discipline (matching 100%-WR simulator benchmark): below `low_elixir_hold` (6.0)
            # with NO enemy threat on our half of the arena, reject low-confidence stochastic tail plays
            # so the AI naturally saves to 6-9 elixir for X-Bow/Tesla & real defensive reactions.
            before_elix = float(pub["own_elixir_milli"]) / 1000.0
            if action > 0 and self.low_elixir_min_play_prob > 0.0 and before_elix < self.low_elixir_hold:
                enemy_on_our_side = any(
                    u.get("side") == "opponent" and float((u.get("position") or (0, 32))[1]) < 16.5
                    for u in (pub.get("units") or ())
                )
                if not enemy_on_our_side:
                    id_p = pack_id_planes(obs) if self.policy.architecture >= 3 else None
                    id_t = torch.as_tensor(id_p).unsqueeze(0) if id_p is not None else None
                    m_t = torch.as_tensor(obs["action_mask"], dtype=torch.bool).unsqueeze(0)
                    with torch.no_grad():
                        lg, _ = self.policy.logits_and_value(
                            torch.as_tensor(obs["spatial"]).unsqueeze(0),
                            torch.as_tensor(obs["vector"]).unsqueeze(0),
                            m_t, id_planes=id_t,
                        )
                        p_play = 1.0 - float(torch.softmax(lg, dim=-1)[0, 0].item())
                    if p_play < self.low_elixir_min_play_prob:
                        action = 0
            sug = {
                "action": action, **describe_action(action),
                "probability": float(np.exp(logp)), "value": round(float(value), 3),
            }
            if sug["kind"] == "play_card":
                sug["card"] = pub["hand"][sug["slot"]]
            hud = self.feed.public_summary()

            played_action = 0
            hand_conf = float(((captured.get("record") or {}).get("visual_state") or {}).get("hand_confidence") or 0.0)
            now = time.monotonic()
            if (
                sug["kind"] == "play_card"
                and hand_conf >= 0.25
                and (now - self._last_play_ts) >= self.play_cooldown_s
            ):
                card_label = card_pt_name(sug["card"])
                self.last_action = f"{card_label} ({sug['x']},{sug['y']})"
                if self.dry_run:
                    self._publish(f"[SIMULAÇÃO] Jogaria {self.last_action} ({int(sug['probability']*100)}%)")
                    played_action = action
                    self._last_play_ts = now
                else:
                    before_elix = float(pub["own_elixir_milli"]) / 1000.0
                    before_hand = list(pub["hand"])
                    hwnd = getattr(self.vision, "current_hwnd", None)
                    slot_px, tile_px = play_card_on_window(
                        hwnd, rect, bgr, int(sug["slot"]), float(sug["x"]), float(sug["y"]),
                        click_mode=self.click_mode, select_delay_s=0.10, click_fn=self.click_fn,
                    )
                    self._last_play_ts = time.monotonic()
                    played_action = action
                    if hasattr(self.vision.analyzer, "record_own_play"):
                        self.vision.analyzer.record_own_play(
                            sug["card"], float(sug["x"]), float(sug["y"])
                        )
                    if self.loop_sleep_s > 0:
                        time.sleep(min(0.40, self.loop_sleep_s))
                        after_bgr, _ = self.vision.grab_once()
                        if after_bgr is not None:
                            after_cap = self.vision.analyzer.analyze(after_bgr)
                            after_vs = (after_cap.get("record") or {}).get("visual_state") or {}
                            after_elix = float(after_vs.get("elixir") or before_elix)
                            after_hand = list(after_vs.get("hand") or before_hand)
                            elix_dropped = (before_elix - after_elix) >= 0.45
                            hand_changed = (after_hand != before_hand)
                            effective = elix_dropped or hand_changed
                            self._publish(
                                f"Jogou {self.last_action} · elixir {before_elix:.1f}→{after_elix:.1f}"
                                + (" ✓" if effective else f" (sem efeito · px {slot_px}→{tile_px})")
                            )
                            if self.click_mode == "background" and before_elix >= 5.0:
                                if not self._register_click_effect(effective, is_card_play=True):
                                    return False
                        else:
                            self._publish(f"Jogou {self.last_action} (elixir {before_elix:.1f})")
                    else:
                        self._publish(f"Jogou {self.last_action} (elixir {before_elix:.1f})")

            step_sug = {**sug, "played_action": played_action}
            self.recorder.record(pub, step_sug, hud)
            if self.overlay is not None:
                _, lt, tt, tw, th = trim_viewport_bounds(bgr)
                own_e = float(hud.get("own_elixir", 0.0))
                opp_e = float(hud.get("enemy_elixir", 0.0))
                self.overlay.push({
                    "lobby": False,
                    "window_rect": rect,
                    "viewport_trim": (lt, tt, tw, th),
                    "detections": captured.get("detections") or [],
                    "suggestion": {**sug, "executed": played_action != 0},
                    "hud": hud,
                    "status_text": (
                        f"J{self.games_played + 1}/{self.max_games} ({self.wins}V-{self.losses}D) · "
                        f"Elixir {own_e:.1f} | Rival ~{opp_e:.1f} · Obj {len(captured.get('detections') or ())}"
                    ),
                })
            return True

        # If we were just in a match and it ended, click OK on post-match first (when live) then finalize & run AWR!
        if self._in_match_seen and kind in ("post_match", "lobby", "mode_picker"):
            post_bgr = bgr if kind == "post_match" else None
            if kind == "post_match" and not self.dry_run:
                self.state = "POST_MATCH"
                ok_pt = screen["ok_button"] or UI_POST_MATCH_OK_BUTTON
                self._publish("Partida encerrada — clicando em OK…")
                self._click_norm(rect, bgr, ok_pt[0], ok_pt[1])
                if self.loop_sleep_s > 0:
                    time.sleep(min(0.7, self.loop_sleep_s * 1.5))
            self._finish_match_and_maybe_learn(post_match_bgr=post_bgr)
            if self.games_played >= self.max_games:
                return False
            return True

        # ---------- POST_MATCH ----------
        if kind == "post_match":
            self.state = "POST_MATCH"
            self._unknown_ticks = 0
            if self.dry_run:
                self._publish("[SIMULAÇÃO] Tela de fim de partida detectada (clicaria em OK).")
                return True
            ok_pt = screen["ok_button"] or UI_POST_MATCH_OK_BUTTON
            self._publish("Tela de fim de partida detectada — clicando em OK…")
            self._click_norm(rect, bgr, ok_pt[0], ok_pt[1])
            if self.loop_sleep_s > 0:
                time.sleep(min(0.7, self.loop_sleep_s * 1.5))
                after_bgr, _ = self.vision.grab_once()
                if after_bgr is not None and not self._register_click_effect(_frame_changed(bgr, after_bgr)):
                    return False
            return True

        # ---------- MODE_PICKER (Modos clássicos) ----------
        if kind == "mode_picker":
            self.state = "SELECT_1V1"
            self._unknown_ticks = 0
            card_hit = screen.get("classic_1v1_card")
            if self.mode != "classic_1v1" or card_hit is None:
                self._publish("Tela de modos aberta — aguardando seleção.")
                return True
            if self.dry_run:
                self._publish(f"[SIMULAÇÃO] Clicaria no card '1x1 Clássico' (conf={card_hit[2]:.2f}).")
                return True
            self._publish(f"Selecionando '1x1 Clássico' (conf={card_hit[2]:.2f})…")
            self._click_norm(rect, bgr, card_hit[0], card_hit[1])
            if self.loop_sleep_s > 0:
                time.sleep(min(0.7, self.loop_sleep_s * 1.5))
                after_bgr, _ = self.vision.grab_once()
                if after_bgr is not None and not self._register_click_effect(_frame_changed(bgr, after_bgr)):
                    return False
            return True

        # ---------- LOBBY ----------
        if kind == "lobby":
            self.state = "LOBBY"
            self._unknown_ticks = 0
            if self.ensure_deck and not self._deck_checked and not self.dry_run:
                self.state = "CHECK_DECK"
                self._publish(f"Validando deck {self.deck} antes de iniciar…")
                res = ensure_deck_by_clicks(
                    self.vision, self.deck, deck_slot=self.deck_slot,
                    templates_dir=self.templates_dir,
                    delay=max(0.0, min(0.5, self.loop_sleep_s)),
                    click_fn=self.click_fn,
                )
                self._deck_checked = True
                if not res.get("ok") and not res.get("already_in_game"):
                    self._publish(f"Aviso no deck: {res.get('error')} (prosseguindo na aba Batalha).")
                return True

            if self.mode == "trainer_camp":
                self._publish("Modo Campo de Treino: abra uma partida contra o Treinador para a IA jogar e gravar.")
                return True

            # Hard anti-trophy guard for Classic 1v1!
            if not screen["classic_1v1_selected"]:
                self._mode_attempts += 1
                if self._mode_attempts > 4:
                    self.status = "paused_unsafe_mode"
                    self.state = "PAUSED_SAFETY"
                    self._publish(
                        "TRAVA ANTI-TROFÉUS: não foi possível confirmar o escudo azul do '1x1 Clássico'. "
                        "Selecione '1x1 Clássico' manualmente no jogo para nunca jogar por troféus."
                    )
                    return False
                if self.dry_run:
                    self._publish("[SIMULAÇÃO] Escudo '1x1 Clássico' não está selecionado — clicaria no seletor de modo.")
                    return True
                self._publish(
                    f"Modo atual não é '1x1 Clássico' (tentativa {self._mode_attempts}/4) — abrindo seletor de modos…"
                )
                self._click_norm(rect, bgr, UI_MODE_SELECTOR_BUTTON[0], UI_MODE_SELECTOR_BUTTON[1])
                if self.loop_sleep_s > 0:
                    time.sleep(min(0.7, self.loop_sleep_s * 1.5))
                    after_bgr, _ = self.vision.grab_once()
                    if after_bgr is not None and not self._register_click_effect(_frame_changed(bgr, after_bgr)):
                        return False
                return True

            # Classic 1v1 verified -> click Battle!
            self._mode_attempts = 0
            if self.dry_run:
                self._publish(f"[SIMULAÇÃO] '1x1 Clássico' confirmado! Clicaria em Batalha ({self.games_played + 1}/{self.max_games}).")
                return True
            self.state = "QUEUING"
            self._publish(f"'1x1 Clássico' confirmado! Iniciando partida {self.games_played + 1}/{self.max_games}…")
            b_pt = screen["battle_button"] or UI_BATTLE_BUTTON
            self._click_norm(rect, bgr, b_pt[0], b_pt[1])
            if self.loop_sleep_s > 0:
                time.sleep(min(0.9, self.loop_sleep_s * 2.0))
                after_bgr, _ = self.vision.grab_once()
                if after_bgr is not None and not self._register_click_effect(_frame_changed(bgr, after_bgr)):
                    return False
            return True

        # ---------- UNKNOWN / TRANSITION ----------
        if self.overlay is not None:
            _, lt, tt, tw, th = trim_viewport_bounds(bgr)
            self.overlay.push({
                "lobby": True,
                "window_rect": rect,
                "viewport_trim": (lt, tt, tw, th),
                "status_text": f"X-Bow AI · {self.state} ({self.games_played}/{self.max_games} · {self.wins}V-{self.losses}D)",
            })
        self._unknown_ticks += 1
        if self._unknown_ticks >= 12 and not self.dry_run:
            # Gentle recovery: return to Battle tab if stuck on Decks/Clan screen.
            self._unknown_ticks = 0
            self._publish("Tela fora do lobby — retornando à aba Batalha…")
            self._click_norm(rect, bgr, UI_TAB_BATTLE[0], UI_TAB_BATTLE[1])
        return True

    def run(self) -> int:
        self.status = "running"
        if self.show_overlay:
            self.overlay = OverlayBridge(window_title=getattr(self.vision, "window_title", None))
            if self.overlay.start():
                self._publish("Overlay visual ativado por cima do jogo (caixas YOLO + decisões, click-through).")
            else:
                self._publish(f"Overlay visual indisponível ({self.overlay.error or 'sem display'}); seguindo sem overlay.")
                self.overlay = None
        self._publish(
            f"Treino real iniciado: modo={self.mode}, meta={self.max_games} jogos, "
            f"AWR a cada {self.learn_every}, cliques={self.click_mode}"
            + (" [SIMULAÇÃO DRY-RUN]" if self.dry_run else "")
        )
        try:
            while not self.stop_event.is_set():
                keep_going = self.step_once()
                if not keep_going:
                    break
                if self.loop_sleep_s > 0:
                    # Keep ~0.50s cadence during IN_MATCH matching training's decision_ms=500 (2 Hz)
                    # so we never over-sample P(play) and drain elixir to 0-4.
                    self.stop_event.wait(self.loop_sleep_s)
        except KeyboardInterrupt:
            self.status = "stopped"
            self._publish("Sessão interrompida pelo usuário.")
        finally:
            if self.overlay is not None:
                self.overlay.close()
                self.overlay = None
            if self._in_match_seen:
                self.recorder.finish()
            self.feed.close()
            if self.status == "running":
                self.status = "finished" if self.games_played >= self.max_games else "stopped"
                self._publish(
                    f"Sessão finalizada ({self.games_played} partidas · "
                    f"{self.wins}V-{self.losses}D-{self.draws}E · {self.awr_updates} updates AWR)."
                )
        return 3 if self.needs_cursor_permission else 0


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", choices=DECKS, default="xbow_30")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mode", choices=("classic_1v1", "trainer_camp"), default="classic_1v1")
    parser.add_argument("--max-games", type=int, default=20)
    parser.add_argument("--learn-every", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--click-mode", choices=("background", "cursor_restore"), default="background")
    parser.add_argument("--ensure-deck", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deck-slot", type=int, choices=range(1, 6), default=None)
    parser.add_argument("--own-max", nargs=3, type=int, default=(4824, 3052, 3052), metavar=("REI", "ESQ", "DIR"))
    parser.add_argument("--opponent-max", nargs=3, type=int, default=(4824, 3052, 3052), metavar=("REI", "ESQ", "DIR"))
    parser.add_argument("--record-dir", type=Path, default=Path("runs/live_matches"))
    parser.add_argument("--templates-dir", type=Path, default=Path("calibration/templates"))
    parser.add_argument("--yolo-model", default="auto",
                        help="Modelo YOLO: 'auto' (custom_yolo.pt > public ONNX > HP), 'public', 'none', ou caminho .pt/.onnx")
    parser.add_argument("--collect-vision-dir", type=Path, default=Path("runs/vision_dataset"),
                        help="Pasta para auto-coleta contínua de quadros/sprites YOLO ao vivo")
    parser.add_argument("--no-collect-vision", action="store_true",
                        help="Desativa a auto-coleta de quadros/sprites YOLO durante o treino real")
    parser.add_argument("--vision-retrain-every", type=int, default=0,
                        help="Opcional: retreina custom_yolo.pt a cada N partidas (0 = manual via xbow_vision_train.py)")
    parser.add_argument("--sim-reinforce-every", type=int, default=0,
                        help="Opcional: roda ciclo híbrido no RoyaleSim + gate A/B a cada N partidas (0 = manual/daemon)")
    parser.add_argument("--sim-ppo-updates", type=int, default=2)
    parser.add_argument("--sim-rollout-games", type=int, default=8)
    parser.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=True,
                        help="Mostra overlay transparente click-through com caixas YOLO e decisões da IA sobre o jogo")
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    trainer = LiveAutoTrainer(
        checkpoint=args.checkpoint,
        deck=args.deck,
        reference=args.reference,
        output=args.output,
        mode=args.mode,
        max_games=args.max_games,
        learn_every=args.learn_every,
        temperature=args.temperature,
        dry_run=args.dry_run,
        click_mode=args.click_mode,
        ensure_deck=args.ensure_deck,
        deck_slot=args.deck_slot,
        own_max=tuple(args.own_max),
        opponent_max=tuple(args.opponent_max),
        record_dir=args.record_dir,
        templates_dir=args.templates_dir,
        yolo_model=args.yolo_model,
        collect_vision_dir=None if args.no_collect_vision else args.collect_vision_dir,
        vision_retrain_every=args.vision_retrain_every,
        sim_reinforce_every=args.sim_reinforce_every,
        sim_ppo_updates=args.sim_ppo_updates,
        sim_rollout_games=args.sim_rollout_games,
        show_overlay=args.overlay,
        state_file=args.state_file,
        threads=args.threads,
    )
    sys.exit(trainer.run())


if __name__ == "__main__":
    main()
