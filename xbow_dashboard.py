"""Serve the local X-Bow training monitor and its run reports."""

import argparse
from datetime import datetime, timezone
import errno
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
from threading import Timer
from urllib.parse import parse_qs, urlsplit
import webbrowser

from royalegym import RustEngine

from xbow_dashboard_jobs import ACTIVE, BusyError, DashboardJobs, DECKS
from xbow_arena import STAGES, live_data
from xbow_environment import RIVAL_DECKS, obs_profile_for_architecture
from xbow_metrics import read_run_journal
from xbow_replays import CARD_NAMES, DECKS as X_BOW_DECKS
from xbow_video_catalog import VideoCatalog
from xbow_live_controller import LiveGameController
from xbow_play_session import ACTIVE_SESSIONS, DECK_LABELS, card_pt_name, list_available_models


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
JOBS = DashboardJobs(ROOT)
LIVE_CTRL = LiveGameController(ROOT)
VIDEOS = VideoCatalog(RUNS / ".replay-sources" / "alphaclash-manifest.jsonl")
ASSETS = {
    "/": ("index.html", "text/html"),
    "/cartas-decks": ("catalog.html", "text/html"),
    "/dashboard.css": ("dashboard.css", "text/css"),
    "/dashboard.js": ("dashboard.js", "text/javascript"),
    "/arena.js": ("arena.js", "text/javascript"),
    "/videos.js": ("videos.js", "text/javascript"),
    "/play.js": ("play.js", "text/javascript"),
    "/catalog.js": ("catalog.js", "text/javascript"),
}


@lru_cache(maxsize=1)
def card_catalog():
    cards = RustEngine().cards()
    decks = X_BOW_DECKS | RIVAL_DECKS
    return {
        "cards": [
            {"name": card.name, "label": card_pt_name(card.name), "elixir": card.elixir,
             "kind": card.card_kind, "flying": card.flying}
            for card in cards
        ],
        "decks": [
            {"id": deck, "label": DECK_LABELS[deck].partition(" (")[0],
             "cards": [CARD_NAMES[key] for key in decks[deck]]}
            for deck in DECK_LABELS
        ],
    }


def json_file(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _: None)
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def latest_evaluation(path, selfplay):
    reports = [
        (modified(path / "selfplay.json"), selfplay.get("evaluation")),
        (modified(path / "evaluation.json"), json_file(path / "evaluation.json").get("evaluation")),
    ]
    ppo_time = modified(path / "ppo.pt")
    for written, data in sorted(reports, key=lambda item: item[0], reverse=True):
        if data and written >= ppo_time:
            return data
    return {}


def modified(path):
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


@lru_cache(maxsize=64)
def _cached_checkpoint_meta(path_str, mtime_ns, size):
    import torch

    try:
        saved = torch.load(path_str, map_location="meta", weights_only=True)
    except Exception:
        return {}
    arch = int(saved.get("architecture", 1))
    arch_cfg = saved.get("arch_config") or {}
    obs_profile = obs_profile_for_architecture(arch, arch_cfg.get("obs_profile"))
    metrics = saved.get("metrics") or {}
    trainer = saved.get("trainer") or {}
    pfsp_wr = trainer.get("pfsp_wr")
    if isinstance(pfsp_wr, dict):
        pfsp_wr = {str(k): round(float(v), 4) for k, v in pfsp_wr.items()}
    else:
        pfsp_wr = None
    spec = list(saved.get("spec") or ())
    return {
        "architecture": arch,
        "obs_profile": obs_profile,
        "channels": int(spec[0]) if spec else None,
        "cnn_width": arch_cfg.get("channels"),
        "spec": spec,
        "mirror_augmentation": metrics.get("mirror_augmentation"),
        "competitive_only": metrics.get("competitive_only"),
        "reward_profile": metrics.get("reward_profile") or trainer.get("reward_profile"),
        "pfsp_wr": pfsp_wr,
        "live_finetune": metrics.get("live_finetune"),
    }


def read_run_model_info(path):
    info = {}
    for stage in ("bc", "ppo"):
        ckpt = path / f"{stage}.pt"
        try:
            st = ckpt.stat()
        except OSError:
            continue
        meta = _cached_checkpoint_meta(str(ckpt), st.st_mtime_ns, st.st_size)
        for key, val in meta.items():
            if val is not None:
                info[key] = val
    return info


@lru_cache(maxsize=128)
def _cached_live_match(path_str, mtime_ns, size):
    try:
        lines = Path(path_str).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    if not lines:
        return None
    header, footer = {}, {}
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if obj.get("kind") == "header" and not header:
            header = obj
        elif obj.get("kind") == "footer":
            footer = obj
    result = footer.get("result", "unknown")
    frames = int(footer.get("frames") or max(0, len(lines) - 2))
    return {
        "file": Path(path_str).name,
        "deck": header.get("deck", "xbow_30"),
        "result": result,
        "frames": frames,
        "started": header.get("started", ""),
        "ended": footer.get("ended", ""),
    }


def live_matches_summary(runs_dir):
    folder = runs_dir / "live_matches"
    summary = {"count": 0, "wins": 0, "losses": 0, "draws": 0, "total_frames": 0, "matches": []}
    if not folder.is_dir():
        return summary
    entries = []
    for path in sorted(folder.glob("*.jsonl"), reverse=True):
        try:
            st = path.stat()
        except OSError:
            continue
        row = _cached_live_match(str(path), st.st_mtime_ns, st.st_size)
        if not row:
            continue
        summary["count"] += 1
        summary["total_frames"] += row["frames"]
        if row["result"] == "win":
            summary["wins"] += 1
        elif row["result"] == "loss":
            summary["losses"] += 1
        elif row["result"] == "draw":
            summary["draws"] += 1
        entries.append(row)
    summary["matches"] = entries[:20]
    return summary


def experiment(path):
    report = json_file(path / "selfplay.json")
    events = list(read_run_journal(path))
    updates = {row["update"]: row for row in report.get("updates", [])
               if isinstance(row, dict) and isinstance(row.get("update"), int)}
    epochs, split, phase = [], {}, {}
    progress = {}
    for event in events:
        kind = event.get("event")
        if kind == "replays" and event.get("done") == 0:
            updates.clear()
            epochs, split = [], {}
        elif kind == "bc_start":
            epochs = []
            split = {key: event[key] for key in ("training", "validation")}
        elif kind == "bc_epoch":
            epochs.append({"epoch": event["done"], **event["metrics"]})
        elif kind == "ppo_start" and not event.get("resumed"):
            updates.clear()
        elif kind == "ppo_update":
            updates[event["done"]] = event["metrics"]
        phase = {key: val for key, val in event.items() if key != "metrics"}
        if kind not in ("started", "finished", "stopped", "failed"):
            progress = phase

    # Completed reports on disk are authoritative. The journal fills gaps during a run.
    current_report = json_file(path / "selfplay.json")
    validation_path = path / "replay_validation.json"
    current_validation = json_file(validation_path)
    if modified(validation_path) < modified(path / "ppo.pt"):
        current_validation.pop("ppo", None)
    if modified(validation_path) < modified(path / "bc.pt"):
        current_validation.pop("bc", None)
    checkpoints = {
        phase_name: (path / f"{phase_name}.pt").is_file() for phase_name in ("bc", "ppo")
    }
    if not (events or current_report or current_validation or any(checkpoints.values())):
        return None
    evaluation = latest_evaluation(path, current_report)
    if phase.get("event") in ("replays", "bc_start", "bc_batch", "bc_epoch", "ppo_start", "ppo_match", "ppo_game",
                              "ppo_update", "evaluation_start", "evaluation_pair"):
        evaluation = {}
    return {
        "id": path.relative_to(RUNS).as_posix(), "deck": path.name,
        "group": path.parent.relative_to(RUNS).as_posix(),
        "updates": [updates[number] for number in sorted(updates)],
        "evaluation": evaluation, "validation": current_validation,
        "phase": phase, "progress": progress, "epochs": epochs, "split": split,
        "checkpoints": checkpoints, "model_info": read_run_model_info(path),
    }


def recommended_summary():
    """Lightweight real-time snapshot for the Recommended Training card in /treinamento."""
    v_models = ROOT / "models" / "vision"
    v_ds = ROOT / "runs" / "vision_dataset"
    custom_pt = v_models / "custom_yolo.pt"
    pub_onnx = v_models / "units_M_480x352.onnx"
    active_vision = (
        "custom_yolo.pt" if custom_pt.is_file()
        else ("units_M_480x352.onnx" if pub_onnx.is_file() else "hp_bar_only")
    )
    stats = {}
    stats_file = v_ds / "stats.json"
    if stats_file.is_file():
        try:
            stats = json.loads(stats_file.read_text(encoding="utf-8"))
        except Exception:
            stats = {}
    live_frames = int(stats.get("frames_saved") or 0)
    if live_frames == 0 and (v_ds / "live" / "images").is_dir():
        live_frames = len(list((v_ds / "live" / "images").glob("*.jpg")))
    sprites = int(stats.get("sprites_harvested") or 0)
    verified = int(stats.get("verified_play_labels") or 0)
    disc_count = 0
    disc_file = RUNS / "discovered_decks.json"
    if disc_file.is_file():
        try:
            disc_count = len((json.loads(disc_file.read_text(encoding="utf-8")).get("decks")) or {})
        except Exception:
            disc_count = 0
    return {
        "active_vision": active_vision,
        "custom_yolo_ready": custom_pt.is_file(),
        "public_yolo_ready": pub_onnx.is_file(),
        "vision_live_frames": live_frames,
        "vision_sprites": sprites,
        "vision_verified_labels": verified,
        "discovered_decks": disc_count,
        "simulator_cards": 119,
    }


def dashboard_data():
    runs = []
    if RUNS.is_dir():
        for folder, names, _ in RUNS.walk():
            names[:] = [name for name in names if not name.startswith((".", "frames-"))]
            if folder.name not in DECKS:
                continue
            names.clear()
            if result := experiment(folder):
                runs.append(result)
    runs.sort(key=lambda row: (row["group"] != "calibrated", row["id"]))
    inventory = JOBS.inventory.snapshot()
    total_cpus = os.cpu_count() or 4
    max_threads = max(1, total_cpus - 1)
    max_workers = max(1, min(12, total_cpus // 2))
    return {
        "runs": runs, "job": JOBS.current(),
        "joint_training": "shared_match",
        "continuous_training": True,
        "continuous_parallel": True,
        "system": {
            "cpu_threads": total_cpus,
            "max_threads": max_threads,
            "max_workers": max_workers,
        },
        "live_matches": live_matches_summary(RUNS),
        "recommended": recommended_summary(),
        "parts": inventory["parts"], "inventory": inventory,
        "refreshed": datetime.now(timezone.utc).isoformat(),
    }


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlsplit(self.path)
        path = url.path
        if path in ("/treinamento", "/treino"):
            self.send_response(302)
            self.send_header("Location", "/#treino")
            self.end_headers()
        elif path == "/api/status":
            payload = json.dumps(dashboard_data(), ensure_ascii=False).encode("utf-8")
            self.respond(payload, "application/json")
        elif path in ASSETS:
            name, media = ASSETS[path]
            self.respond((ROOT / "dashboard" / name).read_bytes(), media)
        elif path == "/api/arena":
            self.arena(parse_qs(url.query))
        elif path == "/api/play/models":
            self.reply_json({"models": list_available_models(), "decks": DECK_LABELS}, 200)
        elif path == "/api/catalog":
            self.reply_json(card_catalog(), 200)
        elif path == "/api/play/state":
            session = ACTIVE_SESSIONS.get_session()
            if not session:
                self.reply_json({"active": False}, 200)
            else:
                self.reply_json({"active": True, "state": session.snapshot()}, 200)
        elif path == "/api/live/status":
            deck = parse_qs(url.query).get("deck", ["xbow_30"])[0]
            self.reply_json(LIVE_CTRL.status(deck), 200)
        elif path == "/api/live/preview":
            kind = parse_qs(url.query).get("kind", ["deck"])[0]
            data_bytes = LIVE_CTRL.preview_bytes(kind)
            if data_bytes is None:
                self.send_error(404)
            else:
                self.respond_raw(data_bytes, "image/png", 200)
        elif path == "/api/videos":
            query = parse_qs(url.query)
            try:
                page = int(query.get("page", ["1"])[0])
                deck = query.get("deck", [""])[0]
                if deck and deck not in DECKS:
                    raise ValueError
                wins = query.get("wins", ["0"])[0] == "1"
                search = query.get("q", [""])[0].lower().strip()[:80]
                payload = VIDEOS.page(page, deck, wins, search)
            except ValueError:
                self.reply_json({"error": "Filtro do catálogo inválido."}, 400)
            except OSError:
                self.reply_json({"error": "O catálogo de vídeos está indisponível."}, 500)
            else:
                self.reply_json(payload, 200)
        else:
            self.send_error(404)

    def arena(self, query):
        def param(name, default=""):
            return query.get(name, [default])[0]

        root = RUNS.resolve()
        stage, deck = param("phase"), param("deck")
        run_id = param("run")
        try:
            run = (root / run_id).resolve() if run_id else None
            if (param("stage") != "live" or stage not in STAGES or deck not in DECKS or
                    (run and (not run.is_relative_to(root) or run.name != deck))):
                raise ValueError
        except (ValueError, OSError):
            self.reply_json({"error": "Seleção de arena inválida."}, 400)
            return
        job = JOBS.current(with_log=False)
        active = bool(run and job and job["status"] in ACTIVE and job["run_id"] == run_id)
        boards = live_data(JOBS.store / job["id"] / "arena-live", stage) if active else []
        self.reply_json({"active": active, "boards": boards}, 200)

    def do_POST(self):
        route = urlsplit(self.path).path
        valid_routes = (
            "/api/job/start", "/api/job/stop",
            "/api/play/start", "/api/play/act", "/api/play/pause",
            "/api/play/speed", "/api/play/stop",
            "/api/live/action",
        )
        if route not in valid_routes:
            self.reply_json({"error": "Ação desconhecida."}, 404)
            return
        allowed = {f"http://{host}:{self.server.server_port}" for host in ("127.0.0.1", "localhost")}
        if self.headers.get("Origin") not in allowed:
            self.reply_json({"error": "Abra o painel no endereço local para controlar o treino."}, 403)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if self.headers.get_content_type() != "application/json" and size > 0:
                raise ValueError("Configuração inválida.")
            payload = json.loads(self.rfile.read(size)) if size > 0 else {}
            if not isinstance(payload, dict):
                raise ValueError("Configuração inválida.")

            if route.startswith("/api/play/"):
                self.handle_play_post(route, payload)
                return

            if route == "/api/live/action":
                result = LIVE_CTRL.action(str(payload.get("action") or ""), payload)
                self.reply_json(result, 200)
                return

            job = JOBS.start(payload) if route.endswith("/start") else JOBS.stop(payload.get("id"))
        except BusyError as exc:
            self.reply_json({"error": str(exc)}, 409)
        except (ValueError, UnicodeError) as exc:
            self.reply_json({"error": str(exc)}, 400)
        except OSError:
            self.reply_json({"error": "O painel não conseguiu acessar os arquivos do treinamento."}, 500)
        else:
            self.reply_json({"job": job}, 202)

    def handle_play_post(self, route, payload):
        try:
            if route == "/api/play/start":
                human_deck = payload.get("human_deck", "hog_26")
                model_id = payload.get("model_id", "scripted:xbow")
                ai_deck = payload.get("ai_deck", "hog_26" if model_id == "rival:live" else "xbow_30")
                speed = float(payload.get("speed", 1.0))
                session = ACTIVE_SESSIONS.start_session(human_deck, ai_deck, model_id, speed)
                self.reply_json({"ok": True, "session_id": session.session_id, "state": session.snapshot()}, 200)
                return

            session_id = payload.get("session_id")
            session = ACTIVE_SESSIONS.get_session()
            if session is None or (session_id is not None and session.session_id != session_id):
                self.reply_json({"ok": False, "error": "Esta partida já terminou ou foi substituída."}, 409)
                return
            if route == "/api/play/act":
                slot, x, y = (payload.get(key) for key in ("slot", "x", "y"))
                if not all(type(value) is int for value in (slot, x, y)):
                    raise ValueError("Selecione uma carta e uma casa válidas.")
                result = session.queue_human_action(slot, x, y, payload.get("card_id"))
                self.reply_json(result, 200 if result["ok"] else 400)
            elif route == "/api/play/pause":
                paused = payload.get("paused")
                if paused is not None and type(paused) is not bool:
                    raise ValueError("Estado de pausa inválido.")
                self.reply_json({"ok": True, "paused": session.toggle_pause(paused)}, 200)
            elif route == "/api/play/speed":
                speed = session.set_speed(float(payload.get("speed", 1.0)))
                self.reply_json({"ok": True, "speed": speed}, 200)
            else:
                stopped = ACTIVE_SESSIONS.stop_session(session.session_id)
                self.reply_json({"ok": stopped}, 200 if stopped else 409)
        except (TypeError, ValueError) as exc:
            self.reply_json({"ok": False, "error": str(exc)}, 400)
        except Exception as exc:
            print(f"Falha no painel de batalha: {exc}", flush=True)
            self.reply_json({"ok": False, "error": "O modo batalha encontrou um erro."}, 500)

    def reply_json(self, value, status):
        self.respond(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json", status)

    def respond_raw(self, data, media, status=200):
        self.send_response(status)
        self.send_header("Content-Type", media)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        try:
            self.end_headers()
            self.wfile.write(data)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def respond(self, data, media, status=200):
        self.send_response(status)
        self.send_header("Content-Type", f"{media}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        try:
            self.end_headers()
            self.wfile.write(data)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def log_message(self, _format, *_args):
        pass


class DashboardServer(ThreadingHTTPServer):
    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.allow_reuse_address = False
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def bind_dashboard(port, *, fallback_port=False):
    try:
        return DashboardServer(("127.0.0.1", port), DashboardHandler)
    except OSError as exc:
        unavailable = {errno.EADDRINUSE}
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            unavailable.update((errno.WSAEADDRINUSE, errno.WSAEACCES))
        if not fallback_port or exc.errno not in unavailable:
            raise
        return DashboardServer(("127.0.0.1", 0), DashboardHandler)


def main(*, open_by_default=False, fallback_port=False):
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--open", action=argparse.BooleanOptionalAction, default=open_by_default,
                        help="Abre o painel no navegador")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port deve estar entre 0 e 65535")
    torch.set_num_threads(1)
    with bind_dashboard(args.port, fallback_port=fallback_port) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Painel X-Bow Control em {url}", flush=True)
        if args.open:
            Timer(0.2, webbrowser.open, (url,)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
