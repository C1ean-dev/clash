"""Serve the local X-Bow training monitor and its run reports."""

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Timer
from urllib.parse import parse_qs, urlsplit
import webbrowser

from xbow_dashboard_jobs import BusyError, DashboardJobs, DECKS
from xbow_arena import STAGES, clip_data
from xbow_video_catalog import VideoCatalog
from xbow_play_session import ACTIVE_SESSIONS, DECK_LABELS, list_available_models


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
JOBS = DashboardJobs(ROOT)
VIDEOS = VideoCatalog(RUNS / ".replay-sources" / "alphaclash-manifest.jsonl")
ASSETS = {
    "/": ("index.html", "text/html"),
    "/dashboard.css": ("dashboard.css", "text/css"),
    "/dashboard.js": ("dashboard.js", "text/javascript"),
    "/arena.js": ("arena.js", "text/javascript"),
    "/videos.js": ("videos.js", "text/javascript"),
    "/play.js": ("play.js", "text/javascript"),
}


def json_file(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _: None)
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def journal(path):
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.endswith("\n"):
                    break
                try:
                    entry = json.loads(line, parse_constant=lambda _: None)
                except ValueError:
                    continue
                if isinstance(entry, dict) and "attempt" in entry:
                    yield entry
    except (OSError, UnicodeError):
        return


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


def experiment(path):
    report = json_file(path / "selfplay.json")
    events = list(journal(path / "progress.jsonl"))
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
    if phase.get("event") in ("replays", "bc_start", "bc_batch", "bc_epoch", "ppo_start", "ppo_game",
                              "ppo_update", "evaluation_start", "evaluation_pair"):
        evaluation = {}
    return {
        "id": path.relative_to(RUNS).as_posix(), "deck": path.name,
        "group": path.parent.relative_to(RUNS).as_posix(),
        "updates": [updates[number] for number in sorted(updates)],
        "evaluation": evaluation, "validation": current_validation,
        "phase": phase, "progress": progress, "epochs": epochs, "split": split,
        "checkpoints": checkpoints,
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
    return {
        "runs": runs, "job": JOBS.current(),
        "parts": inventory["parts"], "inventory": inventory,
        "refreshed": datetime.now(timezone.utc).isoformat(),
    }


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlsplit(self.path)
        path = url.path
        if path == "/api/status":
            payload = json.dumps(dashboard_data(), ensure_ascii=False).encode("utf-8")
            self.respond(payload, "application/json")
        elif path in ASSETS:
            name, media = ASSETS[path]
            self.respond((ROOT / "dashboard" / name).read_bytes(), media)
        elif path == "/api/arena":
            self.arena(parse_qs(url.query))
        elif path == "/api/play/models":
            self.reply_json({"models": list_available_models(), "decks": DECK_LABELS}, 200)
        elif path == "/api/play/state":
            session = ACTIVE_SESSIONS.get_session()
            if not session:
                self.reply_json({"active": False}, 200)
            else:
                self.reply_json({"active": True, "state": session.snapshot()}, 200)
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
        stage, deck = param("stage"), param("deck")
        run_id = param("run")
        try:
            run = (root / run_id).resolve() if run_id else None
            if (stage not in STAGES or deck not in DECKS or
                    (run and (not run.is_relative_to(root) or run.name != deck))):
                raise ValueError
            after = int(param("after", "-1"))
        except (ValueError, OSError):
            self.reply_json({"error": "Seleção de arena inválida."}, 400)
            return
        clip_id, known = param("clip"), param("known")
        data = clip_data(run / "arena", stage, clip_id, known, after) if run else {"clips": [], "clip": None}
        preview = not data["clips"]
        if preview:
            folder = root / ".arena-preview" / deck / "arena"
            data = clip_data(folder, stage, clip_id, known, after)
        self.reply_json({**data, "preview": preview}, 200)

    def do_POST(self):
        route = urlsplit(self.path).path
        valid_routes = (
            "/api/job/start", "/api/job/stop",
            "/api/play/start", "/api/play/act", "/api/play/pause",
            "/api/play/speed", "/api/play/stop"
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
                ai_deck = payload.get("ai_deck", "xbow_30")
                model_id = payload.get("model_id", "scripted:xbow")
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

    def respond(self, data, media, status=200):
        self.send_response(status)
        self.send_header("Content-Type", f"{media}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, _format, *_args):
        pass


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--open", action="store_true", help="Abre o painel no navegador")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler) as server:
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
