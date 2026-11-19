"""Run one local training session at a time, with a persistent log and stop request."""

from contextlib import contextmanager, suppress
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from stat import S_ISREG
import subprocess
import sys
from tempfile import NamedTemporaryFile
from threading import Lock, Thread
from time import monotonic, sleep
from uuid import uuid4


DECKS = {"xbow_30", "icebow"}
ACTIVE = {"starting", "running", "stopping"}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    for attempt in range(10):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == 9:
                raise
            sleep(0.02)


def save_json(path, value):
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(value, ensure_ascii=False).encode("utf-8"))
    try:
        for attempt in range(10):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                sleep(0.02)
    finally:
        temporary.unlink(missing_ok=True)


def process_alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 0x102
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


class BusyError(ValueError):
    pass


class ReplayInventory:
    """Count unique candidates by deck and reuse scans of unchanged partitions."""

    def __init__(self, dataset, cache):
        self.dataset = dataset
        self.cache = cache
        self.lock = Lock()
        try:
            cached = load_json(cache)
        except (OSError, ValueError):
            cached = {}
        self.entries = cached if isinstance(cached, dict) else {}
        self.ready = None
        self.counts = {}
        self.checking = False
        self.failed = None
        self.retry_at = 0.0

    def manifest(self):
        files = {}
        nested = self.dataset / "replays"
        directory = nested if nested.is_dir() else self.dataset
        for path in sorted(directory.glob("part-*.parquet")):
            try:
                metadata = path.stat()
            except FileNotFoundError:
                continue
            if S_ISREG(metadata.st_mode):
                name = path.relative_to(self.dataset).as_posix()
                files[name] = [metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns]
        return files

    def snapshot(self):
        with self.lock:
            try:
                files = self.manifest()
            except OSError:
                return {"parts": len(self.ready or {}), "status": "unavailable"}
            result = {"parts": len(files)}
            if files == self.ready:
                return {**result, "status": "ready", "counts": self.counts}
            if self.checking:
                return {**result, "status": "checking"}
            if files == self.failed and monotonic() < self.retry_at:
                return {**result, "status": "unavailable"}
            self.checking = True
            Thread(target=self.scan, args=(files,), daemon=True).start()
            return {**result, "status": "checking"}

    def scan(self, files):
        entries = {}
        try:
            from xbow_replays import replay_tags

            unique = {deck: set() for deck in DECKS}
            for name, fingerprint in files.items():
                entry = self.entries.get(name, {})
                if entry.get("fingerprint") != fingerprint:
                    entry = {"fingerprint": fingerprint, "tags": replay_tags(self.dataset / name)}
                entries[name] = entry
                for deck, tags in entry["tags"].items():
                    unique[deck].update(tags)

            if entries != self.entries:
                try:
                    self.cache.parent.mkdir(parents=True, exist_ok=True)
                    save_json(self.cache, entries)
                except OSError:
                    pass
            with self.lock:
                self.entries = entries
                self.counts = {deck: len(tags) for deck, tags in unique.items()}
                self.ready = files
                self.failed = None
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"Inventário de replays: {exc}", file=sys.stderr, flush=True)
            with self.lock:
                self.entries.update(entries)
                self.failed = files
                self.retry_at = monotonic() + 5
        finally:
            with self.lock:
                self.checking = False


class DashboardJobs:
    def __init__(self, root):
        self.root = root
        self.runs = (root / "runs").resolve()
        self.store = self.runs / ".portal"
        self.inventory = ReplayInventory(root / "replays", self.runs / ".replay-inventory-v1.json")
        self.lock = Lock()
        self.child = None

    @contextmanager
    def exclusive(self):
        self.store.mkdir(parents=True, exist_ok=True)
        with self.lock, (self.store / "control.lock").open("a+b") as stream:
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def current(self, with_log=True):
        child = self.child
        if child is not None and child.poll() is not None and self.child is child:
            self.child = None
        try:
            pointer = load_json(self.store / "latest.json")
            folder = self.store / pointer["id"]
            config = load_json(folder / "job.json")
            state = load_json(folder / "state.json")
        except FileNotFoundError:
            return None
        if state["status"] in ACTIVE:
            if not process_alive(pointer["pid"]):
                state = load_json(folder / "state.json")
                if state["status"] in ACTIVE:
                    state = {"status": "interrupted"}
            elif (folder / "stop").exists():
                state["status"] = "stopping"
        result = {key: config[key] for key in ("id", "run_id", "deck", "mode", "parameters")}
        result.update(state)
        if with_log:
            try:
                with (folder / "training.log").open("rb") as stream:
                    stream.seek(0, os.SEEK_END)
                    stream.seek(max(0, stream.tell() - 16000))
                    result["log"] = stream.read().decode("utf-8", errors="replace").splitlines()[-70:]
            except FileNotFoundError:
                result["log"] = []
        return result

    def start(self, request):
        if not isinstance(request, dict):
            raise ValueError("Configuração de treinamento inválida.")
        with self.exclusive():
            current = self.current(with_log=False)
            if current and current["status"] in ACTIVE:
                raise BusyError("Já existe um treinamento em andamento.")
            return self.launch(request)

    def launch(self, request):
        deck, mode = request.get("deck"), request.get("mode")
        if not isinstance(deck, str) or deck not in DECKS or mode not in ("new", "resume"):
            raise ValueError("Escolha um deck e um tipo de treinamento válidos.")
        coevolve = request.get("coevolve", False)
        if type(coevolve) is not bool or (coevolve and (deck != "xbow_30" or mode != "resume")):
            raise ValueError("O treino conjunto está disponível ao retomar X-Bow 3.0.")
        limits = {"updates": (1, 100), "games": (1, 32), "evaluate": (0, 48)}
        if mode == "new":
            inventory = self.inventory.snapshot()
            if inventory["status"] != "ready":
                raise ValueError("Aguarde a contagem dos replays do deck.")
            available = inventory["counts"][deck]
            if available < 2:
                raise ValueError("Este deck precisa de pelo menos 2 replays candidatos para iniciar o treino.")
            if type(request.get("matches")) is not int or not 2 <= request["matches"] <= available:
                raise ValueError(f"Escolha entre 2 e {available} replays candidatos para este deck.")
            limits.update(matches=(2, available), epochs=(1, 30))
        params = {}
        continuous = request.get("continuous", False)
        if type(continuous) is not bool or (continuous and not coevolve):
            raise ValueError("O treino contínuo requer o treino conjunto da X-Bow 3.0.")
        if continuous:
            limits["games"] = (1, 8)
        for key, (low, high) in limits.items():
            if continuous and key in ("updates", "evaluate"):
                params[key] = 0
                continue
            value = request.get(key, 4) if continuous and key == "games" else request.get(key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Valor fora do intervalo permitido para {key}.")
            params[key] = value
        if coevolve and not continuous:
            params["games"] = 1
        job_id = uuid4().hex
        args = [
            sys.executable, "-u", str(self.root / "xbow_dual_learning_pipeline.py"),
            "--deck", deck, "--ppo-updates", str(params["updates"]),
            "--rollout-games", str(params["games"]), "--evaluate", str(params["evaluate"]),
            "--kl-coef", "0.3",
        ]
        if mode == "new":
            dataset = self.root / "replays"
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output = self.runs / f"assistido-{stamp}-{job_id[:5]}" / deck
            args += ["--parquet-path", str(dataset), "--matches", str(params["matches"]),
                     "--epochs", str(params["epochs"]), "--disk-frames"]
        else:
            run_id = request.get("run_id")
            if not isinstance(run_id, str):
                raise ValueError("Escolha o experimento que deseja retomar.")
            output = (self.runs / run_id).resolve()
            if (not output.is_relative_to(self.runs) or output.name != deck
                    or not all((output / f"{stage}.pt").is_file() for stage in ("bc", "ppo"))):
                raise ValueError("O experimento selecionado precisa ter os checkpoints de imitação e PPO.")
            args += ["--bc-checkpoint", str(output / "bc.pt"), "--resume-ppo"]
        if coevolve:
            args.append("--coevolve")
        params["coevolve"] = coevolve
        if continuous:
            args += ["--continuous", "--parallel-games", str(params["games"])]
        params["continuous"] = continuous
        args += ["--output", str(output)]
        folder = self.store / job_id
        folder.mkdir()
        save_json(folder / "job.json", {
            "id": job_id, "run_id": output.relative_to(self.runs).as_posix(),
            "deck": deck, "mode": mode, "parameters": params, "command": args,
        })
        save_json(folder / "state.json", {"status": "starting", "started": timestamp()})
        kwargs = (
            {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
            if os.name == "nt" else {"start_new_session": True}
        )
        self.child = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), str(folder)],
            cwd=self.root, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, **kwargs,
        )
        save_json(self.store / "latest.json", {"id": job_id, "pid": self.child.pid})
        return self.current()

    def stop(self, job_id):
        with self.exclusive():
            job = self.current(with_log=False)
            if not job or job["id"] != job_id or job["status"] not in ACTIVE:
                raise BusyError("Esta execução já terminou ou foi substituída.")
            (self.store / job_id / "stop").touch()
            return self.current()


def worker(folder):
    config = load_json(folder / "job.json")
    state = load_json(folder / "state.json")
    state["status"] = "running"
    save_json(folder / "state.json", state)
    live_folder = folder / "arena-live"
    env = {**os.environ, "XBOW_STOP_FILE": str(folder / "stop"), "PYTHONUTF8": "1",
           "XBOW_ARENA": "1", "XBOW_ARENA_DIR": str(live_folder)}
    with (folder / "training.log").open("ab") as log:
        try:
            result = subprocess.run(config["command"], env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        except OSError as exc:
            log.write((str(exc) + "\n").encode("utf-8"))
            state.update(status="failed", ended=timestamp())
        else:
            status = "finished" if result.returncode == 0 else "failed"
            if result.returncode == 130 and (folder / "stop").exists():
                status = "stopped"
            state.update(status=status, ended=timestamp(), exit_code=result.returncode)
    save_json(folder / "state.json", state)
    for path in live_folder.iterdir() if live_folder.is_dir() else ():
        if path.suffix in {".json", ".tmp"}:
            with suppress(OSError):
                path.unlink(missing_ok=True)
    with suppress(OSError):
        live_folder.rmdir()


if __name__ == "__main__":
    worker(Path(sys.argv[1]))
