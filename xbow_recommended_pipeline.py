"""One-click Recommended Training Pipeline for real-game peak performance.

Orchestrates the full Vision + Live-Meta + Hybrid Policy improvement stack with
progress reporting compatible with the X-Bow web dashboard (`progress.json`) and
graceful stop support (`XBOW_STOP_FILE`):

  Stage 1 (vision): Ensure public ONNX YOLO weights, generate synthetic arena frames
                    using harvested real sprites/backgrounds, and fine-tune `custom_yolo.pt`
                    with the mAP50 safety promotion gate.
  Stage 2 (meta):   Analyze recorded live matches (`runs/live_matches/*.jsonl`) to identify
                    weakest real-world enemy archetypes and seed PFSP priorities.
  Stage 3 (hybrid): Consolidate live wins/positive plays via AWR + Reverse KL and drill the
                    6-archetype RoyaleSim meta + league snapshots via PPO.
  Stage 4 (gate):   Run matched-seed A/B evaluation against the pre-cycle incumbent and
                    atomically promote `ppo_live.pt` (creating `.bak.pt` backup) if approved.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

from xbow_hybrid_loop import analyze_live_meta, run_hybrid_cycle
from xbow_metrics import TrainingStopped, check_training_stop
from xbow_replays import DECKS
from xbow_vision_train import generate_synthetic_dataset, train_vision_yolo, vision_status
from xbow_vision_yolo import (
    DEFAULT_VISION_DATASET_DIR,
    DEFAULT_VISION_MODELS_DIR,
    ensure_public_yolo_weights,
)


def _write_progress(run_dir: Path, event: str, **payload) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "progress.json"
    data = {
        "event": event,
        "time": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def run_recommended_pipeline(
    *,
    deck: str,
    run_dir: Path,
    matches_dir: Path = Path("runs/live_matches"),
    vision_dataset_dir: Path = DEFAULT_VISION_DATASET_DIR,
    vision_models_dir: Path = DEFAULT_VISION_MODELS_DIR,
    train_vision: bool = True,
    vision_synthetic_count: int = 300,
    vision_epochs: int = 10,
    vision_imgsz: int = 480,
    vision_batch: int = 8,
    ppo_updates: int = 4,
    rollout_games: int = 10,
    eval_games_per_rival: int = 2,
    parallel_games: int = 4,
    threads: int = 6,
    seed: int = 42,
) -> dict:
    began = perf_counter()
    run_dir = Path(run_dir)
    bc_path = run_dir / "bc.pt"
    live_ckpt = run_dir / "ppo_live.pt"
    base_ppo = run_dir / "ppo.pt"
    source_ckpt = live_ckpt if live_ckpt.is_file() else (base_ppo if base_ppo.is_file() else bc_path)
    if not source_ckpt.is_file():
        raise ValueError(f"Nenhum checkpoint encontrado em {run_dir} (bc.pt, ppo.pt ou ppo_live.pt).")

    print(
        f"=== PIPELINE RECOMENDADA ({deck}) ===\n"
        f"Experimento: {run_dir}\n"
        f"Checkpoint inicial: {source_ckpt.name} | Âncora BC: {bc_path.name if bc_path.is_file() else 'auto'}",
        flush=True,
    )

    # ---------- STAGE 1: VISION YOLO ----------
    check_training_stop()
    _write_progress(run_dir, "rec_vision", done=0, total=2, detail="Verificando pesos YOLO públicos…")
    vision_report: dict = {"skipped": False}
    try:
        pub_paths = ensure_public_yolo_weights(vision_models_dir)
        print(f"[1/4 Visão] Pesos públicos prontos: {list(pub_paths.keys())}", flush=True)
    except Exception as exc:
        print(f"[1/4 Visão] Aviso ao verificar pesos públicos: {exc}", flush=True)

    vstat = vision_status(vision_dataset_dir, vision_models_dir)
    live_frames = int(vstat["dataset"]["live_frames"])
    sprites = int(vstat["dataset"]["harvested_sprites"])
    if train_vision:
        check_training_stop()
        _write_progress(
            run_dir, "rec_vision", done=1, total=2,
            detail=f"Gerando {vision_synthetic_count} frames sintéticos + treinando YOLO ({live_frames} quadros reais, {sprites} sprites)…",
        )
        print(
            f"[1/4 Visão] Gerando {vision_synthetic_count} quadros sintéticos "
            f"(usando {sprites} sprites reais coletados + {live_frames} quadros ao vivo)…",
            flush=True,
        )
        syn = generate_synthetic_dataset(
            vision_dataset_dir, count=vision_synthetic_count, deck=deck, seed=seed,
        )
        check_training_stop()
        print(
            f"[1/4 Visão] Treinando YOLOv8n por {vision_epochs} épocas (imgsz={vision_imgsz}, batch={vision_batch})…",
            flush=True,
        )
        vtrain = train_vision_yolo(
            vision_dataset_dir,
            vision_models_dir,
            epochs=vision_epochs,
            imgsz=vision_imgsz,
            batch=vision_batch,
        )
        vision_report = {"skipped": False, "synthetic": syn, "train": vtrain}
        print(
            f"[1/4 Visão] Concluído: mAP50={vtrain['map50']:.3f} | promovido={vtrain['promoted']} | "
            f"detector ativo={vision_status(vision_dataset_dir, vision_models_dir)['models']['active_auto']}",
            flush=True,
        )
    else:
        vision_report = {"skipped": True, "reason": "desativado pelo usuário", "status": vstat["models"]}
        print("[1/4 Visão] Treino YOLO pulado (mantendo detector ativo atual).", flush=True)
    _write_progress(run_dir, "rec_vision", done=2, total=2, detail="Visão verificada e pronta.")

    # ---------- STAGE 2: LIVE META ANALYSIS ----------
    check_training_stop()
    _write_progress(run_dir, "rec_meta", done=0, total=1, detail="Analisando partidas reais em runs/live_matches/…")
    meta = analyze_live_meta(matches_dir, deck=deck)
    hardest = sorted(meta["pfsp_seed"].items(), key=lambda kv: kv[1])[:3]
    hardest_txt = ", ".join(f"{name} ({wr:.2f})" for name, wr in hardest)
    print(
        f"[2/4 Meta Real] {meta['matches_scanned']} partidas analisadas "
        f"({meta['classified_matches']} classificadas por YOLO) · Foco PFSP: {hardest_txt}",
        flush=True,
    )
    _write_progress(run_dir, "rec_meta", done=1, total=1, detail=f"Foco PFSP: {hardest_txt}")

    # ---------- STAGE 3 & 4: HYBRID AWR + SYNTHETIC PPO + A/B GATE ----------
    check_training_stop()
    _write_progress(
        run_dir, "rec_hybrid", done=0, total=ppo_updates + 1,
        detail=f"Executando AWR + {ppo_updates} updates PPO ({rollout_games} jogos/update) nos 6 arquétipos…",
    )
    print(
        f"[3/4 Política Híbrida] Iniciando Live AWR + {ppo_updates} updates PPO "
        f"({rollout_games} jogos/update, threads={threads}) + Gate A/B…",
        flush=True,
    )
    hybrid_report = run_hybrid_cycle(
        deck=deck,
        checkpoint=source_ckpt,
        bc_checkpoint=bc_path if bc_path.is_file() else None,
        output=live_ckpt,
        matches_dir=matches_dir,
        skip_awr=False,
        ppo_updates=ppo_updates,
        rollout_games=rollout_games,
        eval_games_per_rival=eval_games_per_rival,
        parallel_games=parallel_games,
        threads=threads,
        seed=seed,
    )
    ab = hybrid_report["ab_eval"]
    _write_progress(
        run_dir, "rec_gate", done=1, total=1,
        detail=f"Gate A/B: {ab['reason']} · promovido={hybrid_report['promoted']}",
    )
    print(
        f"[4/4 Gate A/B] Incumbent={ab['incumbent']['win_score']:.3f} vs "
        f"Candidato={ab['candidate']['win_score']:.3f} ({ab['reason']}) · "
        f"Promovido={hybrid_report['promoted']}",
        flush=True,
    )

    elapsed = round(perf_counter() - began, 2)
    final_vision = vision_status(vision_dataset_dir, vision_models_dir)
    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "deck": deck,
        "run_dir": str(run_dir),
        "elapsed_seconds": elapsed,
        "active_vision_model": final_vision["models"]["active_auto"],
        "active_policy": str(live_ckpt if live_ckpt.is_file() else source_ckpt),
        "vision": vision_report,
        "hybrid": hybrid_report,
    }
    _write_progress(
        run_dir, "rec_done", done=1, total=1,
        detail=f"Pipeline concluída em {elapsed:.0f}s · Gate A/B: {'promovido' if hybrid_report['promoted'] else 'mantido incumbent'}",
    )
    print(
        f"=== CONCLUÍDO EM {elapsed:.1f}s ===\n"
        f"Visão ativa: {summary['active_vision_model']}\n"
        f"Política pronta para jogo real: {summary['active_policy']}",
        flush=True,
    )
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", choices=DECKS, default="xbow_30")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--matches-dir", type=Path, default=Path("runs/live_matches"))
    parser.add_argument("--vision-dataset-dir", type=Path, default=DEFAULT_VISION_DATASET_DIR)
    parser.add_argument("--vision-models-dir", type=Path, default=DEFAULT_VISION_MODELS_DIR)
    parser.add_argument("--train-vision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vision-synthetic-count", type=int, default=300)
    parser.add_argument("--vision-epochs", type=int, default=10)
    parser.add_argument("--vision-imgsz", type=int, default=480)
    parser.add_argument("--vision-batch", type=int, default=8)
    parser.add_argument("--ppo-updates", type=int, default=4)
    parser.add_argument("--rollout-games", type=int, default=10)
    parser.add_argument("--eval-games-per-rival", type=int, default=2)
    parser.add_argument("--parallel-games", type=int, default=4)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        run_recommended_pipeline(
            deck=args.deck,
            run_dir=args.run_dir,
            matches_dir=args.matches_dir,
            vision_dataset_dir=args.vision_dataset_dir,
            vision_models_dir=args.vision_models_dir,
            train_vision=args.train_vision,
            vision_synthetic_count=args.vision_synthetic_count,
            vision_epochs=args.vision_epochs,
            vision_imgsz=args.vision_imgsz,
            vision_batch=args.vision_batch,
            ppo_updates=args.ppo_updates,
            rollout_games=args.rollout_games,
            eval_games_per_rival=args.eval_games_per_rival,
            parallel_games=args.parallel_games,
            threads=args.threads,
            seed=args.seed,
        )
    except TrainingStopped:
        print("Pipeline recomendada interrompida com segurança a pedido do usuário.", flush=True)
        sys.exit(130)


if __name__ == "__main__":
    main()
