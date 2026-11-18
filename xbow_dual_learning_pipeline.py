"""Train X-Bow Control on human replays and with RoyaleSim self-play."""

import argparse
from collections import Counter
import math
from pathlib import Path

import torch

from xbow_environment import battle_env
from xbow_metrics import TrainingStopped, check_training_stop, training_journal
from xbow_replays import DECKS, winning_replays
from xbow_training import replay_validation, train_clone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", choices=DECKS, default="xbow_30")
    parser.add_argument("--parquet-path", type=Path, help="Arquivo ou pasta de replays IL_Replay")
    parser.add_argument("--scan", action="store_true", help="Conta partidas vencedoras sem treinar")
    parser.add_argument("--matches", type=int, default=60, help="Limite de partidas reconstruidas")
    parser.add_argument("--disk-frames", action="store_true", help="Guarda quadros temporários em disco")
    parser.add_argument("--holdout-checkpoint", type=Path, help="Reusa a validação de um BC anterior")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bc-checkpoint", type=Path, help="Checkpoint para iniciar PPO")
    parser.add_argument("--resume-ppo", action="store_true", help="Continua a partir de output/ppo.pt")
    parser.add_argument("--ppo-updates", type=int, default=0)
    parser.add_argument("--rollout-games", type=int, default=4)
    parser.add_argument("--kl-coef", type=float, default=0.15)
    parser.add_argument("--evaluate", type=int, default=0, metavar="GAMES")
    parser.add_argument("--evaluate-checkpoint", type=Path, help="Compara um checkpoint PPO salvo com o BC")
    parser.add_argument("--replay-eval", action="store_true",
                        help="Avalia cartas e posicionamento nos replays reservados pelo BC")
    args = parser.parse_args()
    if args.scan:
        if args.parquet_path is None:
            parser.error("--scan requer --parquet-path")
        counts = Counter()
        for _ in winning_replays(args.parquet_path, args.deck, counts):
            pass
        print(f"{args.deck}: {dict(counts)}")
        return
    if not args.parquet_path and not args.bc_checkpoint:
        parser.error("informe --parquet-path ou --bc-checkpoint")
    if not (args.parquet_path or args.ppo_updates or args.evaluate):
        parser.error("selecione treino com --ppo-updates ou avaliacao com --evaluate")
    if args.parquet_path and args.bc_checkpoint and not args.replay_eval:
        parser.error("--parquet-path treina um BC novo; --bc-checkpoint carrega um BC existente")
    if args.holdout_checkpoint and (not args.parquet_path or args.replay_eval):
        parser.error("--holdout-checkpoint requer treino de imitação com --parquet-path")
    if args.resume_ppo and (not args.bc_checkpoint or not args.ppo_updates):
        parser.error("--resume-ppo requer --bc-checkpoint e --ppo-updates")
    if args.evaluate_checkpoint and (not (args.evaluate or args.replay_eval) or args.ppo_updates):
        parser.error("--evaluate-checkpoint requer --evaluate ou --replay-eval sem --ppo-updates")
    if args.replay_eval and (not args.bc_checkpoint or not args.parquet_path
                             or args.ppo_updates or args.evaluate):
        parser.error("--replay-eval requer --bc-checkpoint e --parquet-path")
    if min(args.matches - 1, args.epochs, args.batch_size, args.threads, args.rollout_games) < 1:
        parser.error("informe pelo menos 2 partidas e valores positivos para os demais limites")
    if not math.isfinite(args.kl_coef) or min(args.ppo_updates, args.kl_coef, args.evaluate) < 0:
        parser.error("PPO, KL e avaliacao devem ser nao negativos")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    output = args.output or Path("runs") / args.deck
    bc_path = args.bc_checkpoint or output / "bc.pt"
    with training_journal(output, args.deck) as report:
        run(args, device, output, bc_path, report)


def run(args, device, output, bc_path, report):
    check_training_stop()
    if args.replay_eval:
        report("replay_validation")
        env = battle_env(args.deck)
        try:
            replay_validation(env, args, device, output)
        finally:
            env.close()
        return
    if args.parquet_path:
        env = battle_env(args.deck)
        try:
            train_clone(env, args, device, bc_path, report)
        finally:
            env.close()
        print(f"Checkpoint BC: {bc_path}", flush=True)
    if args.ppo_updates or args.evaluate:
        check_training_stop()
        from xbow_selfplay import run_selfplay
        run_selfplay(args, device, bc_path, output, report)


if __name__ == "__main__":
    try:
        main()
    except TrainingStopped:
        print("Parada solicitada pelo painel. Checkpoints salvos permanecem disponíveis.", flush=True)
        raise SystemExit(130) from None
