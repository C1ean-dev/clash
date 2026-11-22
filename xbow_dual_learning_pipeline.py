"""Train X-Bow Control on human replays and with RoyaleSim self-play."""

import argparse
from collections import Counter
import math
import os
from pathlib import Path

import torch

from xbow_environment import (
    DEFAULT_OBS_PROFILE, JOINT_REWARD_PROFILES, OBS_PROFILES, RIVAL_DECKS, battle_env,
    obs_profile_for_architecture,
)
from xbow_metrics import TrainingStopped, check_training_stop, training_journal
from xbow_replays import DECKS, winning_replays
from xbow_training import replay_validation, train_clone


def main():
    all_decks = tuple(DECKS) + tuple(RIVAL_DECKS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", choices=all_decks, default="xbow_30")
    parser.add_argument("--parquet-path", type=Path, help="Arquivo ou pasta de replays IL_Replay")
    parser.add_argument("--scan", action="store_true", help="Conta partidas vencedoras sem treinar")
    parser.add_argument("--matches", type=int, default=60, help="Limite de partidas reconstruidas")
    parser.add_argument("--disk-frames", action="store_true", help="Guarda quadros temporários em disco")
    parser.add_argument("--holdout-checkpoint", type=Path, help="Reusa a validação de um BC anterior")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int,
                        help="Threads do PyTorch (padrão: 2 na imitação, até 4 no self-play em CPU)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bc-checkpoint", type=Path, help="Checkpoint para iniciar PPO")
    parser.add_argument("--resume-ppo", action="store_true", help="Continua a partir de output/ppo.pt")
    parser.add_argument("--coevolve", action="store_true",
                        help="Atualiza X-Bow 3.0 e o rival após cada confronto com decks fixos")
    parser.add_argument("--ppo-updates", type=int, default=0)
    parser.add_argument("--continuous", action="store_true",
                        help="Treina os dois modelos até uma parada solicitada")
    parser.add_argument("--reward", choices=JOINT_REWARD_PROFILES,
                        help="Recompensa do PPO/self-play (padrão: potential)")
    parser.add_argument("--obs-profile", choices=OBS_PROFILES, default=DEFAULT_OBS_PROFILE,
                        help="Perfil de observacao para novo treino BC (padrao: v3)")
    parser.add_argument("--include-casual", action="store_true",
                        help="Inclui partidas friendly/clanMate na imitacao (padrao: filtra para competitivo)")
    parser.add_argument("--no-mirror", action="store_true",
                        help="Desativa espelhamento horizontal 2x na imitacao")
    parser.add_argument("--early-stop-patience", type=int, default=4,
                        help="Epocas sem melhora na validacao BC antes de parar (0 desativa)")
    parser.add_argument("--compare-rewards", type=Path, metavar="PASTA",
                        help="Compara legacy e potential a partir do mesmo checkpoint conjunto")
    parser.add_argument("--rollout-games", type=int, default=4)
    parser.add_argument("--parallel-games", type=int, default=None,
                        help="Partidas simultâneas (tabuleiros em paralelo)")
    parser.add_argument("--rollout-workers", type=int, default=None,
                        help="Processos paralelos de simulacao CPU (padrao: automatico; 1 desativa)")
    parser.add_argument("--kl-coef", type=float, default=0.12)
    parser.add_argument("--kl-direction", choices=("reverse", "forward"), default="reverse",
                        help="Direcao da divergencia KL contra o BC (padrao: reverse)")
    parser.add_argument("--no-kl-decay", action="store_true",
                        help="Mantem kl-coef constante em vez de decair exponencialmente")
    parser.add_argument("--rival-bc-checkpoint", type=Path,
                        help="Checkpoint BC opcional para inicializar a rede adversaria no coevolve")
    parser.add_argument("--evaluate", type=int, default=0, metavar="GAMES")
    parser.add_argument("--evaluate-checkpoint", type=Path, help="Compara um checkpoint PPO salvo com o BC")
    parser.add_argument("--replay-eval", action="store_true",
                        help="Avalia cartas e posicionamento nos replays reservados pelo BC")
    args = parser.parse_args()
    if args.scan:
        if args.parquet_path is None:
            parser.error("--scan requer --parquet-path")
        counts = Counter()
        for _ in winning_replays(args.parquet_path, args.deck, counts,
                                 deck_cards=(DECKS | RIVAL_DECKS)[args.deck]):
            pass
        print(f"{args.deck}: {dict(counts)}")
        return
    if not args.parquet_path and not args.bc_checkpoint:
        parser.error("informe --parquet-path ou --bc-checkpoint")
    if not (args.parquet_path or args.ppo_updates or args.evaluate or args.continuous):
        parser.error("selecione treino com --ppo-updates ou avaliacao com --evaluate")
    if args.parquet_path and args.bc_checkpoint and not args.replay_eval:
        parser.error("--parquet-path treina um BC novo; --bc-checkpoint carrega um BC existente")
    if args.holdout_checkpoint and (not args.parquet_path or args.replay_eval):
        parser.error("--holdout-checkpoint requer treino de imitação com --parquet-path")
    if args.resume_ppo and (not args.bc_checkpoint or not (args.ppo_updates or args.continuous)):
        parser.error("--resume-ppo requer --bc-checkpoint e --ppo-updates")
    if args.coevolve and (not args.resume_ppo or args.deck != "xbow_30" or args.parquet_path):
        parser.error("--coevolve requer --resume-ppo com o deck xbow_30")
    if args.continuous and (not args.coevolve or args.ppo_updates or args.evaluate or args.compare_rewards):
        parser.error("--continuous requer --coevolve e executa o treino sem limite de partidas")
    if args.compare_rewards and not args.coevolve:
        parser.error("--compare-rewards requer --coevolve")
    if args.rival_bc_checkpoint and not args.coevolve:
        parser.error("--rival-bc-checkpoint requer --coevolve")
    if args.compare_rewards:
        if args.reward or args.evaluate < 2 or args.evaluate % 2:
            parser.error("--compare-rewards requer --evaluate par a partir de 2 e roda os dois perfis")
        if args.compare_rewards.exists():
            parser.error("escolha uma pasta nova para --compare-rewards")
    if args.evaluate_checkpoint and (not (args.evaluate or args.replay_eval) or args.ppo_updates):
        parser.error("--evaluate-checkpoint requer --evaluate ou --replay-eval sem --ppo-updates")
    if args.replay_eval and (not args.bc_checkpoint or not args.parquet_path
                             or args.ppo_updates or args.evaluate):
        parser.error("--replay-eval requer --bc-checkpoint e --parquet-path")
    threads = 2 if args.threads is None else args.threads
    if min(args.matches - 1, args.epochs, args.batch_size, threads, args.rollout_games) < 1:
        parser.error("informe pelo menos 2 partidas e valores positivos para os demais limites")
    if args.parallel_games is not None and args.parallel_games < 1:
        parser.error("parallel-games deve ser pelo menos 1")
    if args.rollout_workers is not None and args.rollout_workers < 1:
        parser.error("rollout-workers deve ser pelo menos 1")
    if not math.isfinite(args.kl_coef) or min(args.ppo_updates, args.kl_coef, args.evaluate, args.early_stop_patience) < 0:
        parser.error("PPO, KL, avaliacao e paciencia de parada devem ser nao negativos")
    torch.set_num_threads(threads)
    torch.manual_seed(args.seed)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    output = args.output or Path("runs") / args.deck
    if args.coevolve and output.name != "xbow_30":
        parser.error("--coevolve requer uma pasta --output com nome xbow_30")
    bc_path = args.bc_checkpoint or output / "bc.pt"
    if args.compare_rewards:
        from xbow_joint_training import compare_rewards

        if not (output / "ppo.pt").is_file() or not bc_path.is_file():
            parser.error("a comparação requer os checkpoints PPO e BC iniciais")
        if args.threads is None and device == "cpu":
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        compare_rewards(args, device, bc_path, output)
        return
    with training_journal(output, args.deck, continuous=args.continuous) as report:
        run(args, device, output, bc_path, report)


def run(args, device, output, bc_path, report):
    check_training_stop()
    if args.replay_eval:
        report("replay_validation")
        saved = torch.load(args.bc_checkpoint, map_location="cpu", weights_only=True)
        arch_cfg = saved.get("arch_config") or {}
        obs_profile = obs_profile_for_architecture(
            saved.get("architecture", 1), arch_cfg.get("obs_profile"),
        )
        env = battle_env(args.deck, obs_profile=obs_profile)
        try:
            replay_validation(env, args, device, output)
        finally:
            env.close()
        return
    if args.parquet_path:
        env = battle_env(args.deck, obs_profile=args.obs_profile)
        try:
            train_clone(env, args, device, bc_path, report)
        finally:
            env.close()
        print(f"Checkpoint BC: {bc_path}", flush=True)
    if args.ppo_updates or args.evaluate or args.continuous:
        check_training_stop()
        if args.threads is None and device == "cpu":
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        if args.coevolve:
            from xbow_joint_training import train_pair
            train_pair(args, device, bc_path, output, report, reward_profile=args.reward or "potential")
        else:
            from xbow_selfplay import run_selfplay
            run_selfplay(args, device, bc_path, output, report)


if __name__ == "__main__":
    try:
        main()
    except TrainingStopped:
        print("Parada solicitada pelo painel. Checkpoints salvos permanecem disponíveis.", flush=True)
        raise SystemExit(130) from None
