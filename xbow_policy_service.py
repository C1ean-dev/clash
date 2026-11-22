"""Serve checkpoint decisions locally over newline-delimited JSON."""

import argparse
import json
import sys
from time import perf_counter

import numpy as np
import torch

from xbow_replays import DECKS
from xbow_training import load_checkpoint


def observation(message, spec):
    channels, features, actions = spec
    spatial = np.asarray(message["spatial"], dtype=np.float32)
    vector = np.asarray(message["vector"], dtype=np.float32)
    mask = np.asarray(message["action_mask"])
    if (spatial.shape, vector.shape, mask.shape) != ((channels, 32, 18), (features,), (actions,)):
        raise ValueError("Dimensoes da observacao incompativeis com o checkpoint")
    if not np.isfinite(spatial).all() or not np.isfinite(vector).all():
        raise ValueError("A observacao contem valores nao finitos")
    if not np.isin(mask, [0, 1]).all() or not mask[0]:
        raise ValueError("A mascara deve ser binaria e permitir esperar")
    obs = {"spatial": spatial, "vector": vector, "action_mask": mask.astype(bool)}
    for key, planes in (("card_ids", 2), ("spell_ids", 4), ("unit_ids", 2)):
        if key in message:
            arr = np.asarray(message[key], dtype=np.uint8)
            if arr.shape != (planes, 32, 18):
                raise ValueError(f"Dimensoes invalidas para {key}")
            obs[key] = arr
    return obs


def describe_action(action):
    if action == 0:
        return {"kind": "wait"}
    card_actions = 4 * 32 * 18
    if action > card_actions:
        return {"kind": "ability", "button": action - card_actions - 1}
    slot, tile = divmod(action - 1, 32 * 18)
    y, x = divmod(tile, 18)
    return {"kind": "play_card", "slot": slot, "x": x, "y": y}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--deck", required=True, choices=DECKS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.45,
                        help="Temperatura de amostragem quando nao deterministico (padrao: 0.45)")
    parser.add_argument("--public-frames", action="store_true",
                        help="Recebe quadros públicos cronológicos em vez de tensores")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    policy, _ = load_checkpoint(args.checkpoint, args.deck, "cpu")
    feed = None
    if args.public_frames:
        from xbow_public_feed import PublicFeed
        feed = PublicFeed(args.deck, obs_profile=policy.obs_profile)
    try:
        for line in sys.stdin:
            try:
                received = perf_counter()
                message = json.loads(line)
                obs = feed.update(message) if feed is not None else observation(message, policy.spec)
                start = perf_counter()
                action, logp, _ = policy.decide(
                    obs, "cpu", args.deterministic, temperature=args.temperature,
                )
                elapsed = (perf_counter() - start) * 1000
                response = {
                    "action": action, **describe_action(action), "probability": float(np.exp(logp)),
                    "inference_ms": round(elapsed, 3),
                }
                if feed is not None:
                    response["tick"] = feed.tick
                    response["observation_ms"] = round((start - received) * 1000, 3)
                    response["hud"] = feed.public_summary()
                    if response["kind"] == "play_card":
                        response["card"] = message["hand"][response["slot"]]
            except (KeyError, TypeError, ValueError) as exc:
                response = {"error": str(exc)}
            print(json.dumps(response), flush=True)
    finally:
        if feed is not None:
            feed.close()


if __name__ == "__main__":
    main()
