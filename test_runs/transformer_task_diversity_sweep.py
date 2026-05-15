#!/usr/bin/env python
"""Tiny CPU task-diversity sweep for the Transformer training code."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import optax


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "transformer_codebase"))

from model.transformer import TransformerConfig  # noqa: E402
from task.regression import FiniteSampler, LinearRegressionCorrect  # noqa: E402
from trainmini import train  # noqa: E402


def metric_to_float(metric, name: str) -> float:
    return float(getattr(metric, name).item())


def run_sweep(args: argparse.Namespace) -> list[dict[str, float]]:
    rows = []
    d = args.dimension
    n_points = int(args.alpha * d) + 1
    p = max(args.batch_size, int(args.tau * d**2))

    for sweep_index, kappa in enumerate(args.kappas):
        seed = args.seed + sweep_index
        np.random.seed(seed)
        random.seed(seed)

        k = max(1, int(round(kappa * d)))
        train_task = FiniteSampler(
            n_points=n_points,
            n_dims=d,
            eta_scale=args.sigma,
            w_scale=1.0,
            diversity=k,
            batch_size=p,
            seed=seed,
        )
        idg_task = train_task
        icl_task = LinearRegressionCorrect(
            n_points=n_points,
            n_dims=d,
            eta_scale=args.sigma,
            w_scale=1.0,
            batch_size=p,
            seed=seed + 10_000,
        )

        config = TransformerConfig(
            pos_emb=False,
            n_hidden=args.hidden_dim,
            n_layers=args.layers,
            n_mlp_layers=0,
            pure_linear_self_att=False,
        )
        _, hist = train(
            config,
            data_iter=iter(train_task),
            idg_iter=iter(idg_task),
            test_iter=iter(icl_task),
            loss="mse",
            batch_size=args.batch_size,
            test_every=args.test_every,
            train_iters=args.train_iters,
            test_iters=args.test_iters,
            optim=optax.adamw,
            lr=args.learning_rate,
            seed=seed,
        )

        rows.append(
            {
                "d": d,
                "alpha": args.alpha,
                "tau": args.tau,
                "kappa": kappa,
                "k": k,
                "batch_contexts": p,
                "train_iters": args.train_iters,
                "idg_loss": metric_to_float(hist["test"][-1], "loss"),
                "icl_loss": metric_to_float(hist["true_test"][-1], "loss"),
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("test_runs/outputs/transformer_task_diversity.csv"))
    parser.add_argument("--dimension", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-iters", type=int, default=20)
    parser.add_argument("--test-every", type=int, default=10)
    parser.add_argument("--test-iters", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kappas", type=float, nargs="+", default=[0.2, 0.5, 1.0, 2.0])
    args = parser.parse_args()

    rows = run_sweep(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
