#!/usr/bin/env python
"""Tiny CPU task-diversity sweep for the reduced linear-attention theory code."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
THEORY_COMMON = REPO_ROOT / "theory_runs" / "Linear" / "Fig2_Context_Length"
sys.path.insert(0, str(THEORY_COMMON))

from common import gen_err_analytical_NEW, learn_Gamma_fast_NEW  # noqa: E402


def finite_dmmse_error(
    *,
    d: int,
    n_context: int,
    k: int,
    sigma_noise: float,
    finite_test_task: bool,
    n_sim: int,
    rng: np.random.Generator,
) -> float:
    """Estimate finite-task dMMSE error with the same estimator as Fig. 3/5."""
    beta_bank = rng.normal(size=(d, k))
    errors = np.zeros(n_sim)

    for i in range(n_sim):
        x = rng.normal(size=(d, n_context)) / np.sqrt(d)
        if finite_test_task:
            beta = beta_bank[:, rng.integers(k)].reshape(d, 1)
        else:
            beta = rng.normal(size=(d, 1))

        y = x.T @ beta + rng.normal(size=(n_context, 1)) * sigma_noise
        log_weights = -np.linalg.norm(y - x.T @ beta_bank, axis=0) ** 2 / (
            2 * sigma_noise**2
        )
        weights = np.exp(log_weights - np.max(log_weights))
        beta_hat = beta_bank @ weights.reshape(k, 1) / np.sum(weights)

        x_query = rng.normal(size=(d, 1)) / np.sqrt(d)
        y_query = (x_query.T @ beta).item() + rng.normal() * sigma_noise
        errors[i] = ((x_query.T @ beta_hat).item() - y_query) ** 2

    return float(np.mean(errors))


def run_sweep(args: argparse.Namespace) -> list[dict[str, float]]:
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    d = args.dimension
    alpha = args.alpha
    n_context = int(alpha * d)
    sigma_noise = np.sqrt(args.rho)
    sigma_beta = 1.0
    rows = []

    for kappa in args.kappas:
        k = max(1, int(round(kappa * d)))
        repeats_per_task = max(1, int(round(args.tau * d**2 / k)))
        n_pretrain = repeats_per_task * k

        icl_errors = []
        idg_errors = []
        for _ in range(args.num_avg):
            beta_bank = rng.normal(size=(k, d)) * sigma_beta
            norms = np.linalg.norm(beta_bank, axis=1)
            beta_bank = beta_bank / norms[:, None] * np.sqrt(d)
            beta = np.repeat(beta_bank[None, :, :], repeats_per_task, axis=0).reshape(
                n_pretrain, d
            )

            gamma = learn_Gamma_fast_NEW(
                beta, alpha, sigma_noise, args.ridge, args.tau_max
            )
            icl_errors.append(
                gen_err_analytical_NEW(
                    gamma,
                    alpha,
                    np.zeros(d),
                    sigma_beta**2 * np.eye(d),
                    (sigma_noise / sigma_beta) ** 2,
                )
            )
            idg_errors.append(
                gen_err_analytical_NEW(
                    gamma,
                    alpha,
                    np.mean(beta_bank, axis=0),
                    (beta_bank.T @ beta_bank) / k,
                    (sigma_noise / sigma_beta) ** 2,
                )
            )

        rows.append(
            {
                "d": d,
                "alpha": alpha,
                "tau": args.tau,
                "kappa": kappa,
                "k": k,
                "n_pretrain": n_pretrain,
                "linear_attention_icl": float(np.mean(icl_errors)),
                "linear_attention_idg": float(np.mean(idg_errors)),
                "dmmse_icl": finite_dmmse_error(
                    d=d,
                    n_context=n_context,
                    k=k,
                    sigma_noise=sigma_noise,
                    finite_test_task=False,
                    n_sim=args.dmmse_samples,
                    rng=rng,
                ),
                "dmmse_idg": finite_dmmse_error(
                    d=d,
                    n_context=n_context,
                    k=k,
                    sigma_noise=sigma_noise,
                    finite_test_task=True,
                    n_sim=args.dmmse_samples,
                    rng=rng,
                ),
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("test_runs/outputs/theory_task_diversity.csv"))
    parser.add_argument("--dimension", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--tau-max", type=float, default=3.0)
    parser.add_argument("--num-avg", type=int, default=3)
    parser.add_argument("--dmmse-samples", type=int, default=50)
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
