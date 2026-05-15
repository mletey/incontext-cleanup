#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
bash test_runs/run_theory_task_diversity_cpu.sh
bash test_runs/run_transformer_task_diversity_cpu.sh
