#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python test_runs/theory_task_diversity_sweep.py "$@"
