#!/usr/bin/env bash
# The segmentation runs reported in README.md (golden-v2 eval split, thresholds chosen on the tune split).
# Needs: python -m golden.export --unreviewed, and OPENROUTER_API_KEY in .env. Cost ~$10, ~40 min.
set -euo pipefail
cd "$(dirname "$0")"
run() { .venv/bin/python -m playground.run "$@"; }
run --task seg --classifier similarity --fit-threshold
run --task seg --classifier jev --context full --fit-threshold
run --task seg --classifier jev --context last2 --fit-threshold
run --task seg --classifier llm --context full --fit-threshold --concurrency 24
.venv/bin/python -m playground.compare
