#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/belllab-cli-workflow.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

run_belllab() {
  set +e
  python3 -m belllab "$@"
  local code=$?
  set -e
  case "$code" in
    0|1|5) return 0 ;;
    *) return "$code" ;;
  esac
}

export WORK_DIR
python3 - <<'PY'
import os
from pathlib import Path

import numpy as np
import soundfile as sf

root = Path(os.environ["WORK_DIR"])
audio = root / "audio"
audio.mkdir(parents=True, exist_ok=True)
sample_rate = 4096
time = np.arange(sample_rate, dtype=np.float64) / sample_rate
for index, label in enumerate(("pp", "p", "mf", "f", "ff")):
    frequency = 300.0 + index
    signal = 0.3 * np.exp(-time / 2.0) * np.sin(2 * np.pi * frequency * time)
    sf.write(audio / f"{label}.wav", signal.astype(np.float32), sample_rate, subtype="FLOAT")
PY

run_belllab analyze \
  --recording "pp=$WORK_DIR/audio/pp.wav" \
  --recording "p=$WORK_DIR/audio/p.wav" \
  --recording "mf=$WORK_DIR/audio/mf.wav" \
  --recording "f=$WORK_DIR/audio/f.wav" \
  --recording "ff=$WORK_DIR/audio/ff.wav" \
  --until-stage global_spectrum \
  --save-result "$WORK_DIR/analysis.json" \
  --output-format json

run_belllab export \
  --analysis "$WORK_DIR/analysis.json" \
  --output-dir "$WORK_DIR/export" \
  --json --csv --manifest \
  --overwrite replace \
  --save-result "$WORK_DIR/export-bundle.json" \
  --output-format json

run_belllab visualize \
  --analysis "$WORK_DIR/analysis.json" \
  --output-dir "$WORK_DIR/figures" \
  --figure global_spectrum \
  --format png --format svg \
  --overwrite replace \
  --save-result "$WORK_DIR/figures-bundle.json" \
  --output-format json

run_belllab report \
  --analysis "$WORK_DIR/analysis.json" \
  --export-result "$WORK_DIR/export-bundle.json" \
  --figure-collection "$WORK_DIR/figures-bundle.json" \
  --output-dir "$WORK_DIR/report" \
  --markdown --latex \
  --overwrite replace \
  --title "BellLab CLI workflow report" \
  --author "BellLab example" \
  --save-result "$WORK_DIR/report-bundle.json" \
  --output-format json

run_belllab inspect "$WORK_DIR/report-bundle.json" --validate --output-format json
