#!/usr/bin/env bash
# Group 14 EV Battery Digital-Twin Virtual Lab: reproduction tasks (Linux/macOS).
#   bash scripts/lab.sh run | data | calibrate | figures | test | lint | all
# Windows: set PYTHONPATH=src and run the python commands below directly.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ -x .venv/bin/python ]; then PY=${PYTHON:-.venv/bin/python}; else PY=${PYTHON:-python3}; fi
export PYTHONPATH="$ROOT/src"
case "${1:-help}" in
  run)       shift; exec bash "$ROOT/run.sh" "$@" ;;                       # start the virtual lab (same as run.sh)
  data)      "$PY" -m g14lab.data_pipeline && "$PY" -m g14lab.telemetry ;;  # raw archives -> data/processed (~2 min)
  calibrate) "$PY" -m g14lab.calibration ;;                               # identification, shadow, benchmarks (~4 min)
  figures)   "$PY" scripts/make_report_figures.py ;;                      # 300 dpi report figures
  test)      "$PY" -m pytest -q tests ;;                                  # engine tests
  lint)      "$PY" -m ruff check . ;;                                     # static checks
  all)       for t in data calibrate figures test; do bash "$ROOT/scripts/lab.sh" "$t"; done ;;
  *)         echo "usage: bash scripts/lab.sh {run|data|calibrate|figures|test|lint|all}"; exit 1 ;;
esac
