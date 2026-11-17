"""Project directory layout (all paths resolved relative to the project root)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
NASA_RAW = RAW / "nasa_pcoe"
NASA_ZIP = NASA_RAW / "5. Battery Data Set.zip"
NASA_EXTRACT = NASA_RAW / "extracted"
PROCESSED = DATA / "processed"
SAMPLES = DATA / "samples"
METADATA = DATA / "metadata"
MODELS = ROOT / "models"
TRAINED = MODELS / "trained_models"
METRICS = MODELS / "model_metrics"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
EXPERIMENTS = OUTPUTS / "experiment_results"

for _p in (PROCESSED, SAMPLES, METADATA, TRAINED, METRICS, FIGURES, EXPERIMENTS):
    _p.mkdir(parents=True, exist_ok=True)
