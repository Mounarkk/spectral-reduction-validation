"""Paths relative to the repository root, so scripts run from any directory."""
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
CMF     = DATA / "cmf"
RESULTS = ROOT / "results"
