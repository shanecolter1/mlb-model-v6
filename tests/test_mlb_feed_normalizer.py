#!/usr/bin/env python3
from pathlib import Path
import subprocess
import tempfile
import pandas as pd

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp)
    subprocess.run([
        "python",
        str(root / "src/ingestion/normalize_mlb_feeds.py"),
        "--raw-dir", str(root / "tests/fixtures/mlb_statsapi"),
        "--output-dir", str(out),
    ], check=True)
    pa = pd.read_parquet(out / "plate_appearances.parquet")
    results = pd.read_parquet(out / "results.parquet")
    assert pa.iloc[0]["event"] == "home_run"
    assert int(results.iloc[0]["total_runs"]) == 7
print("MLB feed normalizer test passed.")
