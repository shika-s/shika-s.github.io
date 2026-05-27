"""
Standalone end-to-end test for the submission.
Mirrors the two notebook cells (submission test + validation) so you can verify
the submission works in a clean Python environment without Jupyter or Colab.

Usage (from project root):
    python3 test_submission.py

Tests:
    1. Phase 1: Model.load() — reads weights.pkl + clusters.pkl
    2. Phase 2: Model.prepare(data_dir) — reads raw parquets, computes 26 features,
                runs ensemble, builds scores dict (slowest step, ~10 min)
    3. Phase 3: Model.predict(timestamp) — spot checks at first/middle/last bars
    4. Format validation — verifies pred shape, dtype, NaN/inf, ticker overlap
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure submission/ is importable
sys.path.insert(0, str(Path(__file__).parent / "submission"))

from model import Model


def main():
    project_root = Path(__file__).parent
    data_dir = project_root / "data" / "train"
    weights_path = project_root / "submission" / "weights.pkl"
    config_path = project_root / "submission" / "config.json"
    universe_path = project_root / "universe.json"

    # Sanity check files exist
    for p in [data_dir, weights_path, config_path, universe_path]:
        if not p.exists():
            print(f"[FAIL] missing: {p}")
            sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    submission_model = Model()

    # ------------------------------------------------------------------
    # Phase 1: load()
    # ------------------------------------------------------------------
    print("=== Phase 1: load() ===")
    t0 = time.time()
    submission_model.load(str(weights_path), config)
    elapsed = time.time() - t0
    print(f"  loaded in {elapsed:.1f}s")

    if hasattr(submission_model, "booster"):
        print(f"  booster type:    {type(submission_model.booster).__name__}")
    elif hasattr(submission_model, "m_lgb"):
        print(f"  ensemble loaded:")
        print(f"    lgb: {type(submission_model.m_lgb).__name__}")
        print(f"    xgb: {type(submission_model.m_xgb).__name__}")
        print(f"    rf:  {type(submission_model.m_rf).__name__}")
        if hasattr(submission_model, "weights"):
            print(f"  weights:         {submission_model.weights}")
    print(f"  cluster_map:     {len(submission_model.cluster_map):,} tickers")
    print(f"  prepare_start:   {submission_model.prepare_start.date()}")
    print(f"  chunk size:      {submission_model.predict_chunk_size:,}")

    # ------------------------------------------------------------------
    # Phase 2: prepare()
    # ------------------------------------------------------------------
    print("\n=== Phase 2: prepare() ===")
    print(f"  data dir: {data_dir}")
    print(f"  (reads 30-min + 5-min + 1-day parquets per ticker, computes 26 features)")
    t0 = time.time()
    submission_model.prepare(str(data_dir))
    elapsed = time.time() - t0
    print(f"\n  prepare() done in {elapsed/60:.1f} min ({elapsed:.0f}s)")
    print(f"  scores dict: {len(submission_model.scores):,} timestamps")

    # ------------------------------------------------------------------
    # Phase 3: predict() spot checks
    # ------------------------------------------------------------------
    print("\n=== Phase 3: predict() spot checks ===")
    all_ts = sorted(submission_model.scores.keys())
    for i in [0, len(all_ts) // 2, -1]:
        ts = all_ts[i]
        t0 = time.time()
        pred = submission_model.predict(ts)
        elapsed_ms = (time.time() - t0) * 1000
        print(f"  {ts}: {len(pred)} tickers, predict took {elapsed_ms:.2f}ms")
        print(f"    sample: {dict(list(pred.head(3).items()))}")

    # ------------------------------------------------------------------
    # Phase 4: format validation
    # ------------------------------------------------------------------
    print("\n=== Phase 4: format validation ===")
    with open(universe_path) as f:
        expected_tickers = set(json.load(f)["tickers"])
    print(f"  universe size: {len(expected_tickers)}")

    sample_indices = np.linspace(0, len(all_ts) - 1, 5, dtype=int)
    all_passed = True
    for i in sample_indices:
        ts = all_ts[i]
        pred = submission_model.predict(ts)

        if not isinstance(pred, pd.Series):
            print(f"  [FAIL] {ts}: predict() returned {type(pred).__name__}, expected pd.Series")
            all_passed = False
            continue

        valid = set(pred.index) & expected_tickers
        issues = []
        if len(valid) == 0:
            issues.append("zero valid tickers")
        if len(pred) > 0 and not np.issubdtype(pred.dtype, np.number):
            issues.append(f"non-numeric ({pred.dtype})")
        if pred.isna().any():
            issues.append(f"{pred.isna().sum()} NaN")
        if len(pred) > 0 and np.isinf(pred.values).any():
            issues.append(f"{np.isinf(pred.values).sum()} inf")

        if issues:
            print(f"  [FAIL] {ts}: {', '.join(issues)}")
            all_passed = False
        else:
            print(f"  [PASS] {ts}: {len(valid)} tickers in universe, clean")

    print()
    if all_passed:
        print("ALL CHECKS PASSED — submission is correctly formatted")
        sys.exit(0)
    else:
        print("SOME CHECKS FAILED — fix before submitting")
        sys.exit(1)


if __name__ == "__main__":
    main()
