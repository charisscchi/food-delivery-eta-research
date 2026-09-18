"""Pipeline orchestrator — the project's main entry point.

Runs the five stages in order (Swiggy-style run_daily pattern):
    ingest -> validate -> features -> train -> evaluate

Each stage logs its own banner, writes its own artifacts, and a
run_summary.json is produced at the end so the run is auditable
without reading logs.

Usage (from repo root, inside .venv):
    python scripts/run_pipeline.py            # all stages
    python scripts/run_pipeline.py --stages ingest,validate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Windows console: force UTF-8 so log formatting never explodes on cp1252
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.config import load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_pipeline")

STAGES = ["ingest", "validate", "features", "train", "evaluate"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the delivery-eta pipeline")
    parser.add_argument("--stages", type=str, default=",".join(STAGES), help="comma-separated stage subset")
    args = parser.parse_args()

    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    invalid = [s for s in requested if s not in STAGES]
    if invalid:
        log.error("Unknown stage(s): %s. Valid: %s", invalid, STAGES)
        return 1

    cfg = load_config()
    summary: dict = {
        "pipeline": "delivery-eta-quantile",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "stages": {},
        "status": "RUNNING",
    }

    t_start = time.time()
    for stage in STAGES:
        if stage not in requested:
            log.info("[skip] %s (not requested)", stage)
            continue

        log.info("=" * 70)
        log.info("STAGE: %s", stage.upper())
        log.info("=" * 70)
        t0 = time.time()
        try:
            if stage == "ingest":
                from src.ingestion.ingest import run as stage_run
            elif stage == "validate":
                from src.validation.validate import run as stage_run
            elif stage == "features":
                from src.features.build_features import run as stage_run
            elif stage == "train":
                from src.training.train import run as stage_run
            elif stage == "evaluate":
                from src.evaluation.evaluate import run as stage_run

            result = stage_run(stage=stage, cfg=cfg)
            elapsed = round(time.time() - t0, 1)
            summary["stages"][stage] = {
                "status": "OK",
                "elapsed_sec": elapsed,
                "detail": _stage_detail(stage, result),
            }
            log.info("STAGE %s done in %.1fs", stage.upper(), elapsed)
        except Exception as e:
            summary["stages"][stage] = {"status": "FAILED", "error": str(e)}
            summary["status"] = "FAILED"
            summary["elapsed_sec"] = round(time.time() - t_start, 1)
            _write_summary(cfg, summary)
            log.exception("Stage %s FAILED: %s", stage, e)
            return 2

    summary["status"] = "SUCCESS"
    summary["elapsed_sec"] = round(time.time() - t_start, 1)
    _write_summary(cfg, summary)

    log.info("=" * 70)
    log.info("PIPELINE COMPLETE in %.1fs — reports/run_summary.json", summary["elapsed_sec"])
    return 0


def _stage_detail(stage: str, result) -> dict:
    """Keep run_summary lean: a couple of headline numbers per stage."""
    if result is None:
        return {}
    if isinstance(result, Path):
        return {"output": str(result)}
    if stage == "validation":
        return {"n_pass": sum(1 for g in result.get("gates", []) if g["status"] == "PASS"),
                "n_warn": result.get("n_warn", 0)}
    if stage == "features":
        return {k: result.get(k) for k in ("n_train", "n_test", "n_new", "late_threshold_min") if k in result}
    if stage in ("train", "evaluate"):
        return {k: result.get(k) for k in ("champion",) if k in result}
    return {}


def _write_summary(cfg: dict, summary: dict) -> None:
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    sys.exit(main())
