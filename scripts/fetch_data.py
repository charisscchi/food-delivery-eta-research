"""Download the raw delivery dataset into data/raw/.

PowerShell-friendly (no wget/curl shell-isms), idempotent: skips if the
file already exists with a sane size.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from urllib.request import urlretrieve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get, load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch_data")

MIN_BYTES = 5 * 1024 * 1024  # 5 MB


def main() -> None:
    cfg = load_config()
    url = get(cfg, "ingestion.source_url")
    raw_path = Path(get(cfg, "paths.raw_data"))
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists() and raw_path.stat().st_size >= MIN_BYTES:
        log.info("Raw data already present (%.1f MB) — skipping download.", raw_path.stat().st_size / 1e6)
        return

    log.info("Downloading %s -> %s", url, raw_path)
    urlretrieve(url, raw_path)
    size_mb = raw_path.stat().st_size / 1e6
    if size_mb < 1.0:
        raise RuntimeError(f"Downloaded file suspiciously small ({size_mb:.2f} MB)")
    log.info("Done: %.2f MB", size_mb)


if __name__ == "__main__":
    main()
