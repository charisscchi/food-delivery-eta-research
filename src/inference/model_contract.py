"""Model contract — the DS-owned interface boundary.

Pattern taken from Swiggy's DS <> DSP split: DS owns the model *and*
its contract — required inputs, valid ranges, output schema and
fallback rules. Anything outside the contract at inference time is
handled deterministically (fallback to global stats / REJECT), never
with a stack trace.

    validate_and_fallback(df)
      -> clean_df   (rows OK or repaired via fallback, contract_status=OK/FALLBACK)
         flagged_df (rows REJECTED — missing critical fields)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.constants import (
    CONTRACT_ALLOWED_CATEGORICAL,
    CONTRACT_INPUT_REQUIRED,
    CONTRACT_NUMERIC_RANGES,
    CONTRACT_OUTPUT_SCHEMA,
    VALID_CONTRACT_STATUS,
)

log = logging.getLogger("model_contract")


@dataclass
class ModelContract:
    """Declarative input/output contract for the ETA model."""

    required_inputs: list[str] = field(default_factory=lambda: list(CONTRACT_INPUT_REQUIRED))
    numeric_ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(CONTRACT_NUMERIC_RANGES)
    )
    allowed_categoricals: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in CONTRACT_ALLOWED_CATEGORICAL.items()}
    )
    output_schema: dict[str, str] = field(default_factory=lambda: dict(CONTRACT_OUTPUT_SCHEMA))
    fallback_p50: float = 30.0   # global median fallback (minutes) — overwritten by bundle
    fallback_p10: float = 20.0
    fallback_p90: float = 45.0
    fallback_late_risk: float = 0.25

    def validate_row_batch(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split a raw batch into (repairable, rejected) with per-row contract notes.

        Repairable rows get fallback values for out-of-range numerics and
        unknown categoricals (imputed to 'unknown' at feature-build time);
        rows missing critical keys (rider_id, order_date, coords) are rejected.
        """
        notes: dict[int, list[str]] = {}
        clean = df.copy()

        # 1. missing critical fields -> REJECT
        critical = ["rider_id", "order_date", "rest_lat", "rest_lon", "dest_lat", "dest_lon"]
        missing_critical = clean[critical].isna().any(axis=1) | clean["rider_id"].isna()
        for col in ["rest_lat", "rest_lon", "dest_lat", "dest_lon"]:
            if col in clean.columns:
                missing_critical |= clean[col].eq(0) & clean[col.replace("lat", "lon")].eq(0)

        rejected = clean[missing_critical].copy()
        clean = clean[~missing_critical].copy()

        # 2. out-of-range numerics -> repair with fallback values, flag FALLBACK
        for col, (lo, hi) in self.numeric_ranges.items():
            if col in clean.columns:
                bad = clean[col].isna() | ~clean[col].between(lo, hi)
                if bad.any():
                    fill = float(clean[col].where(clean[col].between(lo, hi)).median())
                    fill = fill if pd.notna(fill) else 0.0
                    for idx in clean.index[bad]:
                        notes.setdefault(idx, []).append(f"numeric_fallback:{col}")
                    clean.loc[bad, col] = fill

        # 3. unknown categoricals -> 'unknown' token, flag FALLBACK
        for col, allowed in self.allowed_categoricals.items():
            if col in clean.columns:
                bad = ~clean[col].isin(allowed)
                if bad.any():
                    for idx in clean.index[bad]:
                        notes.setdefault(idx, []).append(f"categorical_fallback:{col}")
                    clean.loc[bad, col] = "unknown"

        # 4. assemble status
        clean["contract_status"] = ["FALLBACK" if notes.get(i) else "OK" for i in clean.index]
        clean["contract_notes"] = [";".join(notes.get(i, [])) for i in clean.index]
        rejected["contract_status"] = "REJECTED"
        if len(rejected):
            miss_cols = rejected[critical].isna()
            rejected["contract_notes"] = miss_cols.apply(
                lambda row: ";".join(c for c in critical if bool(row[c])), axis=1
            ).map(lambda s: f"missing critical field(s): {s}" if s else "zero/invalid coordinates")
        else:
            rejected["contract_notes"] = []

        return clean, rejected


def contract_output_schema() -> dict[str, str]:
    return dict(CONTRACT_OUTPUT_SCHEMA)


def is_valid_status(status: str) -> bool:
    return status in VALID_CONTRACT_STATUS


# --------------------------------------------------------------- scoring API
def attach_predictions(
    df: pd.DataFrame,
    bundle: dict,
    contract: ModelContract,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Build features, score quantile models + late-risk, attach promise time.

    This is the single scoring path used by evaluation, drift-checked
    batch scoring, and any future serving wrapper — one implementation,
    no train/serve skew.
    """
    from src.features.build_features import build_core_features
    from src.training.train import _align_columns, _clean_for_lgbm, _matrix

    df = build_core_features(df, cfg)

    X = _align_columns(_matrix(df), bundle["feature_columns"])
    X = _clean_for_lgbm(X)

    q_models = bundle["quantile_models"]
    quantiles = bundle["config_snapshot"]["quantiles"]
    out = df.copy()
    for q in quantiles:
        out[f"p{int(q*100)}_min"] = q_models[q].predict(X)

    # enforce quantile monotonicity row-wise (rare crossing repairs)
    p10c, p50c, p90c = f"p{int(quantiles[0]*100)}_min", f"p{int(quantiles[1]*100)}_min", f"p{int(quantiles[2]*100)}_min"
    out[p50c] = out[[p10c, p50c, p90c]].median(axis=1)
    out[p10c] = np.minimum(out[p10c], out[p50c])
    out[p90c] = np.maximum(out[p90c], out[p50c])

    out["late_risk"] = bundle["late_model"].predict_proba(X)[:, 1]
    out["late_flag"] = out["late_risk"] >= bundle["alert_threshold"]

    # promise: padded, bucketed P90 (customer-facing)
    from src.config import get as _get, load_config as _load
    _cfg = cfg or _load()
    bucket = _get(_cfg, "evaluation.promise_time.bucket_minutes", 5)
    out["promise_min"] = np.ceil(out[p90c] / bucket) * bucket

    # fallback rows keep quantile scores but are flagged
    return out


def summarize_batch(scored: pd.DataFrame) -> dict:
    """Batch-level summary used in the scoring report."""
    return {
        "n_scored": int(len(scored)),
        "n_ok": int((scored["contract_status"] == "OK").sum()),
        "n_fallback": int((scored["contract_status"] == "FALLBACK").sum()),
        "n_rejected": int((scored["contract_status"] == "REJECTED").sum()),
        "share_late_flagged": float(scored["late_flag"].mean()) if len(scored) else 0.0,
    }
