"""Ingestion tests: prefix stripping, NaN sentinel, type coercion."""

from __future__ import annotations

import pandas as pd

from src.constants import INTERIM_COLUMNS, TARGET_COL
from src.ingestion.ingest import _clean_token, _strip_prefix, load_raw_csv


def test_strip_prefix():
    assert _strip_prefix("conditions Sunny", "conditions") == "Sunny"
    assert _strip_prefix(" (min) 24", "(min)") == "24"
    assert _strip_prefix("Sunny", "conditions") == "Sunny"   # no prefix -> unchanged
    assert _strip_prefix(24, "(min)") == 24                  # non-string passthrough


def test_clean_token():
    assert pd.isna(_clean_token("NaN "))
    assert pd.isna(_clean_token(" NaN"))
    assert _clean_token(" Urban ") == "Urban"
    assert _clean_token(4.5) == 4.5


def test_ingest_synthetic(tmp_path, synthetic_deliveries):
    """End-to-end ingestion on synthetic rows with planted artifacts."""
    raw = tmp_path / "raw.csv"

    # loader expects raw column names; remap our clean fixture back to raw
    clean_to_raw = {
        "rider_id": "Delivery_person_ID",
        "rider_age": "Delivery_person_Age",
        "rider_rating": "Delivery_person_Ratings",
        "rest_lat": "Restaurant_latitude",
        "rest_lon": "Restaurant_longitude",
        "dest_lat": "Delivery_location_latitude",
        "dest_lon": "Delivery_location_longitude",
        "order_date": "Order_Date",
        "time_ordered": "Time_Orderd",
        "time_picked": "Time_Order_picked",
        "weather": "Weatherconditions",
        "road_traffic_density": "Road_traffic_density",
        "vehicle_condition": "Vehicle_condition",
        "type_of_order": "Type_of_order",
        "type_of_vehicle": "Type_of_vehicle",
        "multiple_deliveries": "multiple_deliveries",
        "festival": "Festival",
        "city": "City",
        "time_taken_min": "Time_taken(min)",
    }
    raw_df = synthetic_deliveries.rename(columns=clean_to_raw)
    # raw file carries the prefix junk that ingestion strips; re-plant it
    # (normalize first — the fixture already plants a prefix on row 2)
    weather = raw_df["Weatherconditions"].astype(str).str.replace("conditions ", "", regex=False)
    raw_df["Weatherconditions"] = "conditions " + weather
    taken = raw_df["Time_taken(min)"].astype(str).str.replace("(min) ", "", regex=False)
    raw_df["Time_taken(min)"] = "(min) " + taken
    raw_df.to_csv(raw, index=False)

    df = load_raw_csv(raw)

    # dates parsed with the raw format
    assert df["order_date"].notna().all()

    # schema
    assert set(INTERIM_COLUMNS).issubset(df.columns)
    # planted artifacts cleaned
    assert pd.isna(df.loc[0, "rider_age"])          # 'NaN' sentinel -> null
    assert df.loc[1, "city"] == "Urban"             # trailing space stripped
    assert df.loc[2, "weather"] == "Stormy"         # 'conditions ' prefix stripped
    # target coerced numeric
    assert pd.api.types.is_numeric_dtype(df[TARGET_COL])
    assert df[TARGET_COL].notna().mean() == 1.0
