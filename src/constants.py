"""Project-wide constants: column names, cleaning maps, contract ranges.

Kept separate from config.yaml because these are *code facts* (schemas,
string prefixes baked into the raw dump) rather than tunable knobs.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- raw schema
ID_COL = "ID"
TARGET_COL = "time_taken_min"          # cleaned target (minutes, float)
ORDER_DATE_COL = "order_date"
LATE_LABEL_COL = "is_late"              # time_taken_min > train P75

RAW_TIME_TAKEN_PREFIX = "(min)"
RAW_WEATHER_PREFIX = "conditions"

# Raw -> clean column rename map for the ingestion stage.
RAW_COLUMN_RENAME = {
    "ID": "ID",
    "Delivery_person_ID": "rider_id",
    "Delivery_person_Age": "rider_age",
    "Delivery_person_Ratings": "rider_rating",
    "Restaurant_latitude": "rest_lat",
    "Restaurant_longitude": "rest_lon",
    "Delivery_location_latitude": "dest_lat",
    "Delivery_location_longitude": "dest_lon",
    "Order_Date": "order_date_raw",
    "Time_Orderd": "time_ordered",
    "Time_Order_picked": "time_picked",
    "Weatherconditions": "weather_raw",
    "Road_traffic_density": "road_traffic_density",
    "Vehicle_condition": "vehicle_condition",
    "Type_of_order": "type_of_order",
    "Type_of_vehicle": "type_of_vehicle",
    "multiple_deliveries": "multiple_deliveries",
    "Festival": "festival",
    "City": "city",
    "Time_taken(min)": "time_taken_raw",
}

# Columns expected in the validated interim dataset.
INTERIM_COLUMNS = [
    "ID",
    "rider_id",
    "rider_age",
    "rider_rating",
    "rest_lat",
    "rest_lon",
    "dest_lat",
    "dest_lon",
    "order_date",
    "time_ordered",
    "time_picked",
    "weather",
    "road_traffic_density",
    "vehicle_condition",
    "type_of_order",
    "type_of_vehicle",
    "multiple_deliveries",
    "festival",
    "city",
    TARGET_COL,
]

# ------------------------------------------------------------ feature groups
NUMERIC_FEATURES = [
    "rider_age",
    "rider_rating",
    "haversine_km",
    "lat_diff",
    "lon_diff",
    "order_hour",
    "pickup_delay_min",
    "multiple_deliveries",
    "vehicle_condition",
    "is_weekend",
    "is_lunch_peak",
    "is_dinner_peak",
    "rider_hist_median_min",
]

CATEGORICAL_FEATURES = [
    "weather",
    "road_traffic_density",
    "city",
    "type_of_order",
    "type_of_vehicle",
    "festival",
    "rider_city",
    "day_of_week",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ------------------------------------------------------- model contract (I/O)
CONTRACT_INPUT_REQUIRED = [
    "rider_age",
    "rider_rating",
    "rest_lat",
    "rest_lon",
    "dest_lat",
    "dest_lon",
    "order_date",
    "time_ordered",
    "time_picked",
    "weather",
    "road_traffic_density",
    "vehicle_condition",
    "type_of_order",
    "type_of_vehicle",
    "multiple_deliveries",
    "festival",
    "city",
    "rider_id",
]

# Valid ranges for numeric inputs (out-of-range -> fallback, flagged).
CONTRACT_NUMERIC_RANGES = {
    "rider_age": (15, 70),
    "rider_rating": (0.0, 6.0),
    "rest_lat": (-90.0, 90.0),
    "rest_lon": (-180.0, 180.0),
    "dest_lat": (-90.0, 90.0),
    "dest_lon": (-180.0, 180.0),
    "haversine_km": (0.0, 60.0),
    "multiple_deliveries": (0, 5),
    "vehicle_condition": (0, 3),
}

CONTRACT_ALLOWED_CATEGORICAL = {
    "weather": ["Sunny", "Stormy", "Sandstorms", "Cloudy", "Fog", "Windy"],
    "road_traffic_density": ["Low", "Medium", "High", "Jam"],
    "city": ["Urban", "Metropolitian", "Semi-Urban"],  # 'Metropolitian' is a dataset artifact, kept as-is
    "type_of_order": ["Snack", "Drinks", "Meal", "Buffet"],
    "type_of_vehicle": ["motorcycle", "scooter", "electric_scooter", "bicycle"],
    "festival": ["Yes", "No"],
}

CONTRACT_OUTPUT_SCHEMA = {
    "ID": "object",
    "p10_min": "float",
    "p50_min": "float",
    "p90_min": "float",
    "promise_min": "float",
    "late_risk": "float",
    "late_flag": "bool",
    "contract_status": "object",
    "contract_notes": "object",
}

VALID_CONTRACT_STATUS = ["OK", "FALLBACK", "REJECTED"]

# Malformed sentinel used across the raw dump (not pandas NaN).
NA_TOKEN = "NaN"
