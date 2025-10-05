from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import json
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
import requests
from collections import defaultdict

try:
    import h3
except Exception as e:
    raise ImportError(
        "The 'h3' package is required for generating hex boundaries. Install with: pip install h3"
    ) from e

# Import the route optimizer
from recommendation_system import BangkokTaxiOptimizer

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent

# Data locations
PREPARED_DIR = PROJECT_ROOT / "PickUP rate" / "prepared_data" / "models"
SEASONAL_PARQUET = PREPARED_DIR / "seasonal_baseline.parquet"
SEASONAL_CSV = PREPARED_DIR / "seasonal_baseline.csv"
THRESHOLDS_PATH = PREPARED_DIR / "color_thresholds.json"
BOOSTER_JSON = PREPARED_DIR / "xgb_pickup_5min.json"
FEATURE_NAMES_JSON = PREPARED_DIR / "feature_names.json"
CALIBRATOR_JOBLIB = PREPARED_DIR / "isotonic_calibrator.joblib"

# Model directory for optimizer (same directory as app.py)
MODELS_DIR = APP_ROOT / "models"

# Map defaults
BANGKOK_CENTER = [13.7563, 100.5018]
DEFAULT_ZOOM = 12
TARGET_H3_RES = 7

# OSRM API endpoint
OSRM_API = "http://router.project-osrm.org/route/v1/driving/"


def _load_thresholds() -> Dict[str, float]:
    if THRESHOLDS_PATH.exists():
        return json.loads(THRESHOLDS_PATH.read_text())
    return {"green": 0.7, "yellow": 0.5, "red": 0.3}


def _cell_to_polygon(cell: str) -> List[List[float]]:
    """Returns a closed polygon as list of [lon, lat] for the given H3 cell."""
    boundary = []
    try:
        if hasattr(h3, "h3_to_geo_boundary"):
            boundary = h3.h3_to_geo_boundary(cell, geo_json=True)
        elif hasattr(h3, "cell_to_boundary"):
            try:
                boundary = h3.cell_to_boundary(cell, geo_json=True)
            except TypeError:
                boundary = h3.cell_to_boundary(cell)
    except Exception:
        boundary = []

    poly: List[List[float]] = []
    for pt in boundary or []:
        if isinstance(pt, dict):
            lat, lng = pt.get("lat"), pt.get("lng")
        elif hasattr(pt, "lat") and hasattr(pt, "lng"):
            lat, lng = getattr(pt, "lat"), getattr(pt, "lng")
        else:
            lat, lng = pt[0], pt[1]
        poly.append([float(lng), float(lat)])
    if poly and poly[0] != poly[-1]:
        poly.append(poly[0])
    return poly


def _get_resolution(cell: str) -> int:
    if hasattr(h3, "h3_get_resolution"):
        return int(h3.h3_get_resolution(cell))
    if hasattr(h3, "get_resolution"):
        return int(h3.get_resolution(cell))
    return -1


def _children(cell: str, res: int) -> List[str]:
    if hasattr(h3, "h3_to_children"):
        return list(h3.h3_to_children(cell, res))
    if hasattr(h3, "cell_to_children"):
        return list(h3.cell_to_children(cell, res))
    return [cell]


def _prob_to_color(prob: float, th: Dict[str, float]) -> str:
    if prob >= th["green"]:
        return "green"
    if prob >= th["yellow"]:
        return "yellow"
    if prob >= th["red"]:
        return "red"
    return "transparent"


def _minute_of_day(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute


def get_osrm_route(
    start_lng: float, start_lat: float, end_lng: float, end_lat: float
) -> Tuple[List[List[float]], int]:
    """
    Get road route from OSRM API
    Returns tuple of (coordinates, trip_number) for progress tracking
    """
    try:
        url = f"{OSRM_API}{start_lng},{start_lat};{end_lng},{end_lat}?overview=full&geometries=geojson"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            if data.get("code") == "Ok" and data.get("routes"):
                coords = data["routes"][0]["geometry"]["coordinates"]
                return coords, 1

        return [[start_lng, start_lat], [end_lng, end_lat]], 0

    except Exception as e:
        print(f"OSRM routing error: {e}")
        return [[start_lng, start_lat], [end_lng, end_lat]], 0


def calculate_offset_position(
    lat: float, lng: float, offset_index: int, total_at_location: int
) -> Tuple[float, float]:
    """
    Calculate offset position for overlapping markers
    Distributes markers in a circle around the original point
    """
    if total_at_location <= 1:
        return lat, lng

    # Offset radius in degrees (roughly 50 meters)
    radius = 0.0005

    # Calculate angle for this marker
    angle = (2 * np.pi * offset_index) / total_at_location

    # Calculate offset
    lat_offset = radius * np.cos(angle)
    lng_offset = radius * np.sin(angle)

    return lat + lat_offset, lng + lng_offset


app = Flask(__name__)

# Lazy-load globals
_seasonal_df: pd.DataFrame | None = None
_thresholds: Dict[str, float] | None = None
_feature_names: List[str] | None = None
_booster = None
_calibrator = None
_optimizer: BangkokTaxiOptimizer | None = None


def _ensure_loaded() -> None:
    global _seasonal_df, _thresholds, _feature_names, _booster, _calibrator
    if _thresholds is None:
        _thresholds = _load_thresholds()
    if _seasonal_df is None:
        if SEASONAL_PARQUET.exists():
            _seasonal_df = pd.read_parquet(SEASONAL_PARQUET)
        elif SEASONAL_CSV.exists():
            _seasonal_df = pd.read_csv(SEASONAL_CSV)
        else:
            raise FileNotFoundError(
                f"Missing seasonal baseline at {SEASONAL_PARQUET} or {SEASONAL_CSV}"
            )
    if _feature_names is None:
        if not FEATURE_NAMES_JSON.exists():
            raise FileNotFoundError(
                f"Missing feature_names.json at {FEATURE_NAMES_JSON}"
            )
        _feature_names = json.loads(FEATURE_NAMES_JSON.read_text())
    if _booster is None:
        try:
            import xgboost as xgb
        except Exception as e:
            raise ImportError(
                "xgboost is required. Install with: pip install xgboost"
            ) from e
        _booster = xgb.Booster()
        if not BOOSTER_JSON.exists():
            raise FileNotFoundError(f"Missing booster JSON at {BOOSTER_JSON}")
        _booster.load_model(str(BOOSTER_JSON))
    if _calibrator is None:
        try:
            import joblib
        except Exception as e:
            raise ImportError(
                "joblib is required. Install with: pip install joblib"
            ) from e
        if not CALIBRATOR_JOBLIB.exists():
            raise FileNotFoundError(f"Missing calibrator at {CALIBRATOR_JOBLIB}")
        _calibrator = joblib.load(CALIBRATOR_JOBLIB)


def _ensure_optimizer() -> BangkokTaxiOptimizer:
    """Lazy load the route optimizer"""
    global _optimizer
    if _optimizer is None:
        print(f"Loading optimizer from: {MODELS_DIR}")
        _optimizer = BangkokTaxiOptimizer(models_dir=str(MODELS_DIR))
    return _optimizer


def _build_feature_frame_for_dt(dt: datetime, cells: List[str]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "h3_cell": cells,
            "time_bin": pd.to_datetime(dt.replace(second=0, microsecond=0)),
        }
    )
    fn: List[str] = _feature_names or []
    for col in fn:
        frame[col] = 0.0
    hour = dt.hour
    dow = dt.weekday()
    minute = dt.minute
    is_weekend = 1 if dow >= 5 else 0
    time_setters = {
        "hour": float(hour),
        "dow": float(dow),
        "is_weekend": float(is_weekend),
        "sin_hour": float(np.sin(2 * np.pi * hour / 24.0)),
        "cos_hour": float(np.cos(2 * np.pi * hour / 24.0)),
        "sin_min": float(np.sin(2 * np.pi * minute / 60.0)),
        "cos_min": float(np.cos(2 * np.pi * minute / 60.0)),
    }
    for k, v in time_setters.items():
        if k in frame.columns:
            frame[k] = v
    return frame


def _predict_probs_for_dt(dt: datetime) -> pd.DataFrame:
    sdf = _seasonal_df
    cells = sdf["h3_cell"].dropna().unique().tolist()
    if not cells:
        return pd.DataFrame(columns=["h3_cell", "prob"])
    feats = _build_feature_frame_for_dt(dt, cells)
    try:
        import xgboost as xgb
    except Exception as e:
        raise ImportError(
            "xgboost is required. Install with: pip install xgboost"
        ) from e
    dtest = xgb.DMatrix(feats[_feature_names], feature_names=_feature_names)
    preds_raw = _booster.predict(dtest)
    probs = _calibrator.predict(preds_raw)
    out = pd.DataFrame(
        {
            "h3_cell": feats["h3_cell"].values,
            "prob": probs.astype(float),
        }
    )
    return out


@app.route("/")
def index() -> Any:
    _ensure_loaded()
    return render_template(
        "index.html",
        center=BANGKOK_CENTER,
        zoom=DEFAULT_ZOOM,
        thresholds=_thresholds,
    )


@app.route("/api/seasonal")
def api_seasonal() -> Any:
    _ensure_loaded()
    dt_str = request.args.get("dt")
    try:
        dt = datetime.fromisoformat(dt_str) if dt_str else datetime.utcnow()
    except Exception:
        return (
            jsonify({"error": "Invalid dt; use ISO format like 2024-12-01T09:00"}),
            400,
        )

    dow = dt.weekday()
    mod = _minute_of_day(dt)
    sdf = _seasonal_df
    slice_df = sdf[(sdf["dow"] == dow) & (sdf["minute_of_day"] == mod)][
        ["h3_cell", "prob"]
    ]
    th = _thresholds
    features = []
    for row in slice_df.itertuples(index=False):
        cell = getattr(row, "h3_cell")
        prob = float(getattr(row, "prob"))
        color = _prob_to_color(prob, th)
        if color == "transparent":
            continue
        cells_to_draw: List[str] = [cell]
        if TARGET_H3_RES is not None:
            res_cur = _get_resolution(cell)
            if res_cur >= 0 and TARGET_H3_RES > res_cur:
                cells_to_draw = _children(cell, TARGET_H3_RES)
        for cdraw in cells_to_draw:
            polygon = _cell_to_polygon(cdraw)
            if not polygon:
                continue
            feature = {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [polygon]},
                "properties": {"h3_cell": cdraw, "prob": prob, "color": color},
            }
            features.append(feature)
    return jsonify({"type": "FeatureCollection", "features": features})


@app.route("/api/predict_map")
def api_predict_map() -> Any:
    _ensure_loaded()
    dt_str = request.args.get("dt")
    try:
        dt = datetime.fromisoformat(dt_str) if dt_str else datetime.utcnow()
    except Exception:
        return (
            jsonify({"error": "Invalid dt; use ISO format like 2024-12-01T09:00"}),
            400,
        )
    preds = _predict_probs_for_dt(dt)
    th = _thresholds
    features: List[Dict[str, Any]] = []
    for row in preds.itertuples(index=False):
        cell = getattr(row, "h3_cell")
        prob = float(getattr(row, "prob"))
        color = _prob_to_color(prob, th)
        if color == "transparent":
            continue
        cells_to_draw: List[str] = [cell]
        if TARGET_H3_RES is not None:
            res_cur = _get_resolution(cell)
            if res_cur >= 0 and TARGET_H3_RES > res_cur:
                cells_to_draw = _children(cell, TARGET_H3_RES)
        for cdraw in cells_to_draw:
            polygon = _cell_to_polygon(cdraw)
            if not polygon:
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [polygon]},
                    "properties": {"h3_cell": cdraw, "prob": prob, "color": color},
                }
            )
    return jsonify({"type": "FeatureCollection", "features": features})


@app.route("/api/predict_point")
def api_predict_point() -> Any:
    _ensure_loaded()
    lat_str = request.args.get("lat")
    lon_str = request.args.get("lon")
    dt_str = request.args.get("dt")
    if lat_str is None or lon_str is None:
        return jsonify({"error": "lat and lon are required"}), 400
    try:
        lat = float(lat_str)
        lon = float(lon_str)
    except Exception:
        return jsonify({"error": "lat/lon must be float"}), 400
    try:
        dt = datetime.fromisoformat(dt_str) if dt_str else datetime.utcnow()
    except Exception:
        return (
            jsonify({"error": "Invalid dt; use ISO format like 2024-12-01T09:00"}),
            400,
        )
    try:
        if hasattr(h3, "geo_to_h3"):
            cell = h3.geo_to_h3(lat, lon, 7)
        else:
            cell = h3.latlng_to_cell(lat, lon, 7)
    except Exception:
        return jsonify({"error": "Failed to compute H3 cell"}), 400
    feats = _build_feature_frame_for_dt(dt, [cell])
    try:
        import xgboost as xgb
    except Exception as e:
        raise ImportError(
            "xgboost is required. Install with: pip install xgboost"
        ) from e
    dtest = xgb.DMatrix(feats[_feature_names], feature_names=_feature_names)
    prob = float(_calibrator.predict(_booster.predict(dtest))[0])
    color = _prob_to_color(prob, _thresholds)
    return jsonify(
        {
            "h3_cell": cell,
            "prob": prob,
            "color": color,
            "dt": dt.isoformat(timespec="minutes"),
        }
    )


# ==================== ROUTE OPTIMIZATION ENDPOINTS ====================


@app.route("/api/optimize_route", methods=["POST"])
def api_optimize_route() -> Any:
    """
    Start route optimization using Monte Carlo simulation
    """
    try:
        data = request.get_json()

        start_lat = float(data.get("start_lat"))
        start_lng = float(data.get("start_lng"))
        end_lat = float(data.get("end_lat"))
        end_lng = float(data.get("end_lng"))
        start_time = datetime.fromisoformat(data.get("start_time"))
        end_time = datetime.fromisoformat(data.get("end_time"))
        n_simulations = int(data.get("n_simulations", 500))

        if not (13.4 <= start_lat <= 14.2 and 99.9 <= start_lng <= 101.3):
            return jsonify({"error": "Start location outside Bangkok bounds"}), 400
        if not (13.4 <= end_lat <= 14.2 and 99.9 <= end_lng <= 101.3):
            return jsonify({"error": "End location outside Bangkok bounds"}), 400

        if end_time <= start_time:
            return jsonify({"error": "End time must be after start time"}), 400

        optimizer = _ensure_optimizer()

        print(f"\n{'='*80}")
        print(f"NEW OPTIMIZATION REQUEST")
        print(f"{'='*80}")
        print(f"Start: ({start_lat}, {start_lng})")
        print(f"End: ({end_lat}, {end_lng})")
        print(f"Time: {start_time} to {end_time}")
        print(f"Simulations: {n_simulations}")

        top_routes = optimizer.optimize_route(
            start_location=(start_lat, start_lng),
            end_location=(end_lat, end_lng),
            start_time=start_time,
            end_time=end_time,
            n_simulations=min(n_simulations, 1000),
            top_n=3,
        )

        def serialize_route(route):
            return {
                "total_revenue": float(route["total_revenue"]),
                "total_trips": int(route["total_trips"]),
                "total_distance_km": float(route["total_distance_km"]),
                "total_trip_time_minutes": float(route["total_trip_time_minutes"]),
                "total_idle_time_minutes": float(route["total_idle_time_minutes"]),
                "total_empty_time_minutes": float(route["total_empty_time_minutes"]),
                "arrival_time": route["arrival_time"].isoformat(),
                "time_difference_minutes": float(route["time_difference_minutes"]),
                "time_accuracy": float(route["time_accuracy"]),
                "trips": [
                    {
                        "trip_number": int(trip["trip_number"]),
                        "pickup_zone": trip["pickup_zone"],
                        "dropoff_zone": trip["dropoff_zone"],
                        "pickup_time": trip["pickup_time"],
                        "dropoff_time": trip["dropoff_time"],
                        "distance_km": float(trip["distance_km"]),
                        "duration_minutes": float(trip["duration_minutes"]),
                        "idle_minutes": float(trip["idle_minutes"]),
                        "fare_thb": float(trip["fare_thb"]),
                        "pickup_probability": float(trip["pickup_probability"]),
                    }
                    for trip in route["trips"]
                ],
            }

        result = {
            "success": True,
            "routes": [serialize_route(r) for r in top_routes],
            "parameters": {
                "start_location": {"lat": start_lat, "lng": start_lng},
                "end_location": {"lat": end_lat, "lng": end_lng},
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "n_simulations": n_simulations,
            },
        }

        print(f"\nOptimization complete!")
        print(f"Top route revenue: {top_routes[0]['total_revenue']:.0f} THB")
        print(f"{'='*80}\n")

        return jsonify(result)

    except Exception as e:
        print(f"\nOptimization error: {str(e)}\n")
        return jsonify({"error": str(e)}), 500


@app.route("/api/route_geojson/<int:route_index>", methods=["POST"])
def api_route_geojson(route_index: int) -> Any:
    """
    Convert a route's trips into GeoJSON with OSRM routing.
    Connects start, end, and all intermediate trip points.
    """
    try:
        # ***FIX 1: Get the new payload structure***
        data = request.get_json()
        route = data.get("route", {})
        trips = route.get("trips", [])
        start_location = data.get("start_location")
        end_location = data.get("end_location")

        optimizer = _ensure_optimizer()

        features = []
        colors = ["#28a745", "#007bff", "#ff6600"]
        color = colors[route_index % len(colors)]

        print(
            f"\nGenerating GeoJSON for route #{route_index + 1} with {len(trips)} trips"
        )

        # Track overlapping pickup locations only
        pickup_locations = defaultdict(list)
        for trip in trips:
            pickup_locations[trip["pickup_zone"]].append(trip["trip_number"])

        # ***FIX 2: Add route from Start Point to First Pickup***
        if trips and start_location:
            start_lat, start_lng = start_location["lat"], start_location["lng"]
            first_pickup_lat, first_pickup_lng = optimizer.h3_to_latlng(
                trips[0]["pickup_zone"]
            )
            coords, _ = get_osrm_route(
                start_lng, start_lat, first_pickup_lng, first_pickup_lat
            )
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {"type": "empty_leg", "color": "#FF13F0"},
                }
            )

        # Process each trip
        for i, trip in enumerate(trips):
            trip_num = trip["trip_number"]

            # Get coordinates
            pickup_lat, pickup_lng = optimizer.h3_to_latlng(trip["pickup_zone"])
            dropoff_lat, dropoff_lng = optimizer.h3_to_latlng(trip["dropoff_zone"])

            # ***FIX 3: Add route from previous dropoff to current pickup***
            if i > 0:
                prev_dropoff_lat, prev_dropoff_lng = optimizer.h3_to_latlng(
                    trips[i - 1]["dropoff_zone"]
                )
                coords, _ = get_osrm_route(
                    prev_dropoff_lng, prev_dropoff_lat, pickup_lng, pickup_lat
                )
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": coords},
                        "properties": {"type": "empty_leg", "color": "#FF13F0"},
                    }
                )

            # Calculate offsets for overlapping pickup markers
            pickup_trips_at_location = pickup_locations[trip["pickup_zone"]]
            pickup_offset_index = pickup_trips_at_location.index(trip_num)
            pickup_lat_offset, pickup_lng_offset = calculate_offset_position(
                pickup_lat,
                pickup_lng,
                pickup_offset_index,
                len(pickup_trips_at_location),
            )

            # Get OSRM road route for the actual trip
            route_coords, _ = get_osrm_route(
                pickup_lng, pickup_lat, dropoff_lng, dropoff_lat
            )

            # Create line feature for the trip
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": route_coords},
                    "properties": {
                        "type": "trip_leg",
                        "trip_number": trip_num,
                        "color": color,
                        "route_index": route_index,
                    },
                }
            )

            # Add pickup marker
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [pickup_lng_offset, pickup_lat_offset],
                    },
                    "properties": {
                        "type": "pickup",
                        "trip_number": trip_num,
                        "time": trip["pickup_time"],
                        "fare_thb": trip["fare_thb"],
                        "distance_km": trip["distance_km"],
                        "duration_minutes": trip["duration_minutes"],
                        "color": color,
                        "route_index": route_index,
                        "is_shared": len(pickup_trips_at_location) > 1,
                        "shared_trips": (
                            pickup_trips_at_location
                            if len(pickup_trips_at_location) > 1
                            else None
                        ),
                    },
                }
            )

        # ***FIX 4: Add route from Last Dropoff to End Point***
        if trips and end_location:
            last_dropoff_lat, last_dropoff_lng = optimizer.h3_to_latlng(
                trips[-1]["dropoff_zone"]
            )
            end_lat, end_lng = end_location["lat"], end_location["lng"]
            coords, _ = get_osrm_route(
                last_dropoff_lng, last_dropoff_lat, end_lng, end_lat
            )
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {"type": "empty_leg", "color": "#FF13F0"},
                }
            )

        print(f"Generated {len(features)} features for a complete, connected route.")

        return jsonify({"type": "FeatureCollection", "features": features})

    except Exception as e:
        print(f"Error generating route GeoJSON: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"BANGKOK TAXI ROUTE OPTIMIZER WITH OSRM ROUTING")
    print(f"{'='*80}")
    print(f"App directory: {APP_ROOT}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"OSRM API: {OSRM_API}")
    print(f"Features: Offset overlapping markers + Route labels")
    print(f"Starting server on http://0.0.0.0:8888")
    print(f"{'='*80}\n")

    app.run(host="0.0.0.0", port=8888, debug=True)
