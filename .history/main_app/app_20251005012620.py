from pathlib import Path
from datetime import datetime
from typing import Any, List, Tuple
import numpy as np
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

# Model directory for optimizer (same directory as app.py)
MODELS_DIR = APP_ROOT / "models"

# Map defaults
BANGKOK_CENTER = [13.7563, 100.5018]
DEFAULT_ZOOM = 12

# OSRM API endpoint
OSRM_API = "http://router.project-osrm.org/route/v1/driving/"


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
_optimizer: BangkokTaxiOptimizer | None = None


def _ensure_optimizer() -> BangkokTaxiOptimizer:
    """Lazy load the route optimizer"""
    global _optimizer
    if _optimizer is None:
        print(f"Loading optimizer from: {MODELS_DIR}")
        _optimizer = BangkokTaxiOptimizer(models_dir=str(MODELS_DIR))
    return _optimizer


@app.route("/")
def index() -> Any:
    return render_template(
        "index.html",
        center=BANGKOK_CENTER,
        zoom=DEFAULT_ZOOM,
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

        # Add route from Start Point to First Pickup
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

            # Add route from previous dropoff to current pickup
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

        # Add route from Last Dropoff to End Point
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
