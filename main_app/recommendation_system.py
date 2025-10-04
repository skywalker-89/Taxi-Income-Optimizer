"""
Bangkok Taxi Route Optimizer using Monte Carlo Simulation
Recommends top 3 routes with REVENUE as primary priority, TIME as secondary
"""

import pandas as pd
import numpy as np
import h3
import xgboost as xgb
import joblib
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import warnings
from pathlib import Path


warnings.filterwarnings("ignore")


class BangkokTaxiOptimizer:
    """
    Route optimizer for Bangkok taxi drivers using trained ML models
    Two-tier ranking: 1) Revenue (highest first), 2) Time accuracy (closest to end_time)
    """

    def __init__(
        self,
        models_dir: str = None,
    ):
        """Load all trained models"""
        print("Loading trained models...")

        # Use relative path if not provided
        if models_dir is None:
            models_dir = Path(__file__).parent / "models"
        else:
            models_dir = Path(models_dir)

        # Load Next Destination Model
        self.destination_model = xgb.Booster()
        self.destination_model.load_model(
            f"{models_dir}/xgboost_destination_model.json"
        )

        # Load encoders for destination model
        encoders_dict = joblib.load(f"{models_dir}/model_encoders.pkl")
        self.destination_label_encoders = encoders_dict["label_encoders"]
        self.destination_target_encoder = encoders_dict["target_encoder"]
        self.destination_feature_names = encoders_dict["feature_names"]
        self.top_destinations = encoders_dict["top_destinations"]

        # Load Inter-Zone Travel Time Model
        self.interzone_model = joblib.load(f"{models_dir}/xgboost_inter_zone_model.pkl")
        self.interzone_features = joblib.load(f"{models_dir}/model_feature_names.pkl")

        # Load Trip Duration Model
        self.duration_model = joblib.load(f"{models_dir}/duration_model.joblib")

        # Load Trip Distance Model
        self.distance_model = xgb.XGBRegressor()
        self.distance_model.load_model(f"{models_dir}/best_xgb_trip_distance.model")

        # Load Pickup Rate Model
        self.pickup_model = xgb.Booster()
        self.pickup_model.load_model(f"{models_dir}/xgb_pickup_5min.json")
        self.pickup_calibrator = joblib.load(f"{models_dir}/isotonic_calibrator.joblib")

        print("✅ All models loaded successfully!")

        # Thai taxi fare structure (2024 rates)
        self.fare_structure = {
            "base_fare": 35.0,
            "rate_1_10km": 6.50,
            "rate_10_20km": 7.00,
            "rate_20_40km": 8.00,
            "rate_40_60km": 8.50,
            "rate_60_80km": 9.00,
            "rate_80plus": 10.50,
        }

        # Bangkok region bounds
        self.bkk_bounds = {
            "min_lat": 13.4,
            "max_lat": 14.2,
            "min_lon": 99.9,
            "max_lon": 101.3,
        }

        # Bangkok center for distance calculations
        self.bkk_center = (13.7563, 100.5018)

    def calculate_fare(self, distance_km: float) -> float:
        """Calculate taxi fare based on distance using Thai taxi rates"""
        if distance_km <= 0:
            return 0

        fare = self.fare_structure["base_fare"]

        if distance_km > 1:
            fare += min(distance_km - 1, 9) * self.fare_structure["rate_1_10km"]
        if distance_km > 10:
            fare += min(distance_km - 10, 10) * self.fare_structure["rate_10_20km"]
        if distance_km > 20:
            fare += min(distance_km - 20, 20) * self.fare_structure["rate_20_40km"]
        if distance_km > 40:
            fare += min(distance_km - 40, 20) * self.fare_structure["rate_40_60km"]
        if distance_km > 60:
            fare += min(distance_km - 60, 20) * self.fare_structure["rate_60_80km"]
        if distance_km > 80:
            fare += (distance_km - 80) * self.fare_structure["rate_80plus"]

        return fare

    def haversine_distance(
        self, lon1: float, lat1: float, lon2: float, lat2: float
    ) -> float:
        """Calculate distance in km between two points"""
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return c * 6371

    def h3_to_latlng(self, h3_cell: str) -> Tuple[float, float]:
        """Convert H3 cell to lat/lng coordinates"""
        return h3.cell_to_latlng(h3_cell)

    def latlng_to_h3(self, lat: float, lng: float, resolution: int = 7) -> str:
        """Convert lat/lng to H3 cell"""
        return h3.latlng_to_cell(lat, lng, resolution)

    def predict_pickup_probability(self, h3_zone: str, timestamp: datetime) -> float:
        """Predict probability of getting a pickup in a zone at given time"""
        hour = timestamp.hour
        minute = timestamp.minute
        day_of_week = timestamp.weekday()

        # Create features for pickup model
        features = {
            "hour": hour,  # First!
            "dow": day_of_week,  # Second!
            "is_weekend": 1 if day_of_week >= 5 else 0,  # Third!
            "sin_hour": np.sin(2 * np.pi * hour / 24),
            "cos_hour": np.cos(2 * np.pi * hour / 24),
            "sin_min": np.sin(2 * np.pi * minute / 60),
            "cos_min": np.cos(2 * np.pi * minute / 60),
            "pickup_cnt": 0,
            "dropoff_cnt": 0,
            "net_inflow": 0,
            "pickup_lag_1": 0,
            "pickup_lag_2": 0,
            "pickup_lag_3": 0,
            "pickup_ma_6": 0,
            "pickup_ma_12": 0,
            "dropoff_lag_1": 0,
            "dropoff_lag_2": 0,
            "dropoff_lag_3": 0,
            "dropoff_ma_6": 0,
            "dropoff_ma_12": 0,
            "nbr1_mean_pickup_cnt": 0.0,
            "nbr2_mean_pickup_cnt": 0.0,
            "nbr1_mean_pickup_lag_1": 0.0,
            "nbr2_mean_pickup_lag_1": 0.0,
            "nbr1_mean_pickup_lag_2": 0.0,
            "nbr2_mean_pickup_lag_2": 0.0,
            "nbr1_mean_pickup_lag_3": 0.0,
            "nbr2_mean_pickup_lag_3": 0.0,
            "nbr1_mean_dropoff_cnt": 0.0,
            "nbr2_mean_dropoff_cnt": 0.0,
        }

        # Add neighbor features
        neighbor_cols = [
            "nbr1_mean_pickup_cnt",
            "nbr2_mean_pickup_cnt",
            "nbr1_mean_pickup_lag_1",
            "nbr2_mean_pickup_lag_1",
            "nbr1_mean_pickup_lag_2",
            "nbr2_mean_pickup_lag_2",
            "nbr1_mean_pickup_lag_3",
            "nbr2_mean_pickup_lag_3",
            "nbr1_mean_dropoff_cnt",
            "nbr2_mean_dropoff_cnt",
        ]
        for col in neighbor_cols:
            features[col] = 0.0

        feature_names = list(features.keys())
        X = np.array([list(features.values())]).reshape(1, -1)
        dmatrix = xgb.DMatrix(X, feature_names=feature_names)

        # Predict and calibrate
        raw_pred = self.pickup_model.predict(dmatrix)[0]
        calibrated_pred = self.pickup_calibrator.predict([raw_pred])[0]

        return max(0.01, min(0.99, calibrated_pred))

    def predict_next_destination(
        self, origin_h3: str, timestamp: datetime, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Predict top-K most likely destinations from origin"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        day = timestamp.day
        is_weekend = 1 if day_of_week >= 5 else 0

        # Create features
        features = {
            "origin_h3": origin_h3,
            "pickup_hour": hour,
            "pickup_day_of_week": day_of_week,
            "pickup_is_weekend": is_weekend,
            "pickup_day": day,
            "is_rush_hour": 1 if (7 <= hour <= 9) or (17 <= hour <= 20) else 0,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "day_sin": np.sin(2 * np.pi * day_of_week / 7),
            "day_cos": np.cos(2 * np.pi * day_of_week / 7),
            "day_of_month_sin": np.sin(2 * np.pi * day / 31),
            "day_of_month_cos": np.cos(2 * np.pi * day / 31),
            "time_period": self._get_time_period(hour),
            "distance_from_center_km": 0,
            "zone_historical_trips": 0,
            "zone_avg_duration": 0,
            "zone_avg_distance": 0,
            "od_pair_historical_count": 0,
            "origin_to_dest_popularity": 0,
            "od_rush_historical_count": 0,
            "od_timeperiod_historical_count": 0,
        }

        # Get origin center coordinates
        try:
            origin_lat, origin_lon = self.h3_to_latlng(origin_h3)
            features["origin_center_lat"] = origin_lat
            features["origin_center_lon"] = origin_lon
            features["distance_from_center_km"] = self.haversine_distance(
                origin_lon, origin_lat, self.bkk_center[1], self.bkk_center[0]
            )
        except:
            features["origin_center_lat"] = 13.7563
            features["origin_center_lon"] = 100.5018

        # Encode categorical features
        X_dict = features.copy()
        if "origin_h3" in self.destination_label_encoders:
            le = self.destination_label_encoders["origin_h3"]
            if origin_h3 in le.classes_:
                X_dict["origin_h3"] = le.transform([origin_h3])[0]
            else:
                X_dict["origin_h3"] = -1

        if "time_period" in self.destination_label_encoders:
            le = self.destination_label_encoders["time_period"]
            if features["time_period"] in le.classes_:
                X_dict["time_period"] = le.transform([features["time_period"]])[0]
            else:
                X_dict["time_period"] = -1

        # Create feature array in correct order
        X = np.array([[X_dict.get(f, 0) for f in self.destination_feature_names]])

        # Predict
        dmatrix = xgb.DMatrix(X, feature_names=self.destination_feature_names)
        pred_proba = self.destination_model.predict(dmatrix)[0]

        # Get top-K destinations
        top_k_indices = np.argsort(pred_proba)[-top_k:][::-1]
        top_k_destinations = []

        for idx in top_k_indices:
            dest_h3 = self.destination_target_encoder.classes_[idx]
            if dest_h3 != "OTHER":
                top_k_destinations.append((dest_h3, pred_proba[idx]))

        return top_k_destinations[:top_k]

    def predict_interzone_travel_time(
        self, origin_h3: str, dest_h3: str, timestamp: datetime
    ) -> float:
        """Predict empty travel time between zones in minutes"""
        # Get coordinates
        origin_lat, origin_lon = self.h3_to_latlng(origin_h3)
        dest_lat, dest_lon = self.h3_to_latlng(dest_h3)

        # Calculate direct distance
        direct_distance = self.haversine_distance(
            origin_lon, origin_lat, dest_lon, dest_lat
        )

        # Time features
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0

        # Create features
        features = {
            "direct_distance_km": direct_distance,
            "bearing_sin": 0,
            "bearing_cos": 0,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * day_of_week / 7),
            "dow_cos": np.cos(2 * np.pi * day_of_week / 7),
            "is_weekend": is_weekend,
            "is_morning_rush": 1 if 7 <= hour <= 9 else 0,
            "is_evening_rush": 1 if 17 <= hour <= 19 else 0,
            "is_rush_hour": 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0,
            "origin_zone_encoded": 0,
            "destination_zone_encoded": 0,
            "route_pair_id": 0,
        }

        # Add time period dummies
        time_period = self._get_time_period(hour)
        for period in [
            "evening",
            "evening_rush",
            "late_night",
            "midday",
            "morning_rush",
        ]:
            features[f"time_period_{period}"] = 1 if time_period == period else 0

        # Create feature array
        X = pd.DataFrame([features])
        X = X.reindex(columns=self.interzone_features, fill_value=0)

        # Predict
        travel_time = self.interzone_model.predict(X)[0]
        return max(1.0, travel_time)

    def predict_trip_duration(
        self, origin_h3: str, dest_h3: str, distance_km: float, timestamp: datetime
    ) -> float:
        """Predict trip duration in minutes"""
        origin_lat, origin_lon = self.h3_to_latlng(origin_h3)
        dest_lat, dest_lon = self.h3_to_latlng(dest_h3)

        hour = timestamp.hour
        day_of_week = timestamp.weekday()

        # Estimate average speed
        avg_speed = 25.0

        # Calculate straight-line distance
        straight_dist = self.haversine_distance(
            origin_lon, origin_lat, dest_lon, dest_lat
        )

        # Simple duration estimate
        duration = (distance_km / avg_speed) * 60

        # Add time-of-day adjustment
        if (7 <= hour <= 9) or (17 <= hour <= 19):
            duration *= 1.3

        return max(5.0, duration)

    def predict_trip_distance(self, origin_h3: str, dest_h3: str) -> float:
        """Predict trip distance in km"""
        origin_lat, origin_lon = self.h3_to_latlng(origin_h3)
        dest_lat, dest_lon = self.h3_to_latlng(dest_h3)

        # Calculate straight-line distance
        straight_dist = self.haversine_distance(
            origin_lon, origin_lat, dest_lon, dest_lat
        )

        # Apply typical route factor
        trip_distance = straight_dist * 1.4

        return max(0.5, trip_distance)

    def _get_time_period(self, hour: int) -> str:
        """Get time period label from hour"""
        if 5 <= hour < 10:
            return "morning_rush"
        elif 10 <= hour < 16:
            return "midday"
        elif 16 <= hour < 20:
            return "evening_rush"
        elif 20 <= hour < 24:
            return "evening"
        else:
            return "late_night"

    def simulate_route(
        self,
        start_h3: str,
        end_h3: str,
        start_time: datetime,
        end_time: datetime,
        max_trips: int = 20,
    ) -> Dict:
        """
        Simulate a single route using Monte Carlo approach
        Returns route details with revenue and timing
        """
        current_zone = start_h3
        current_time = start_time
        time_limit = end_time

        trips = []
        total_revenue = 0
        total_distance = 0
        total_trip_time = 0
        total_empty_time = 0
        total_idle_time = 0

        trip_count = 0

        while current_time < time_limit and trip_count < max_trips:
            # Get potential destinations
            potential_dests = self.predict_next_destination(
                current_zone, current_time, top_k=10
            )

            if not potential_dests:
                break

            # Randomly select destination weighted by probability
            dest_zones = [d[0] for d in potential_dests]
            dest_probs = np.array([d[1] for d in potential_dests])
            dest_probs = dest_probs / dest_probs.sum()

            selected_dest = np.random.choice(dest_zones, p=dest_probs)

            # Predict pickup probability
            pickup_prob = self.predict_pickup_probability(current_zone, current_time)

            # Estimate idle time
            idle_minutes = np.random.exponential(
                scale=(1.0 / max(0.1, pickup_prob)) * 5
            )
            idle_minutes = min(idle_minutes, 30)

            current_time += timedelta(minutes=float(idle_minutes))

            if current_time >= time_limit:
                break

            # Predict trip details
            trip_distance = self.predict_trip_distance(current_zone, selected_dest)
            trip_duration = self.predict_trip_duration(
                current_zone, selected_dest, trip_distance, current_time
            )
            trip_fare = self.calculate_fare(trip_distance)

            # Check if trip fits in remaining time
            time_after_trip = current_time + timedelta(minutes=float(trip_duration))
            return_time = self.predict_interzone_travel_time(
                selected_dest, end_h3, time_after_trip
            )

            if time_after_trip + timedelta(
                minutes=float(return_time)
            ) > time_limit + timedelta(minutes=10):
                break

            # Accept trip
            trips.append(
                {
                    "trip_number": trip_count + 1,
                    "pickup_zone": current_zone,
                    "dropoff_zone": selected_dest,
                    "pickup_time": current_time.strftime("%H:%M"),
                    "dropoff_time": (
                        current_time + timedelta(minutes=float(trip_duration))
                    ).strftime("%H:%M"),
                    "distance_km": trip_distance,
                    "duration_minutes": trip_duration,
                    "idle_minutes": idle_minutes,
                    "fare_thb": trip_fare,
                    "pickup_probability": pickup_prob,
                }
            )

            total_revenue += trip_fare
            total_distance += trip_distance
            total_trip_time += trip_duration
            total_idle_time += idle_minutes

            # Move to next zone
            current_zone = selected_dest
            current_time += timedelta(minutes=float(trip_duration))
            trip_count += 1

        # Calculate return journey
        if current_zone != end_h3:
            return_time = self.predict_interzone_travel_time(
                current_zone, end_h3, current_time
            )
            total_empty_time += return_time
            arrival_time = current_time + timedelta(minutes=float(return_time))
        else:
            arrival_time = current_time

        return {
            "trips": trips,
            "total_revenue": total_revenue,
            "total_trips": len(trips),
            "total_distance_km": total_distance,
            "total_trip_time_minutes": total_trip_time,
            "total_idle_time_minutes": total_idle_time,
            "total_empty_time_minutes": total_empty_time,
            "arrival_time": arrival_time,
            "time_difference_minutes": (arrival_time - end_time).total_seconds() / 60,
        }

    def optimize_route(
        self,
        start_location: Tuple[float, float],
        end_location: Tuple[float, float],
        start_time: datetime,
        end_time: datetime,
        n_simulations: int = 1000,
        top_n: int = 3,
    ) -> List[Dict]:
        """
        Find top N routes using Monte Carlo simulation
        TWO-TIER RANKING: 1) Revenue (highest), 2) Time accuracy (closest to end_time)

        Args:
            start_location: (lat, lng) tuple
            end_location: (lat, lng) tuple
            start_time: datetime object
            end_time: datetime object
            n_simulations: number of Monte Carlo simulations
            top_n: number of top routes to return

        Returns:
            List of top N routes sorted by revenue first, then time accuracy
        """
        print(f"\n🚕 Starting Route Optimization")
        print(f"📍 Start: {start_location} → End: {end_location}")
        print(
            f"⏰ Time Window: {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}"
        )
        print(f"🎲 Running {n_simulations} Monte Carlo simulations...")
        print(f"🎯 Ranking: PRIMARY=Revenue | SECONDARY=Time Accuracy\n")

        # Convert locations to H3
        start_h3 = self.latlng_to_h3(start_location[0], start_location[1])
        end_h3 = self.latlng_to_h3(end_location[0], end_location[1])

        print(f"H3 Zones: {start_h3} → {end_h3}")

        # Run simulations
        all_routes = []

        for i in range(n_simulations):
            if (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{n_simulations} simulations completed...")

            route = self.simulate_route(start_h3, end_h3, start_time, end_time)

            # Calculate time accuracy score (lower is better)
            route["time_accuracy"] = abs(route["time_difference_minutes"])

            all_routes.append(route)

        # TWO-TIER SORTING
        # Primary: Revenue (descending - highest first)
        # Secondary: Time Accuracy (ascending - closest to target)
        all_routes.sort(key=lambda x: (-x["total_revenue"], x["time_accuracy"]))

        # Get top N routes
        top_routes = all_routes[:top_n]

        print(f"\n✅ Optimization complete!")
        print(f"📊 Generated {len(all_routes)} valid routes")
        print(f"🏆 Returning top {top_n} routes\n")

        return top_routes

    def print_route_summary(self, routes: List[Dict]):
        """Print formatted summary of top routes"""
        for rank, route in enumerate(routes, 1):
            print(f"\n{'='*80}")
            print(f"🏆 ROUTE #{rank}")
            print(f"{'='*80}")
            print(
                f"💰 Total Revenue: {route['total_revenue']:.2f} THB (PRIMARY RANKING)"
            )
            print(
                f"⏰ Time Accuracy: {route['time_accuracy']:.1f} min difference (SECONDARY RANKING)"
            )
            print(f"🚖 Total Trips: {route['total_trips']}")
            print(f"🛣️  Total Distance: {route['total_distance_km']:.2f} km")
            print(f"⏱️  Total Trip Time: {route['total_trip_time_minutes']:.1f} minutes")
            print(f"⏸️  Total Idle Time: {route['total_idle_time_minutes']:.1f} minutes")
            print(
                f"🚗 Total Empty Drive Time: {route['total_empty_time_minutes']:.1f} minutes"
            )
            print(f"🏁 Estimated Arrival: {route['arrival_time'].strftime('%H:%M')}")

            # Time difference indicator
            time_diff = route["time_difference_minutes"]
            if abs(time_diff) <= 10:
                time_status = "✅ ON TIME"
            elif time_diff > 0:
                time_status = "⚠️ LATE"
            else:
                time_status = "⚠️ EARLY"
            print(f"📍 Time Difference: {time_diff:+.1f} minutes ({time_status})")

            print(f"\n📋 Trip Details:")
            print(f"{'-'*80}")

            for trip in route["trips"]:
                print(
                    f"  Trip {trip['trip_number']}: "
                    f"{trip['pickup_zone'][:8]}... → {trip['dropoff_zone'][:8]}..."
                )
                print(
                    f"    ⏰ {trip['pickup_time']} - {trip['dropoff_time']} "
                    f"({trip['duration_minutes']:.1f} min)"
                )
                print(
                    f"    💰 {trip['fare_thb']:.2f} THB | "
                    f"📏 {trip['distance_km']:.2f} km | "
                    f"⏸️ Idle: {trip['idle_minutes']:.1f} min"
                )
                print()


# Example Usage
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = BangkokTaxiOptimizer(
        models_dir="/Users/jul/Desktop/uni/Data Analytics/Taxi-Income-Optimizer/main_app/models"
    )

    # Define parameters
    start_location = (13.6644, 100.6026)  # Bangna (lat, lng)
    end_location = (13.6644, 100.6026)  # Return to Bangna
    start_time = datetime(2024, 12, 15, 9, 0)  # 9:00 AM
    end_time = datetime(2024, 12, 15, 18, 0)  # 6:00 PM

    # Run optimization
    top_routes = optimizer.optimize_route(
        start_location=start_location,
        end_location=end_location,
        start_time=start_time,
        end_time=end_time,
        n_simulations=500,
        top_n=3,
    )

    # Print results
    optimizer.print_route_summary(top_routes)

    # Save to CSV for further analysis
    import json

    def convert_to_json_serializable(obj):
        """Convert numpy types to Python types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {
                key: convert_to_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        return obj

    with open("top_routes.json", "w") as f:
        routes_serializable = convert_to_json_serializable(top_routes)
        json.dump(routes_serializable, f, indent=2)

    print("\n💾 Results saved to 'top_routes.json'")
