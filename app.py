from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

app = FastAPI(title="Real-Time Demand Surge Predictor")

# Load trained model
model = joblib.load("lightgbm_surge_model.joblib")

# Load zone centroids
zones = pd.read_csv("zone_centroids.csv")

# Convert live GPS to nearest zone
def latlon_to_zone(lat, lon):
    zones["dist"] = (zones["lat"] - lat)**2 + (zones["lon"] - lon)**2
    return int(zones.sort_values("dist").iloc[0]["zone_id"])

# Request format
class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    timestamp: datetime

# Prediction API
@app.post("/predict_surge")
def predict_surge(data: LocationRequest):

    zone_id = latlon_to_zone(data.latitude, data.longitude)

    hour = data.timestamp.hour
    dayofweek = data.timestamp.weekday()

    is_weekend = 1 if dayofweek >= 5 else 0
    is_rush_hour = 1 if hour in [7,8,9,16,17,18,19] else 0

    # Feature vector (matches training)
    features = pd.DataFrame([{
        "pickup_count": 0,
        "od_trip_count": 0,
        "avg_travel_time": 0,
        "avg_speed": 0,
        "hour": hour,
        "dayofweek": dayofweek,
        "is_weekend": is_weekend,
        "is_rush_hour": is_rush_hour
    }])

    surge_prob = model.predict_proba(features)[0][1]

    return {
        "zone_id": zone_id,
        "surge_probability": round(float(surge_prob), 3)
    }
