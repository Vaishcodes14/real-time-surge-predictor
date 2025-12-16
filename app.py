from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime
import requests
import os

# ----------------------------
# App initialization
# ----------------------------
app = FastAPI(title="Real-Time Route Demand Surge Predictor")

# ----------------------------
# Health check routes
# ----------------------------
@app.get("/")
def home():
    return {"status": "API is running"}

@app.get("/test")
def test():
    return {"test": "ok"}

# ----------------------------
# Load model and zone data
# ----------------------------
model = joblib.load("lightgbm_surge_model.joblib")
zones = pd.read_csv("zone_centroids.csv")

# ----------------------------
# Helper: map lat/lon to nearest zone
# ----------------------------
def latlon_to_zone(lat, lon):
    zones["dist"] = (zones["lat"] - lat) ** 2 + (zones["lon"] - lon) ** 2
    return int(zones.sort_values("dist").iloc[0]["zone_id"])

# ----------------------------
# Helper: area name → lat/lon (Google Geocoding)
# ----------------------------
def geocode_area(area_name: str):
    api_key = os.getenv("AIzaSyCdLCL3NZhOnEtR-n87ia13tJvjABAOpGI")

    if not api_key:
        raise HTTPException(status_code=500, detail="Google API key not configured")

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": area_name,
        "key": api_key
    }

    response = requests.get(url, params=params).json()

    if response["status"] != "OK":
        raise HTTPException(
            status_code=400,
            detail=f"Invalid area name: {area_name}"
        )

    location = response["results"][0]["geometry"]["location"]
    return location["lat"], location["lng"]

# ----------------------------
# Request schema (A → B by area name)
# ----------------------------
class RouteAreaRequest(BaseModel):
    origin_area: str
    destination_area: str
    timestamp: datetime

# ----------------------------
# Prediction endpoint (A → B)
# ----------------------------
@app.post("/predict_surge_route")
def predict_surge_route(data: RouteAreaRequest):

    # Convert area names to coordinates
    o_lat, o_lon = geocode_area(data.origin_area)
    d_lat, d_lon = geocode_area(data.destination_area)

    # Map to zones
    origin_zone = latlon_to_zone(o_lat, o_lon)
    destination_zone = latlon_to_zone(d_lat, d_lon)

    # Time features
    hour = data.timestamp.hour
    dayofweek = data.timestamp.weekday()
    is_weekend = 1 if dayofweek >= 5 else 0
    is_rush_hour = 1 if hour in [7, 8, 9, 16, 17, 18, 19] else 0

    # Feature vector (must match training)
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
        "origin_area": data.origin_area,
        "destination_area": data.destination_area,
        "origin_zone": origin_zone,
        "destination_zone": destination_zone,
        "surge_probability": round(float(surge_prob), 3)
    }
