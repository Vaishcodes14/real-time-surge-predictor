from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import joblib
import requests
import os

# ---------------------------------
# App initialization
# ---------------------------------
app = FastAPI(title="Route Demand Surge Predictor")

# ---------------------------------
# Load ML model and zone data
# ---------------------------------
model = joblib.load("lightgbm_surge_model.joblib")
zones = pd.read_csv("zone_centroids.csv")

# ---------------------------------
# Helper: map lat/lon to nearest zone
# ---------------------------------
def latlon_to_zone(lat, lon):
    zones["dist"] = (zones["lat"] - lat) ** 2 + (zones["lon"] - lon) ** 2
    return int(zones.sort_values("dist").iloc[0]["zone_id"])

# ---------------------------------
# Helper: area name → lat/lon (Google Geocoding)
# ---------------------------------
def geocode_area(area_name: str):
    api_key = os.getenv("GOOGLE_API_KEY")
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

# ---------------------------------
# Frontend UI (served from backend)
# ---------------------------------
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return """
<!DOCTYPE html>
<html>
<head>
  <title>Route Demand Surge Predictor</title>
</head>

<body style="font-family: Arial; padding: 40px">

  <h2>🚕 Route Demand Surge Predictor</h2>
  <p>Predict demand surge for trips from A to B</p>

  <input id="from" placeholder="From area" /><br><br>
  <input id="to" placeholder="To area" /><br><br>

  <button onclick="predict()">Predict Surge</button>

  <h3 id="result"></h3>

  <script>
    function predict() {
      fetch("/predict_surge_route", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          origin_area: document.getElementById("from").value,
          destination_area: document.getElementById("to").value,
          timestamp: new Date().toISOString()
        })
      })
      .then(res => res.json())
      .then(data => {
        document.getElementById("result").innerText =
          "Surge Probability: " + data.surge_probability;
      })
      .catch(() => alert("Error predicting surge"));
    }
  </script>

</body>
</html>
"""

# ---------------------------------
# Request schema (A → B by area name)
# ---------------------------------
class RouteAreaRequest(BaseModel):
    origin_area: str
    destination_area: str
    timestamp: datetime

# ---------------------------------
# Prediction endpoint (A → B)
# ---------------------------------
@app.post("/predict_surge_route")
def predict_surge_route(data: RouteAreaRequest):

    # Convert area names to coordinates
    o_lat, o_lon = geocode_area(data.origin_area)
    d_lat, d_lon = geocode_area(data.destination_area)

    # Map to nearest zones
    origin_zone = latlon_to_zone(o_lat, o_lon)
    destination_zone = latlon_to_zone(d_lat, d_lon)

    # Time features
    hour = data.timestamp.hour
    dayofweek = data.timestamp.weekday()
    is_weekend = 1 if dayofweek >= 5 else 0
    is_rush_hour = 1 if hour in [7, 8, 9, 16, 17, 18, 19] else 0

    # Feature vector (must match training features)
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

# ---------------------------------
# Health check
# ---------------------------------
@app.get("/test")
def test():
    return {"status": "ok"}
