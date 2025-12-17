from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import requests
from datetime import datetime
import streamlit as st
st.write("🚀 App is starting...")


# -----------------------------
# App init
# -----------------------------
app = FastAPI(title="Route Demand Surge Predictor")

# -----------------------------
# Load model & zone data
# -----------------------------
model = joblib.load("lightgbm_surge_model.joblib")
zones = pd.read_csv("zone_centroids.csv")

# Ensure expected columns exist
# zone_centroids.csv MUST have: zone_id, lat, lon
required_cols = {"zone_id", "lat", "lon"}
if not required_cols.issubset(zones.columns):
    raise ValueError("zone_centroids.csv must contain zone_id, lat, lon columns")

# -----------------------------
# Helpers
# -----------------------------
def geocode_place(place_name: str):
    """Convert place name to lat/lon using OpenStreetMap"""
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place_name,
        "format": "json",
        "limit": 1
    }
    headers = {"User-Agent": "surge-predictor-app"}
    resp = requests.get(url, params=params, headers=headers, timeout=10)

    data = resp.json()
    if not data:
        raise HTTPException(status_code=400, detail=f"Invalid place: {place_name}")

    return float(data[0]["lat"]), float(data[0]["lon"])


def latlon_to_zone(lat, lon):
    zones["dist"] = (zones["lat"] - lat) ** 2 + (zones["lon"] - lon) ** 2
    return int(zones.sort_values("dist").iloc[0]["zone_id"])


def build_features(timestamp):
    hour = timestamp.hour
    dayofweek = timestamp.weekday()
    return pd.DataFrame([{
        "pickup_count": 0,
        "od_trip_count": 0,
        "avg_travel_time": 0,
        "avg_speed": 0,
        "hour": hour,
        "dayofweek": dayofweek,
        "is_weekend": 1 if dayofweek >= 5 else 0,
        "is_rush_hour": 1 if hour in [7, 8, 9, 16, 17, 18, 19] else 0
    }])

# -----------------------------
# Request schema
# -----------------------------
class RouteRequest(BaseModel):
    from_area: str
    to_area: str

# -----------------------------
# API routes
# -----------------------------
@app.get("/")
def health():
    return {"status": "API is running"}

@app.post("/predict_surge")
def predict_surge(req: RouteRequest):
    # Convert place names → coordinates
    from_lat, from_lon = geocode_place(req.from_area)
    to_lat, to_lon = geocode_place(req.to_area)

    from_zone = latlon_to_zone(from_lat, from_lon)
    to_zone = latlon_to_zone(to_lat, to_lon)

    features = build_features(datetime.utcnow())
    surge_prob = model.predict_proba(features)[0][1]

    return {
        "from_area": req.from_area,
        "to_area": req.to_area,
        "from_zone": from_zone,
        "to_zone": to_zone,
        "surge_probability": round(float(surge_prob), 3)
    }

# -----------------------------
# Frontend (served by backend)
# -----------------------------
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html>
<head>
  <title>Route Demand Surge Predictor</title>
</head>
<body style="font-family: Arial; padding: 40px">

<h2>🚕 Route Demand Surge Predictor</h2>

<input id="from" placeholder="From area (e.g. JFK Airport)" style="width:300px;padding:8px"><br><br>
<input id="to" placeholder="To area (e.g. Times Square)" style="width:300px;padding:8px"><br><br>

<button onclick="predict()">Predict Surge</button>

<h3 id="result"></h3>

<script>
function predict() {
  const fromArea = document.getElementById("from").value;
  const toArea = document.getElementById("to").value;

  fetch("/predict_surge", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      from_area: fromArea,
      to_area: toArea
    })
  })
  .then(r => r.json())
  .then(d => {
    document.getElementById("result").innerText =
      "Surge Probability: " + d.surge_probability;
  })
  .catch(() => {
    document.getElementById("result").innerText = "Error predicting surge";
  });
}
</script>

</body>
</html>
"""
