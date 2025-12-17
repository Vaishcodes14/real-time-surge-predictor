from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import joblib
from datetime import datetime
import math

# -----------------------------
# App initialization
# -----------------------------
app = FastAPI(title="Route Demand Surge Predictor")

# -----------------------------
# Load model & zone centroids
# -----------------------------
model = joblib.load("lightgbm_surge_model.joblib")
zones = pd.read_csv("zone_centroids.csv")

# -----------------------------
# Helper: lat/lon → nearest zone
# -----------------------------
def latlon_to_zone(lat, lon):
    zones["dist"] = (zones["lat"] - lat) ** 2 + (zones["lon"] - lon) ** 2
    return int(zones.sort_values("dist").iloc[0]["zone_id"])

# -----------------------------
# Request schema (A → B)
# -----------------------------
class RouteRequest(BaseModel):
    from_lat: float
    from_lon: float
    to_lat: float
    to_lon: float

# -----------------------------
# Frontend UI
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
  <title>Route Demand Surge Predictor</title>
</head>

<body style="font-family: Arial; padding: 40px">

  <h2>🚕 Route Demand Surge Predictor</h2>
  <p>Enter latitude & longitude for source and destination</p>

  <h4>From</h4>
  <input id="from_lat" placeholder="From latitude" /><br><br>
  <input id="from_lon" placeholder="From longitude" /><br><br>

  <h4>To</h4>
  <input id="to_lat" placeholder="To latitude" /><br><br>
  <input id="to_lon" placeholder="To longitude" /><br><br>

  <button onclick="predict()">Predict Surge</button>

  <h3 id="result"></h3>

  <script>
    function predict() {
      fetch("/predict_surge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          from_lat: parseFloat(document.getElementById("from_lat").value),
          from_lon: parseFloat(document.getElementById("from_lon").value),
          to_lat: parseFloat(document.getElementById("to_lat").value),
          to_lon: parseFloat(document.getElementById("to_lon").value)
        })
      })
      .then(res => res.json())
      .then(data => {
        if (data.surge_probability !== undefined) {
          document.getElementById("result").innerText =
            "Surge Probability: " + data.surge_probability;
        } else {
          document.getElementById("result").innerText =
            "Error: " + JSON.stringify(data);
        }
      })
      .catch(() => {
        document.getElementById("result").innerText = "Backend error";
      });
    }
  </script>

</body>
</html>
"""

# -----------------------------
# Health check
# -----------------------------
@app.get("/test")
def test():
    return {"status": "ok"}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict_surge")
def predict_surge(req: RouteRequest):

    from_zone = latlon_to_zone(req.from_lat, req.from_lon)
    to_zone = latlon_to_zone(req.to_lat, req.to_lon)

    now = datetime.now()
    hour = now.hour
    dayofweek = now.weekday()
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
        "from_zone": from_zone,
        "to_zone": to_zone,
        "surge_probability": round(float(surge_prob), 3)
    }
