from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import joblib
from datetime import datetime
import requests
import os

app = FastAPI(title="Route Demand Surge Predictor")

# -----------------------------
# Load model & zones
# -----------------------------
model = joblib.load("lightgbm_surge_model.joblib")
zones = pd.read_csv("zone_centroids.csv")

# -----------------------------
# Helpers
# -----------------------------
def geocode_place(place):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Google API key missing")

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": place, "key": api_key}
    res = requests.get(url, params=params).json()

    if res["status"] != "OK":
        raise HTTPException(status_code=400, detail=f"Invalid place: {place}")

    loc = res["results"][0]["geometry"]["location"]
    return loc["lat"], loc["lng"]


def latlon_to_zone(lat, lon):
    zones["dist"] = (zones["lat"] - lat) ** 2 + (zones["lon"] - lon) ** 2
    return int(zones.sort_values("dist").iloc[0]["zone_id"])

# -----------------------------
# Request schema
# -----------------------------
class RouteRequest(BaseModel):
    from_area: str
    to_area: str

# -----------------------------
# Homepage (AREA NAMES UI)
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
<p>Predict demand surge for trips from A to B</p>

<input id="from" placeholder="From area (e.g. Times Square)" style="width:300px"><br><br>
<input id="to" placeholder="To area (e.g. JFK Airport)" style="width:300px"><br><br>

<button onclick="predict()">Predict Surge</button>

<h3 id="result"></h3>

<script>
function predict() {
  fetch("/predict_surge", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      from_area: document.getElementById("from").value,
      to_area: document.getElementById("to").value
    })
  })
  .then(res => res.json())
  .then(data => {
    document.getElementById("result").innerText =
      "Surge Probability: " + data.surge_probability;
  })
  .catch(() => {
    document.getElementById("result").innerText = "Error predicting surge";
  });
}
</script>

</body>
</html>
"""

# -----------------------------
# Prediction API
# -----------------------------
@app.post("/predict_surge")
def predict_surge(req: RouteRequest):

    from_lat, from_lon = geocode_place(req.from_area)
    to_lat, to_lon = geocode_place(req.to_area)

    from_zone = latlon_to_zone(from_lat, from_lon)
    to_zone = latlon_to_zone(to_lat, to_lon)

    now = datetime.now()
    hour = now.hour
    day = now.weekday()

    features = pd.DataFrame([{
        "pickup_count": 0,
        "od_trip_count": 0,
        "avg_travel_time": 0,
        "avg_speed": 0,
        "hour": hour,
        "dayofweek": day,
        "is_weekend": int(day >= 5),
        "is_rush_hour": int(hour in [7,8,9,16,17,18,19])
    }])

    prob = model.predict_proba(features)[0][1]

    return {
        "from_zone": from_zone,
        "to_zone": to_zone,
        "surge_probability": round(float(prob), 3)
    }
