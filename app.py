from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import requests
from datetime import datetime
import os
from math import radians, cos, sin, asin, sqrt

# -------------------------------------------------
# App init
# -------------------------------------------------
app = FastAPI(title="Ride Demand Surge API")

# -------------------------------------------------
# Load model & zone data
# -------------------------------------------------
model = joblib.load("lightgbm_surge_model.joblib")
zones = pd.read_csv("zone_centroids.csv")

GOOGLE_API_KEY = os.getenv("AIzaSyCdLCL3NZhOnEtR-n87ia13tJvjABAOpGI")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set in environment variables")

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def geocode_google(place):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": place,
        "key": GOOGLE_API_KEY
    }

    r = requests.get(url, params=params, timeout=10)
    data = r.json()

    if data.get("status") != "OK":
        return None

    loc = data["results"][0]["geometry"]["location"]
    return loc["lat"], loc["lng"]


def latlon_to_zone(lat, lon):
    df = zones.copy()
    df["dist"] = (df["lat"] - lat) ** 2 + (df["lon"] - lon) ** 2
    return int(df.sort_values("dist").iloc[0]["zone_id"])


def estimate_demand(place):
    place = place.lower()
    if any(w in place for w in ["airport", "station", "downtown", "mall", "terminal"]):
        return 140
    if any(w in place for w in ["square", "road", "avenue", "park"]):
        return 80
    return 30


def build_features(from_place, to_place):
    now = datetime.utcnow()
    hour = now.hour
    day = now.weekday()

    pickup = estimate_demand(from_place)
    drop = estimate_demand(to_place)

    avg_speed = 15 if pickup > 100 else 30

    return pd.DataFrame([{
        "pickup_count": pickup,
        "od_trip_count": drop,
        "avg_travel_time": 0,
        "avg_speed": avg_speed,
        "hour": hour,
        "dayofweek": day,
        "is_weekend": int(day >= 5),
        "is_rush_hour": int(hour in [7,8,9,16,17,18,19])
    }])


def surge_label(prob):
    if prob >= 0.75:
        return "VERY BUSY"
    elif prob >= 0.45:
        return "MODERATELY BUSY"
    return "NOT BUSY"

# -------------------------------------------------
# Request schema
# -------------------------------------------------
class RouteRequest(BaseModel):
    from_place: str
    to_place: str

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/predict")
def predict_surge(req: RouteRequest):

    from_geo = geocode_google(req.from_place)
    to_geo = geocode_google(req.to_place)

    if not from_geo or not to_geo:
        raise HTTPException(
            status_code=400,
            detail="Invalid location name"
        )

    from_zone = latlon_to_zone(*from_geo)
    to_zone = latlon_to_zone(*to_geo)

    features = build_features(req.from_place, req.to_place)
    surge_prob = model.predict_proba(features)[0][1]

    return {
        "from_zone": from_zone,
        "to_zone": to_zone,
        "surge_status": surge_label(surge_prob),
        "surge_score": round(float(surge_prob), 3)
    }
