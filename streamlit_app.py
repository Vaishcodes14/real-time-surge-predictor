import streamlit as st
import pandas as pd
import joblib
import requests
import os
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

# =================================================
# LOAD API KEYS (MUST BE HERE, RIGHT AFTER IMPORTS)
# =================================================
GOOGLE_API_KEY = os.getenv("AIzaSyA-UGewPptEcN_i3dLalNe7kpkr93FlUH0")
OPENWEATHER_API_KEY = os.getenv("fc66323ad12fd29d89668cd000db815c")

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="Ride Demand Surge Predictor",
    page_icon="🚕",
    layout="centered"
)

st.title("🚕 Ride Demand Surge Predictor")
st.caption("Google Maps + OpenWeather + ML (NYC Taxi Data)")

# =================================================
# LOAD ML MODEL
# =================================================
@st.cache_resource
def load_model():
    return joblib.load("lightgbm_surge_model.joblib")

model = load_model()

# =================================================
# GOOGLE GEOCODING
# =================================================
def geocode_place(place):
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

# =================================================
# WEATHER (OPENWEATHER)
# =================================================
def get_weather(lat, lon):
    if not OPENWEATHER_API_KEY:
        return "Unknown"

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    r = requests.get(url, params=params, timeout=10)
    data = r.json()
    return data["weather"][0]["main"]

# =================================================
# DISTANCE (HAVERSINE)
# =================================================
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return 6371 * 2 * asin(sqrt(a))  # km

# =================================================
# DEMAND HEURISTIC
# =================================================
def estimate_demand(place):
    p = place.lower()
    if "airport" in p:
        return 150
    if "station" in p or "terminal" in p:
        return 120
    if "downtown" in p or "square" in p:
        return 90
    return 40

# =================================================
# UI INPUT
# =================================================
st.subheader("📍 Enter trip locations")

from_place = st.text_input(
    "From location",
    placeholder="JFK International Airport, New York"
)

to_place = st.text_input(
    "To location",
    placeholder="Times Square, Manhattan, New York"
)

time_type = st.radio(
    "⏰ Time of travel",
    ["Peak Hours", "Off-Peak Hours"],
    horizontal=True
)

is_peak = time_type == "Peak Hours"

# =================================================
# PREDICTION
# =================================================
if st.button("🔍 Predict Surge"):

    if not from_place or not to_place:
        st.warning("Please enter both locations")

    else:
        from_geo = geocode_place(from_place)
        to_geo = geocode_place(to_place)

        if not from_geo or not to_geo:
            st.error("❌ Could not detect one or both locations")
        else:
            weather = get_weather(from_geo[0], from_geo[1])

            distance_km = haversine(
                from_geo[0], from_geo[1],
                to_geo[0], to_geo[1]
            )

            pickup = estimate_demand(from_place)
            drop = estimate_demand(to_place)

            now = datetime.utcnow()

            avg_speed = 28
            if is_peak:
                avg_speed -= 8
            if weather in ["Rain", "Thunderstorm"]:
                avg_speed -= 5

            avg_speed = max(avg_speed, 12)

            features = pd.DataFrame([{
                "pickup_count": pickup,
                "od_trip_count": drop,
                "avg_travel_time": 0,
                "avg_speed": avg_speed,
                "hour": now.hour,
                "dayofweek": now.weekday(),
                "is_weekend": int(now.weekday() >= 5),
                "is_rush_hour": int(is_peak)
            }])

            surge_prob = model.predict_proba(features)[0][1]
            eta_min = (distance_km / avg_speed) * 60

            st.markdown("---")
            st.subheader("🚦 Result")

            if surge_prob > 0.7:
                st.error("🔥 HIGH DEMAND (SURGE LIKELY)")
            elif surge_prob > 0.4:
                st.warning("⚠️ MODERATE DEMAND")
            else:
                st.success("✅ LOW DEMAND")

            st.write(f"🌦️ Weather: **{weather}**")
            st.write(f"🛣️ Distance: **{distance_km:.1f} km**")
            st.write(f"🚗 Avg Speed: **{avg_speed} km/h**")
            st.write(f"⏱️ ETA: **{eta_min:.0f} minutes**")

# =================================================
# FOOTER
# =================================================
st.markdown("---")
st.caption(
    "Google Maps for geocoding | OpenWeather for weather | "
    "LightGBM model trained on NYC taxi data"
)
