import streamlit as st
import pandas as pd
import joblib
import requests
import os
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="Ride Surge, ETA & Weather Predictor",
    page_icon="🚕",
    layout="centered"
)

st.title("🚕 Ride Surge, ETA & Weather Predictor")
st.caption("NYC-based | Google Geocoding | Weather-aware")

# =================================================
# LOAD API KEYS (NO CHECKS)
# =================================================
GOOGLE_API_KEY = os.getenv("AIzaSyCdLCL3NZhOnEtR-n87ia13tJvjABAOpGI")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# =================================================
# LOAD MODEL
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
        "key": AIzaSyCdLCL3NZhOnEtR-n87ia13tJvjABAOpGI
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("status") != "OK":
            return None

        loc = data["results"][0]["geometry"]["location"]
        return loc["lat"], loc["lng"]
    except:
        return None

# =================================================
# WEATHER (OPENWEATHERMAP)
# =================================================
def get_weather(lat, lon):
    if not fc66323ad12fd29d89668cd000db815c:
        return "Unknown"

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": fc66323ad12fd29d89668cd000db815c,
        "units": "metric"
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        return data["weather"][0]["main"]
    except:
        return "Unknown"

# =================================================
# DISTANCE (HAVERSINE)
# =================================================
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))  # km

# =================================================
# DEMAND ESTIMATION
# =================================================
def estimate_demand(place):
    p = place.lower()
    if any(w in p for w in ["airport", "station", "terminal", "downtown"]):
        return 140
    if any(w in p for w in ["square", "avenue", "road", "mall"]):
        return 80
    return 30

# =================================================
# FEATURE GENERATION (TIME + WEATHER)
# =================================================
def build_features(from_place, to_place, is_peak, weather):
    now = datetime.utcnow()
    hour = now.hour
    day = now.weekday()

    pickup = estimate_demand(from_place)
    drop = estimate_demand(to_place)

    avg_speed = 30 if pickup < 100 else 18

    if is_peak:
        avg_speed -= 6

    if weather in ["Rain", "Thunderstorm"]:
        avg_speed -= 5

    avg_speed = max(avg_speed, 10)

    features = pd.DataFrame([{
        "pickup_count": pickup,
        "od_trip_count": drop,
        "avg_travel_time": 0,
        "avg_speed": avg_speed,
        "hour": hour,
        "dayofweek": day,
        "is_weekend": int(day >= 5),
        "is_rush_hour": int(is_peak)
    }])

    return features, avg_speed

# =================================================
# UI
# =================================================
st.subheader("📍 Enter trip locations (New York City)")

from_place = st.text_input(
    "From location",
    placeholder="e.g. JFK International Airport, New York"
)

to_place = st.text_input(
    "To location",
    placeholder="e.g. Times Square, Manhattan"
)

time_mode = st.radio(
    "⏰ Time of travel",
    ["Peak Hours", "Off-Peak Hours"],
    horizontal=True
)

is_peak = time_mode == "Peak Hours"

# =================================================
# PREDICT
# =================================================
if st.button("🔍 Analyze Route"):

    if not from_place or not to_place:
        st.warning("Please enter both locations")

    else:
        with st.spinner("Resolving locations..."):
            from_geo = geocode_place(from_place)
            to_geo = geocode_place(to_place)

        if not from_geo or not to_geo:
            st.error("❌ Could not detect one or both locations.")
        else:
            weather = get_weather(from_geo[0], from_geo[1])

            distance_km = haversine(
                from_geo[0], from_geo[1],
                to_geo[0], to_geo[1]
            )

            features, avg_speed = build_features(
                from_place, to_place, is_peak, weather
            )

            surge_prob = model.predict_proba(features)[0][1]
            travel_time_min = (distance_km / avg_speed) * 60

            st.markdown("---")
            st.subheader("🚦 Route Result")

            if surge_prob >= 0.75:
                st.error("🔥 VERY BUSY")
            elif surge_prob >= 0.45:
                st.warning("⚠️ MODERATELY BUSY")
            else:
                st.success("✅ NOT BUSY")

            st.markdown("### 🌦️ Weather")
            st.write(weather)

            st.markdown("### ⏱️ Estimated Travel Time")
            st.write(f"🛣️ Distance: **{distance_km:.1f} km**")
            st.write(f"🚗 Avg Speed: **{avg_speed} km/h**")
            st.write(f"⏰ ETA: **{travel_time_min:.0f} minutes**")

# =================================================
# FOOTER
# =================================================
st.markdown("---")
st.caption(
    "Google Geocoding for location, OpenWeatherMap for weather. "
    "Surge model trained on NYC taxi data. ETA is estimated."
)
