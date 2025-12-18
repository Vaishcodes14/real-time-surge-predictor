import streamlit as st
import pandas as pd
import joblib
import requests
import os
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

st.write("API KEY FOUND:", GOOGLE_API_KEY is not None)


# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="Ride Surge & ETA Predictor",
    page_icon="🚕",
    layout="centered"
)

st.title("🚕 Ride Demand Surge & ETA Predictor")
st.caption("NYC-based | Google Geocoding | Peak vs Off-Peak")

# =================================================
# LOAD MODEL
# =================================================
@st.cache_resource
def load_model():
    return joblib.load("lightgbm_surge_model.joblib")

model = load_model()

# =================================================
# GOOGLE API KEY
# =================================================
GOOGLE_API_KEY = os.getenv("AIzaSyCdLCL3NZhOnEtR-n87ia13tJvjABAOpGI")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not set in environment variables")
    st.stop()

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
    name = place.lower()
    if any(w in name for w in ["airport", "station", "downtown", "terminal"]):
        return 140
    if any(w in name for w in ["square", "mall", "avenue", "road"]):
        return 80
    return 30

# =================================================
# FEATURE BUILDING
# =================================================
def build_features(from_place, to_place, is_peak):
    now = datetime.utcnow()
    hour = now.hour
    day = now.weekday()

    pickup = estimate_demand(from_place)
    drop = estimate_demand(to_place)

    avg_speed = 30 if pickup < 100 else 18
    if is_peak:
        avg_speed -= 6

    avg_speed = max(avg_speed, 12)

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
if st.button("🔍 Predict Surge"):

    if not from_place or not to_place:
        st.warning("Please enter both locations")
        st.stop()

    with st.spinner("Resolving locations using Google API..."):
        from_geo = geocode_place(from_place)
        to_geo = geocode_place(to_place)

    if not from_geo or not to_geo:
        st.error("❌ Invalid location name. Please be specific.")
        st.stop()

    distance_km = haversine(
        from_geo[0], from_geo[1],
        to_geo[0], to_geo[1]
    )

    features, avg_speed = build_features(from_place, to_place, is_peak)
    surge_prob = model.predict_proba(features)[0][1]

    travel_time_min = (distance_km / avg_speed) * 60

    st.markdown("---")
    st.subheader("🚦 Route Result")

    if surge_prob >= 0.75:
        st.error("🔥 VERY BUSY")
        label = "High demand and congestion"
    elif surge_prob >= 0.45:
        st.warning("⚠️ MODERATELY BUSY")
        label = "Moderate demand"
    else:
        st.success("✅ NOT BUSY")
        label = "Normal traffic"

    st.write(label)

    st.markdown("### ⏱️ Estimated Travel Time")
    st.write(f"🛣️ Distance: **{distance_km:.1f} km**")
    st.write(f"🚗 Avg Speed: **{avg_speed} km/h**")
    st.write(f"⏰ ETA: **{travel_time_min:.0f} minutes**")

# =================================================
# FOOTER
# =================================================
st.markdown("---")
st.caption(
    "Geocoding via Google API. "
    "Surge model trained on NYC taxi data. ETA is estimated."
)
