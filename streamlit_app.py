import streamlit as st
import requests
import time
import joblib
import pandas as pd
from datetime import datetime

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Route Demand Surge Predictor",
    page_icon="🚕",
    layout="centered"
)

# -------------------------------------------------
# Load model & zone data
# -------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("lightgbm_surge_model.joblib")

@st.cache_data
def load_zones():
    return pd.read_csv("zone_centroids.csv")

model = load_model()
zones = load_zones()

# -------------------------------------------------
# Helper: Geocode using OpenStreetMap (Nominatim)
# -------------------------------------------------
def geocode_place(place_name):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place_name,
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "DemandSurgePredictor/1.0 (student-project)"
    }

    for _ in range(3):  # retry 3 times
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            if not data:
                return None, None

            return float(data[0]["lat"]), float(data[0]["lon"])

        except requests.exceptions.RequestException:
            time.sleep(1)

    return None, None

# -------------------------------------------------
# Helper: Map lat/lon → nearest zone
# -------------------------------------------------
def latlon_to_zone(lat, lon):
    zones_copy = zones.copy()
    zones_copy["dist"] = (zones_copy["lat"] - lat) ** 2 + (zones_copy["lon"] - lon) ** 2
    return int(zones_copy.sort_values("dist").iloc[0]["zone_id"])

# -------------------------------------------------
# Helper: Build feature vector
# -------------------------------------------------
def build_features():
    now = datetime.utcnow()
    hour = now.hour
    dayofweek = now.weekday()

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

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🚕 Route Demand Surge Predictor")
st.caption("Predict demand surge between two locations using ML")

st.markdown("### 📍 Enter trip locations")

from_area = st.text_input(
    "From location",
    placeholder="e.g. JFK International Airport, New York"
)

to_area = st.text_input(
    "To location",
    placeholder="e.g. Times Square, Manhattan"
)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("🔮 Predict Surge"):
    if not from_area or not to_area:
        st.error("❌ Please enter both locations")
        st.stop()

    with st.spinner("Detecting locations..."):
        from_lat, from_lon = geocode_place(from_area)
        to_lat, to_lon = geocode_place(to_area)

    if from_lat is None or to_lat is None:
        st.error("❌ Unable to detect one or both locations. Try a more specific name.")
        st.stop()

    from_zone = latlon_to_zone(from_lat, from_lon)
    to_zone = latlon_to_zone(to_lat, to_lon)

    features = build_features()
    surge_prob = model.predict_proba(features)[0][1]

    st.success(f"📊 **Surge Probability: {round(float(surge_prob), 3)}**")
    st.info(f"📍 Route: Zone {from_zone} → Zone {to_zone}")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Powered by OpenStreetMap + LightGBM | Academic Project Demo")
