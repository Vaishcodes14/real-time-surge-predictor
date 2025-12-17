import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import datetime

# -------------------------------------------------
# PAGE CONFIG (must be first Streamlit command)
# -------------------------------------------------
st.set_page_config(
    page_title="Route Demand Surge Predictor",
    page_icon="🚕",
    layout="centered"
)

st.write("🚀 App is starting...")
st.title("🚕 Route Demand Surge Predictor")
st.caption("Predict demand surge between two locations using ML")

# -------------------------------------------------
# CACHED LOADERS (CRITICAL FIX)
# -------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("lightgbm_surge_model.joblib")

@st.cache_data
def load_zones():
    return pd.read_csv("zone_centroids.csv")

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def geocode_place(place_name: str):
    """Convert place name to lat/lon using OpenStreetMap (Nominatim)"""
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place_name,
        "format": "json",
        "limit": 1
    }
    headers = {"User-Agent": "surge-predictor-app"}
    response = requests.get(url, params=params, headers=headers, timeout=10)
    data = response.json()

    if not data:
        st.error(f"❌ Location not found: {place_name}")
        st.stop()

    return float(data[0]["lat"]), float(data[0]["lon"])


def latlon_to_zone(lat, lon, zones_df):
    zones_df = zones_df.copy()
    zones_df["dist"] = (zones_df["lat"] - lat) ** 2 + (zones_df["lon"] - lon) ** 2
    return int(zones_df.sort_values("dist").iloc[0]["zone_id"])


def build_features(ts):
    hour = ts.hour
    day = ts.weekday()

    return pd.DataFrame([{
        "pickup_count": 0,
        "od_trip_count": 0,
        "avg_travel_time": 0,
        "avg_speed": 0,
        "hour": hour,
        "dayofweek": day,
        "is_weekend": 1 if day >= 5 else 0,
        "is_rush_hour": 1 if hour in [7, 8, 9, 16, 17, 18, 19] else 0
    }])

# -------------------------------------------------
# UI INPUTS
# -------------------------------------------------
st.subheader("📍 Enter trip locations")

from_area = st.text_input(
    "From location",
    placeholder="e.g. JFK Airport New York"
)

to_area = st.text_input(
    "To location",
    placeholder="e.g. Times Square Manhattan"
)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if st.button("🔮 Predict Surge"):

    if not from_area or not to_area:
        st.warning("Please enter both From and To locations")
        st.stop()

    with st.spinner("Loading model and data..."):
        model = load_model()
        zones = load_zones()

    with st.spinner("Resolving locations..."):
        from_lat, from_lon = geocode_place(from_area)
        to_lat, to_lon = geocode_place(to_area)

        from_zone = latlon_to_zone(from_lat, from_lon, zones)
        to_zone = latlon_to_zone(to_lat, to_lon, zones)

    with st.spinner("Predicting demand surge..."):
        features = build_features(datetime.utcnow())
        surge_prob = model.predict_proba(features)[0][1]

    st.success(f"🚦 Surge Probability: **{round(float(surge_prob), 3)}**")
    st.caption(f"From Zone {from_zone} → To Zone {to_zone}")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()
st.caption(
    "Powered by OpenStreetMap + LightGBM | Academic Project Demo"
)
