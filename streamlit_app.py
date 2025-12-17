import streamlit as st
import pandas as pd
import joblib
import requests
import time
from datetime import datetime

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="Ride Demand Surge Predictor",
    page_icon="🚕",
    layout="centered"
)

st.title("🚕 Ride Demand Surge Predictor")
st.caption("Predict surge probability between two locations")

# =================================================
# LOAD MODEL & DATA (CACHED)
# =================================================
@st.cache_resource
def load_model():
    return joblib.load("lightgbm_surge_model.joblib")

@st.cache_data
def load_zones():
    return pd.read_csv("zone_centroids.csv")

model = load_model()
zones = load_zones()

# =================================================
# GEOCODING (OPENSTREETMAP – SAFE VERSION)
# =================================================
def geocode_place(place_name):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place_name,
        "format": "json",
        "addressdetails": 1,
        "limit": 1
    }
    headers = {
        "User-Agent": "RideSurgePredictor/1.0 (academic-project)"
    }

    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()

            if len(data) == 0:
                return None

            return {
                "lat": float(data[0]["lat"]),
                "lon": float(data[0]["lon"]),
                "display": data[0].get("display_name", "")
            }

        except requests.exceptions.RequestException:
            time.sleep(1)

    return None

# =================================================
# MAP COORDINATES → ZONE
# =================================================
def latlon_to_zone(lat, lon):
    df = zones.copy()
    df["dist"] = (df["lat"] - lat) ** 2 + (df["lon"] - lon) ** 2
    return int(df.sort_values("dist").iloc[0]["zone_id"])

# =================================================
# BUILD FEATURE VECTOR
# =================================================
def build_features():
    now = datetime.utcnow()
    hour = now.hour
    day = now.weekday()

    return pd.DataFrame([{
        "pickup_count": 0,
        "od_trip_count": 0,
        "avg_travel_time": 0,
        "avg_speed": 0,
        "hour": hour,
        "dayofweek": day,
        "is_weekend": int(day >= 5),
        "is_rush_hour": int(hour in [7, 8, 9, 16, 17, 18, 19])
    }])

# =================================================
# UI INPUTS
# =================================================
st.subheader("📍 Enter locations")

from_place = st.text_input(
    "From location",
    placeholder="e.g. JFK International Airport, New York, USA"
)

to_place = st.text_input(
    "To location",
    placeholder="e.g. Times Square, Manhattan, New York, USA"
)

# =================================================
# PREDICTION LOGIC
# =================================================
if st.button("🔮 Predict Surge"):

    if not from_place or not to_place:
        st.warning("Please enter both locations")
        st.stop()

    with st.spinner("Resolving locations using OpenStreetMap..."):
        from_geo = geocode_place(from_place)
        to_geo = geocode_place(to_place)

    if from_geo is None:
        st.error(f"❌ Could not detect FROM location:\n\n**{from_place}**")
        st.stop()

    if to_geo is None:
        st.error(f"❌ Could not detect TO location:\n\n**{to_place}**")
        st.stop()

    st.success("📍 Locations detected successfully")

    st.write("**From:**", from_geo["display"])
    st.write("**To:**", to_geo["display"])

    from_zone = latlon_to_zone(from_geo["lat"], from_geo["lon"])
    to_zone = latlon_to_zone(to_geo["lat"], to_geo["lon"])

    features = build_features()
    surge_prob = model.predict_proba(features)[0][1]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    st.metric(
        label="Surge Probability",
        value=round(float(surge_prob), 3)
    )

    st.caption(f"Route: Zone {from_zone} → Zone {to_zone}")

# =================================================
# FOOTER
# =================================================
st.markdown("---")
st.caption("OpenStreetMap (Nominatim) + LightGBM | Academic ML Project")
