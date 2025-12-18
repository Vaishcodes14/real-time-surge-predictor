import streamlit as st
import pandas as pd
import joblib
import requests
import time
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="Ride Demand & Travel Time Estimator",
    page_icon="🚕",
    layout="centered"
)

st.title("🚕 Ride Demand & Travel Time Estimator")
st.caption("NYC-based surge & travel time prediction")

# =================================================
# LOAD MODEL & ZONE DATA (CACHED)
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
# GEOCODING (OPENSTREETMAP)
# =================================================
def geocode_place(place_name):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place_name,
        "format": "json",
        "limit": 1
    }
    headers = {"User-Agent": "RideDemandEstimator/1.0 (academic-project)"}

    for _ in range(3):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()
            if not data:
                return None
            return {
                "lat": float(data[0]["lat"]),
                "lon": float(data[0]["lon"]),
                "display": data[0]["display_name"]
            }
        except requests.exceptions.RequestException:
            time.sleep(1)

    return None

# =================================================
# DISTANCE (HAVERSINE)
# =================================================
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c  # km

# =================================================
# DEMAND ESTIMATION (NYC STYLE)
# =================================================
def estimate_demand(place):
    name = place.lower()
    if any(w in name for w in ["airport", "station", "downtown", "central", "mall", "terminal"]):
        return 140
    if any(w in name for w in ["square", "avenue", "road", "park"]):
        return 80
    return 30

# =================================================
# FEATURE GENERATION (PEAK AWARE)
# =================================================
def build_features(from_place, to_place, is_peak):
    now = datetime.utcnow()
    hour = now.hour
    day = now.weekday()

    pickup = estimate_demand(from_place)
    drop = estimate_demand(to_place)

    # Speed logic
    if is_peak:
        avg_speed = 12 if pickup > 100 else 18
    else:
        avg_speed = 25 if pickup > 100 else 35

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
# UI INPUTS
# =================================================
st.subheader("📍 Enter trip locations (NYC)")

from_place = st.text_input(
    "From location",
    placeholder="e.g. JFK International Airport, New York"
)

to_place = st.text_input(
    "To location",
    placeholder="e.g. Times Square, Manhattan"
)

# PEAK / OFF-PEAK TOGGLE
time_mode = st.radio(
    "⏰ Select travel time",
    ["Peak Hours (8–10 AM / 5–7 PM)", "Off-Peak Hours"],
    horizontal=True
)

is_peak = time_mode.startswith("Peak")

# =================================================
# PREDICTION
# =================================================
if st.button("🔍 Analyze Route"):

    if not from_place or not to_place:
        st.warning("Please enter both locations")
        st.stop()

    with st.spinner("Resolving locations..."):
        from_geo = geocode_place(from_place)
        to_geo = geocode_place(to_place)

    if from_geo is None or to_geo is None:
        st.error("❌ Unable to detect one or both locations.")
        st.stop()

    # Distance
    distance_km = haversine(
        from_geo["lat"], from_geo["lon"],
        to_geo["lat"], to_geo["lon"]
    )

    features, avg_speed = build_features(from_place, to_place, is_peak)
    surge_prob = model.predict_proba(features)[0][1]

    travel_time_min = (distance_km / avg_speed) * 60

    st.markdown("---")
    st.subheader("🚦 Route Analysis")

    # BUSY STATUS
    if surge_prob >= 0.75:
        st.error("🔥 VERY BUSY")
        status = "Heavy demand and congestion expected"
    elif surge_prob >= 0.45:
        st.warning("⚠️ MODERATELY BUSY")
        status = "Moderate demand, possible delays"
    else:
        st.success("✅ NOT BUSY")
        status = "Smooth traffic conditions"

    st.write(status)

    st.markdown("### ⏱️ Estimated Time to Reach")
    st.write(f"🛣️ Distance: **{distance_km:.1f} km**")
    st.write(f"🚗 Average speed: **{avg_speed} km/h**")
    st.write(f"⏰ Estimated time: **{travel_time_min:.0f} minutes**")

    st.caption(f"Mode selected: {'Peak Hours' if is_peak else 'Off-Peak Hours'}")

# =================================================
# FOOTER
# =================================================
st.markdown("---")
st.caption(
    "Peak hours simulate rush-hour congestion. "
    "Model trained on NYC taxi data. Travel time is estimated."
)
