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
    page_title="Ride Demand Status Checker",
    page_icon="🚕",
    layout="centered"
)

st.title("🚕 Ride Demand Status Checker")
st.caption("Check how busy a route is before booking a ride")

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
# OPENSTREETMAP GEOCODING (SAFE)
# =================================================
def geocode_place(place_name):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place_name,
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "RideDemandChecker/1.0 (academic-project)"
    }

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
# MAP COORDINATES → ZONE
# =================================================
def latlon_to_zone(lat, lon):
    df = zones.copy()
    df["dist"] = (df["lat"] - lat) ** 2 + (df["lon"] - lon) ** 2
    return int(df.sort_values("dist").iloc[0]["zone_id"])

# =================================================
# DEMAND SIMULATION (KEY LOGIC)
# =================================================
def estimate_demand(place_name):
    name = place_name.lower()

    high_demand = [
        "airport", "station", "downtown", "central",
        "mall", "market", "stadium", "tech park",
        "it park", "terminal", "business"
    ]

    medium_demand = [
        "city", "road", "circle", "square",
        "junction", "plaza"
    ]

    for word in high_demand:
        if word in name:
            return 140

    for word in medium_demand:
        if word in name:
            return 80

    return 30  # residential / calm area

# =================================================
# FEATURE GENERATION
# =================================================
def build_features(from_place, to_place):
    now = datetime.utcnow()
    hour = now.hour
    day = now.weekday()

    pickup_count = estimate_demand(from_place)
    od_trip_count = estimate_demand(to_place)

    avg_travel_time = 35 if pickup_count > 100 else 20
    avg_speed = 15 if pickup_count > 100 else 30

    return pd.DataFrame([{
        "pickup_count": pickup_count,
        "od_trip_count": od_trip_count,
        "avg_travel_time": avg_travel_time,
        "avg_speed": avg_speed,
        "hour": hour,
        "dayofweek": day,
        "is_weekend": int(day >= 5),
        "is_rush_hour": int(hour in [7, 8, 9, 16, 17, 18, 19])
    }])

# =================================================
# UI INPUTS
# =================================================
st.subheader("📍 Enter trip locations")

from_place = st.text_input(
    "From location",
    placeholder="e.g. JFK International Airport, New York"
)

to_place = st.text_input(
    "To location",
    placeholder="e.g. Times Square, Manhattan"
)

# =================================================
# PREDICTION
# =================================================
if st.button("🔍 Check Route Status"):

    if not from_place or not to_place:
        st.warning("Please enter both locations")
        st.stop()

    with st.spinner("Detecting locations..."):
        from_geo = geocode_place(from_place)
        to_geo = geocode_place(to_place)

    if from_geo is None:
        st.error(f"❌ Could not detect FROM location: {from_place}")
        st.stop()

    if to_geo is None:
        st.error(f"❌ Could not detect TO location: {to_place}")
        st.stop()

    from_zone = latlon_to_zone(from_geo["lat"], from_geo["lon"])
    to_zone = latlon_to_zone(to_geo["lat"], to_geo["lon"])

    features = build_features(from_place, to_place)
    surge_prob = model.predict_proba(features)[0][1]

    st.markdown("---")
    st.subheader("🚦 Route Demand Status")

    # =================================================
    # BUSY STATUS (USER FRIENDLY OUTPUT)
    # =================================================
    if surge_prob >= 0.75:
        st.error("🔥 VERY BUSY")
        st.write("High demand detected. Expect surge pricing and longer wait times.")
    elif surge_prob >= 0.45:
        st.warning("⚠️ MODERATELY BUSY")
        st.write("Demand is rising. Some delays may occur.")
    else:
        st.success("✅ NOT BUSY")
        st.write("Normal demand. Easy availability of rides.")

    st.caption(f"Route: Zone {from_zone} → Zone {to_zone}")

# =================================================
# FOOTER
# =================================================
st.markdown("---")
st.caption("OpenStreetMap + LightGBM | Academic Project Demo")
