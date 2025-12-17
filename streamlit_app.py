import streamlit as st
import pandas as pd
import joblib
import requests
import os
from datetime import datetime

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Route Demand Surge Predictor", page_icon="🚕")

st.title("🚕 Route Demand Surge Predictor")
st.write("Predict demand surge for trips from one area to another")

# -----------------------------
# Load model & data
# -----------------------------
model = joblib.load("lightgbm_surge_model.joblib")
zones = pd.read_csv("zone_centroids.csv")

# -----------------------------
# Helper functions
# -----------------------------
def geocode_place(place):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Google API key missing")
        st.stop()

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": place, "key": api_key}
    res = requests.get(url, params=params).json()

    if res["status"] != "OK":
        st.error(f"Invalid place: {place}")
        st.stop()

    loc = res["results"][0]["geometry"]["location"]
    return loc["lat"], loc["lng"]

def latlon_to_zone(lat, lon):
    zones["dist"] = (zones["lat"] - lat) ** 2 + (zones["lon"] - lon) ** 2
    return int(zones.sort_values("dist").iloc[0]["zone_id"])

# -----------------------------
# UI Inputs
# -----------------------------
from_area = st.text_input("From area", placeholder="e.g. Times Square New York")
to_area = st.text_input("To area", placeholder="e.g. JFK Airport")

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict Surge"):
    if not from_area or not to_area:
        st.warning("Please enter both locations")
        st.stop()

    with st.spinner("Predicting surge..."):
        from_lat, from_lon = geocode_place(from_area)
        to_lat, to_lon = geocode_place(to_area)

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

        st.success(f"🚦 Surge Probability: **{round(float(prob), 3)}**")
        st.caption(f"From zone {from_zone} → To zone {to_zone}")
