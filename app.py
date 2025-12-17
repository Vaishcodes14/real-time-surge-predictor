from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

# -----------------------------
# App initialization
# -----------------------------
app = FastAPI(title="Route Demand Surge Predictor")

# -----------------------------
# Load model & zone data
# -----------------------------
model = joblib.load("lightgbm_surge_model.joblib")
zones = pd.read_csv("zone_centroids.csv")

# Normalize names for matching
zones["zone_name"] = zones["zone_name"].str.lower().str.strip()

# -----------------------------
# Helper: area name -> zone_id
# -----------------------------
def area_to_zone(area_name: str):
    area_name = area_name.lower().strip()
    match = zones[zones["zone_name"] == area_name]

    if match.empty:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown area name: {area_name}"
        )

    return int(match.iloc[0]["zone_id"])

# -----------------------------
# Request schema
# -----------------------------
class RouteRequest(BaseModel):
    from_area: str
    to_area: str

# -----------------------------
# Home (frontend served from backend)
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Route Demand Surge Predictor</title>
        <style>
            body { font-family: Arial; padding: 40px; }
            input { padding: 8px; width: 250px; }
            button { padding: 10px 20px; margin-top: 10px; }
        </style>
    </head>
    <body>

        <h2>🚕 Route Demand Surge Predictor</h2>
        <p>Predict demand surge for trips from A to B</p>

        <input id="from" placeholder="From area" /><br><br>
        <input id="to" placeholder="To area" /><br><br>

        <button onclick="predict()">Predict Surge</button>

        <h3 id="result"></h3>

        <script>
            function predict() {
                const fromArea = document.getElementById("from").value;
                const toArea = document.getElementById("to").value;

                fetch("/predict_surge", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        from_area: fromArea,
                        to_area: toArea
                    })
                })
                .then(res => res.json())
                .then(data => {
                    document.getElementById("result").innerText =
                        "Surge Probability: " + data.surge_probability;
                })
                .catch(err => {
                    document.getElementById("result").innerText =
                        "Error predicting surge";
                });
            }
        </script>

    </body>
    </html>
    """

# -----------------------------
# Health check
# -----------------------------
@app.get("/test")
def test():
    return {"status": "ok"}

# -----------------------------
# Prediction endpoint (A -> B)
# -----------------------------
@app.post("/predict_surge")
def predict_surge(req: RouteRequest):

    from_zone = area_to_zone(req.from_area)
    to_zone = area_to_zone(req.to_area)

    now = datetime.now()
    hour = now.hour
    dayofweek = now.weekday()
    is_weekend = 1 if dayofweek >= 5 else 0
    is_rush_hour = 1 if hour in [7, 8, 9, 16, 17, 18, 19] else 0

    # Feature vector (must match training)
    features = pd.DataFrame([{
        "pickup_count": 0,
        "od_trip_count": 0,
        "avg_travel_time": 0,
        "avg_speed": 0,
        "hour": hour,
        "dayofweek": dayofweek,
        "is_weekend": is_weekend,
        "is_rush_hour": is_rush_hour
    }])

    surge_prob = model.predict_proba(features)[0][1]

    return {
        "from_zone": from_zone,
        "to_zone": to_zone,
        "surge_probability": round(float(surge_prob), 3)
    }
