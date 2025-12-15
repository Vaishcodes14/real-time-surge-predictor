from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

app = FastAPI(title="Real-Time Demand Surge Predictor")

@app.get("/")
def home():
    return {"status": "API running"}

class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    timestamp: datetime

@app.post("/predict_surge")
def predict_surge(data: LocationRequest):
    return {"message": "prediction endpoint works"}
