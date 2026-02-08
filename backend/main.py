from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Stroke Risk Prediction API")

# ---------- CORS (SUPER IMPORTANT FOR VERCEL) ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # allow Vercel + everyone
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- HANDLE OPTIONS (Preflight fix for 401) ----------
@app.options("/predict")
async def predict_options():
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )

# ---------- LOAD MODEL ----------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "stroke_xgb.pkl")

model = joblib.load(MODEL_PATH)

# ---------- SAME ENCODING AS TRAINING ----------
gender_map = {"Male": 1, "Female": 0}
yesno_map = {"Yes": 1, "No": 0}
work_map = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3}
res_map = {"Urban": 1, "Rural": 0}
smoke_map = {
    "never smoked": 0,
    "formerly smoked": 1,
    "smokes": 2,
    "Unknown": 3
}

class StrokeInput(BaseModel):
    age: int
    gender: str
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

@app.get("/")
def home():
    return {"message": "Stroke API running 🚀"}

@app.post("/predict")
def predict(data: StrokeInput):

    encoded = {
        "gender": gender_map.get(data.gender, 1),
        "age": data.age,
        "hypertension": data.hypertension,
        "heart_disease": data.heart_disease,
        "ever_married": yesno_map.get(data.ever_married, 1),
        "work_type": work_map.get(data.work_type, 0),
        "Residence_type": res_map.get(data.Residence_type, 1),
        "avg_glucose_level": data.avg_glucose_level,
        "bmi": data.bmi,
        "smoking_status": smoke_map.get(data.smoking_status, 0)
    }

    columns = [
        "gender","age","hypertension","heart_disease",
        "ever_married","work_type","Residence_type",
        "avg_glucose_level","bmi","smoking_status"
    ]

    df = pd.DataFrame([encoded], columns=columns)

    try:
        pred = int(model.predict(df.values)[0])
        return {
            "stroke_risk": pred,
            "risk_level": "High" if pred == 1 else "Low"
        }
    except Exception as e:
        return {"error": str(e)}
