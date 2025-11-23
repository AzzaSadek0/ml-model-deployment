from fastapi import FastAPI
from joblib import load
import pandas as pd

app = FastAPI()

# Load model
model = load("model.joblib")

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return {"prediction": float(pred)}
