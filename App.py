from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(data: dict):
    # Expected input: TV, Radio, Newspaper
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"prediction": float(prediction)}
