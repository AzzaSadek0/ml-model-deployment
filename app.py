from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def root():
    return {"message": "ML Model API is running successfully!"}

@app.post("/predict")
def predict(input_data: dict):
    # Example input: { "TV": 200, "Radio": 30, "Newspaper": 45 }
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return {"prediction": float(prediction)}
