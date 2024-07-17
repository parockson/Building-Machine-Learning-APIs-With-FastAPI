from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

class PatientFeatures(BaseModel):
    # ID: str
    PRG: int
    PL: int
    PR: int
    SK: int
    TS: int
    # M11: float
    BD2: float
    Age: int
    # Insurance: int
    # Sepsis: str

@app.get('/')
def status_check():
    return {"Status": "App is Online..."}

# @app.get('/documents')
# def documentation():
#     return {"All Documentation": "API Documentation"}

XGB_pipeline = joblib.load('../exports/XGB.joblib')

@app.post('/xgb_prediction')
def predict_sepsis_status(data: PatientFeatures):
    # Assuming XGB_pipeline has a predict method and expects a DataFrame or similar input format
    df = pd.DataFrame([data.model_dump()])
    prediction = XGB_pipeline.predict(df)
    prediction = int(prediction[0])
    return {'prediction': prediction}
