from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
import joblib
import uvicorn

# Create an instance of FastAPI
app = FastAPI(
    title="Sepsis Prediction API",
    description="Predicts whether a patient at the ICU has the Sepsis disease or not",
    version="1.0.0",
    contact={
        "name": "API Support",
        "email": "parockson@gmail.com"
    }
)

# Load model components
def load_ml_components():
    model_components = {
        'LGBM': joblib.load('../exports/models/LGBM_best_model.pkl'),
        'RandomForest': joblib.load('../exports/models/RandomForest_best_model.pkl'),
        'SVM': joblib.load('../exports/models/SVM_best_model.pkl')
    }
    return model_components

# Call the model components
model_components = load_ml_components()

# Get API status
@app.get("/")
def get_status():
    return {"status": "API is running"}

# Create SepsisFeatures class from pydantic BaseModel
class SepsisFeatures(BaseModel):
    PRG: int
    PL: int
    PR: int
    SK: int
    TS: int
    BD2: float
    Age: int

# Create Endpoint for the LGBM Classifier
@app.post("/predict_sepsis/lgbm")
async def predict_sepsis_lgbm(data: SepsisFeatures):
    try:
        # Create dataframe from sepsis data
        df = pd.DataFrame([data.dict()])
        # Call the LGBM model
        lgbm_model = model_components["LGBM"]
        # Make prediction
        prediction = lgbm_model.predict(df)
        prediction = int(prediction[0])
        prediction_proba = lgbm_model.predict_proba(df)[0].tolist()
        response = {
            "model_used": "LGBM",
            "prediction": prediction,
            "prediction_probability": {
                "Negative": round(prediction_proba[0], 2),
                "Positive": round(prediction_proba[1], 2)
            }
        }
        return response
    except Exception as e:
        return {"error": str(e)}

# Create Endpoint for the Random Forest Model
@app.post("/predict_sepsis/random_forest")
async def predict_sepsis_random_forest(data: SepsisFeatures):
    try:
        # Create dataframe from sepsis data
        df = pd.DataFrame([data.dict()])
        # Call the Random Forest model
        rf_model = model_components["RandomForest"]
        # Make prediction
        prediction = rf_model.predict(df)
        prediction = int(prediction[0])
        prediction_proba = rf_model.predict_proba(df)[0].tolist()
        response = {
            "model_used": "Random Forest",
            "prediction": prediction,
            "prediction_probability": {
                "Negative": round(prediction_proba[0], 2),
                "Positive": round(prediction_proba[1], 2)
            }
        }
        return response
    except Exception as e:
        return {"error": str(e)}

# Create Endpoint for the SVM Model
@app.post("/predict_sepsis/svm")
async def predict_sepsis_svm(data: SepsisFeatures):
    try:
        # Create dataframe from sepsis data
        df = pd.DataFrame([data.dict()])
        # Call the SVM model
        svm_model = model_components["SVM"]
        # Make prediction
        prediction = svm_model.predict(df)
        prediction = int(prediction[0])
        prediction_proba = svm_model.predict_proba(df)[0].tolist() if hasattr(svm_model, "predict_proba") else [1 - prediction, prediction]
        response = {
            "model_used": "SVM",
            "prediction": prediction,
            "prediction_probability": {
                "Negative": round(prediction_proba[0], 2),
                "Positive": round(prediction_proba[1], 2)
            }
        }
        return response
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
