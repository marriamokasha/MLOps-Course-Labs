# main.py
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
import joblib
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(title="Churn Prediction API")

# Load model and preprocessor
try:
    # Load preprocessor (this is correct)
    preprocessor = joblib.load("models/preprocessor.pkl")
    
    # Load model - it's saved as a dictionary with 'model_object' key
    model_info = joblib.load("models/best_model.pkl")
    model = model_info["model_object"]  # Extract the actual model
    model_name = model_info["model_name"]  # Get the model name
    
    logger.info(f"Successfully loaded {model_name} model and preprocessor")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    model = None
    preprocessor = None
    model_name = None

# Define input schema - MUST match the exact columns used in training
class CustomerData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.get("/")
def home():
    logger.info("Homepage accessed.")
    return {
        "message": "Welcome to the Churn Prediction API",
        "model_loaded": model is not None,
        "model_name": model_name if model_name else "None"
    }

@app.get("/health")
def health():
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    logger.info("Health check OK.")
    return {
        "status": "OK",
        "model_name": model_name,
        "model_type": type(model).__name__
    }

@app.post("/predict")
def predict(data: CustomerData):
    
    # Convert input to DataFrame with EXACT same column order as training
    # This matches the filter_feat order in your preprocess function
    input_data = {
        "CreditScore": data.CreditScore,
        "Geography": data.Geography,
        "Gender": data.Gender,
        "Age": data.Age,
        "Tenure": data.Tenure,
        "Balance": data.Balance,
        "NumOfProducts": data.NumOfProducts,
        "HasCrCard": data.HasCrCard,
        "IsActiveMember": data.IsActiveMember,
        "EstimatedSalary": data.EstimatedSalary
    }
    
    df = pd.DataFrame([input_data])
    logger.info(f"Input DataFrame shape: {df.shape}, columns: {list(df.columns)}")
    
    # Transform using the same preprocessor from training
    X_transformed = preprocessor.transform(df)
    logger.info(f"Transformed data shape: {X_transformed.shape}")
    
    # Make prediction
    prediction = model.predict(X_transformed)[0]
    
    # Get prediction probability if available
    prediction_proba = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_transformed)[0]
        prediction_proba = {
            "no_churn_probability": float(proba[0]),
            "churn_probability": float(proba[1])
        }
    
    logger.info(f"Prediction: {prediction}")
    
    response = {
        "churn_prediction": int(prediction),
        "prediction_label": "Will Churn" if prediction == 1 else "Will Not Churn",
    }
    
    if prediction_proba:
        response["confidence"] = float(max(proba))
    
    return response 
