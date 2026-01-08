import os
import joblib
import numpy as np

# Get absolute path to app directory
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "diabetes",
    "model.pkl"
)

model = joblib.load(MODEL_PATH)

def predict_diabetes(data):
    features = np.array([[
        data.pregnancies,
        data.glucose,
        data.blood_pressure,
        data.skin_thickness,
        data.insulin,
        data.bmi,
        data.diabetes_pedigree_function,
        data.age
    ]])

    probability = model.predict_proba(features)[0][1]

    return {
        "probability": round(float(probability), 2),
        "risk_level": "High" if probability >= 0.6 else "Low"
    }
