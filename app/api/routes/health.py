from fastapi import APIRouter
from schemas.patient import DiabetesInput
from services.prediction_service import predict_diabetes

router = APIRouter(
    prefix="",
    tags=["Health & Prediction"]
)

@router.get("/health")
def health_check():
    return {"status": "Backend is running successfully"}

@router.post("/predict/diabetes")
def diabetes_prediction(data: DiabetesInput):
    return predict_diabetes(data)
