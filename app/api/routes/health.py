from fastapi import APIRouter, HTTPException
from app.core.logs import logger

from schemas.patient import DiabetesInput
from schemas.heart import HeartInput

from services.diabetes_prediction_service import predict_diabetes
from services.heart_prediction_service import predict_heart_disease

from fastapi import UploadFile, File
from PIL import Image
from models.xray.prediction_service import predict_xray
import io


router = APIRouter(
    prefix="",
    tags=["Health & Prediction"]
)

@router.get("/health")
def health_check():
    logger.info("Health check endpoint accessed")
    return {"status": "Backend is running successfully"}

@router.post("/predict/diabetes")
def diabetes_prediction(data: DiabetesInput):
    try:
        logger.info("Diabetes prediction request received")
        return predict_diabetes(data)
    except Exception as e:
        logger.error(f"Diabetes prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Diabetes prediction failed"
        )

@router.post("/predict/heart")
def heart_prediction(data: HeartInput):
    try:
        logger.info("Heart disease prediction request received")
        return predict_heart_disease(data)
    except Exception as e:
        logger.error(f"Heart disease prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Heart disease prediction failed"
        )
@router.post("/predict/xray")
async def xray_prediction(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        return predict_xray(image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
