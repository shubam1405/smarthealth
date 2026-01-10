import os
import numpy as np
import tensorflow as tf
import json
from PIL import Image
from app.core.logs import logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class labels
with open(LABELS_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse mapping: 0 -> COVID, 1 -> NORMAL, ...
labels = {v: k for k, v in class_indices.items()}

IMG_SIZE = 224

def preprocess_image(image: Image.Image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_xray(image: Image.Image):
    try:
        logger.info("X-ray prediction request received")

        img = preprocess_image(image)
        preds = model.predict(img)[0]

        class_id = np.argmax(preds)
        confidence = preds[class_id]
        disease = labels[class_id]

        logger.info(f"X-ray prediction: {disease} | Confidence: {confidence}")

        return {
            "disease": disease,
            "confidence": round(float(confidence), 2)
        }

    except Exception as e:
        logger.error(f"X-ray prediction failed: {str(e)}")
        raise
