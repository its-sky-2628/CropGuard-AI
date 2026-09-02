from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
import numpy as np
import json
import io
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "cropguard_model.keras")
CLASS_PATH = os.path.join(BASE_DIR, "model", "class_names.json")

app = FastAPI(
    title="CropGuard API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading CropGuard AI model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

print(f"Model loaded successfully. Classes: {len(class_names)}")


@app.get("/")
def home():
    return {
        "status": "online",
        "app": "CropGuard",
        "model_loaded": True,
        "classes": len(class_names),
        "message": "CropGuard AI API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        return {
            "success": False,
            "message": "Please upload a valid image file."
        }

    try:
        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        image = image.resize((224, 224))

        image_array = np.array(image, dtype=np.float32)

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        predictions = model.predict(
            image_array,
            verbose=0
        )

        predicted_index = int(
            np.argmax(predictions[0])
        )

        confidence = float(
            np.max(predictions[0]) * 100
        )

        raw_prediction = class_names[predicted_index]

        prediction = (
            raw_prediction
            .replace("___", " - ")
            .replace("_", " ")
        )

        return {
            "success": True,
            "filename": file.filename,
            "message": "Analysis completed successfully",
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "raw_prediction": raw_prediction
        }

    except Exception as e:
        print("Prediction error:", str(e))

        return {
            "success": False,
            "message": str(e)
        }
