from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from pathlib import Path
import shutil
import uuid

app = FastAPI()

MODEL_PATH = Path(
    "E:/car_object_detection/car_object_detection/agents/"
    "runs/detect/runs_yolov8/car_only_yolov8n/weights/best.pt"
)

model = None

@app.on_event("startup")
def load_model():
    global model
    model = YOLO(MODEL_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = f"temp_{uuid.uuid4()}.jpg"

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    results = model(temp_path)
    detections = results[0].boxes.xyxy.tolist()

    return {"detections": detections}
