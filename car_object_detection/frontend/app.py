import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.title(" Car Detection using YOLOv8")

model = YOLO(
    "E:/car_object_detection/car_object_detection/agents/"
    "runs/detect/runs_yolov8/car_only_yolov8n/weights/best.pt"
)

file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name, format="JPEG")
        results = model(tmp.name)

    st.image(results[0].plot(), caption="Detection Result", use_column_width=True)
