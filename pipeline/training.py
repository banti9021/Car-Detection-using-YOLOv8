from ultralytics import YOLO

def train_yolo(
    data_yaml: str,
    epochs=5,
    batch=5,
    imgsz=640,
    run_name="car_yolo"
):
    model = YOLO("yolov8n.pt")

    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project="runs_yolov8",
        name=run_name,
        exist_ok=True
    )

    return model
