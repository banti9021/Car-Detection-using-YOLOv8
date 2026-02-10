from pathlib import Path
from pipeline.data_preparation import prepare_yolo_dataset
from pipeline.training import train_yolo

DATA_ROOT = Path(r"E:\car_object_detection\data\archive (14)")
CSV_PATH = DATA_ROOT / "train_solution_bounding_boxes (1).csv"
WORK_ROOT = Path(r"E:\car_object_detection\artifacts\yolo_dataset")

if __name__ == "__main__":
    prepare_yolo_dataset(
        data_root=DATA_ROOT,
        work_root=WORK_ROOT,
        csv_path=CSV_PATH
    )

    train_yolo(
        data_yaml=str(WORK_ROOT / "data.yaml"),
        epochs=5,
        batch=5
    )
