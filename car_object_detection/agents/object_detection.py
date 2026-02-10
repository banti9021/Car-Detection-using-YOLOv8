
# IMPORTS

from pathlib import Path
import shutil
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt



# PATH CONFIG (WINDOWS SAFE)


DATA_ROOT = Path(r"E:\car_object_detection\data\archive (14)")

TRAIN_IMG_DIR = DATA_ROOT / "training_images"
CSV_PATH = DATA_ROOT / "train_solution_bounding_boxes (1).csv"

WORK_ROOT = Path(
    r"E:\car_object_detection\car_object_detection\embeddings\policy_vectors\archive (14)\working\car_yolo"
)

IMAGES_DIR = WORK_ROOT / "images"
LABELS_DIR = WORK_ROOT / "labels"



# TRAINING CONFIG

IMG_SIZE = 640
BATCH = 5
EPOCHS = 5
RUN_NAME = "car_only_yolov8n"
CLASS_ID = 0


# LOAD CSV

df = pd.read_csv(CSV_PATH)
print(df.head())



# CREATE FOLDERS

for split in ["train", "val"]:
    (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
    (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)



# TRAIN / VAL SPLIT

all_imgs = sorted(df["image"].unique())
split_idx = int(0.9 * len(all_imgs))

train_imgs = set(all_imgs[:split_idx])
val_imgs = set(all_imgs[split_idx:])


def get_split(name):
    return "train" if name in train_imgs else "val"


# CONVERT TO YOLO FORMAT

grouped = df.groupby("image")

for img_name, group in grouped:
    src_img = TRAIN_IMG_DIR / img_name
    if not src_img.exists():
        print("Missing:", src_img)
        continue

    split = get_split(img_name)
    dst_img = IMAGES_DIR / split / img_name
    shutil.copy2(src_img, dst_img)

    with Image.open(src_img) as im:
        w, h = im.size

    label_path = LABELS_DIR / split / f"{src_img.stem}.txt"

    with open(label_path, "w") as f:
        for _, row in group.iterrows():
            cx = ((row.xmin + row.xmax) / 2) / w
            cy = ((row.ymin + row.ymax) / 2) / h
            bw = (row.xmax - row.xmin) / w
            bh = (row.ymax - row.ymin) / h
            f.write(f"{CLASS_ID} {cx} {cy} {bw} {bh}\n")

print(" YOLO dataset ready at:", WORK_ROOT)



# DEBUG CHECK (VERY IMPORTANT)

print("Source images:", len(list(TRAIN_IMG_DIR.glob("*.jpg"))))
print("Train images:", len(list((IMAGES_DIR / 'train').glob('*.jpg'))))
print("Val images:", len(list((IMAGES_DIR / 'val').glob('*.jpg'))))



# DATA.YAML

data_yaml = WORK_ROOT / "data.yaml"
data_yaml.write_text(f"""
path: {WORK_ROOT}
train: images/train
val: images/val

names:
  0: car
""")


# TRAIN MODEL

model = YOLO("yolov8n.pt")

model.train(
    data=str(data_yaml),
    imgsz=IMG_SIZE,
    epochs=EPOCHS,
    batch=BATCH,
    project="runs_yolov8",
    name=RUN_NAME,
    exist_ok=True
)
