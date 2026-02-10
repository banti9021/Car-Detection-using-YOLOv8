from pathlib import Path
import shutil
import pandas as pd
from PIL import Image

CLASS_ID = 0

def prepare_yolo_dataset(
    data_root: Path,
    work_root: Path,
    csv_path: Path
):
    train_img_dir = data_root / "training_images"

    images_dir = work_root / "images"
    labels_dir = work_root / "labels"

    for split in ["train", "val"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    all_imgs = sorted(df["image"].unique())
    split_idx = int(0.9 * len(all_imgs))

    train_imgs = set(all_imgs[:split_idx])

    def get_split(name):
        return "train" if name in train_imgs else "val"

    for img_name, group in df.groupby("image"):
        src_img = train_img_dir / img_name
        if not src_img.exists():
            continue

        split = get_split(img_name)
        dst_img = images_dir / split / img_name
        shutil.copy2(src_img, dst_img)

        with Image.open(src_img) as im:
            w, h = im.size

        label_file = labels_dir / split / f"{src_img.stem}.txt"
        with open(label_file, "w") as f:
            for _, row in group.iterrows():
                cx = ((row.xmin + row.xmax) / 2) / w
                cy = ((row.ymin + row.ymax) / 2) / h
                bw = (row.xmax - row.xmin) / w
                bh = (row.ymax - row.ymin) / h
                f.write(f"{CLASS_ID} {cx} {cy} {bw} {bh}\n")

    data_yaml = work_root / "data.yaml"
    data_yaml.write_text(f"""
path: {work_root}
train: images/train
val: images/val

names:
  0: car
""")

    print("✅ Dataset prepared")
