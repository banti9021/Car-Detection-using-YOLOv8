import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]: %(message)s'
)

project_name = "car_object_detection"

list_of_files = [

    # ================= DATA =================
    f"{project_name}/data/customer_chats.csv",
    f"{project_name}/data/customer_churn.csv",
    f"{project_name}/data/policies.txt",

   
    f"{project_name}/agents/object_detection.py",
   

    # ================= MODELS =================
    f"{project_name}/models/sentiment_model/.gitkeep",
    f"{project_name}/models/churn_model.joblib",

    # ================= EMBEDDINGS =================
    f"{project_name}/embeddings/policy_vectors/.gitkeep",

    # ================= BACKEND =================
    f"{project_name}/backend/__init__.py",
    f"{project_name}/backend/app.py",

    # ================= FRONTEND =================
    f"{project_name}/frontend/app.py",

    # ================= UTILS =================
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/text_cleaner.py",
    f"{project_name}/utils/logger.py",

    # ================= LOGS =================
    f"{project_name}/logs/app.log",

    # ================= ROOT FILES =================
    f"{project_name}/requirements.txt",
    f"{project_name}/Dockerfile",
    f"{project_name}/docker-compose.yml",
    f"{project_name}/README.md",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir}")

    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")
