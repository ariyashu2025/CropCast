from pathlib import Path
BASE_DIR=Path(__file__).resolve().parent
DATA_PATH=BASE_DIR/"data"/"cropcast_dataset.csv"
CHART_DIR=BASE_DIR/"static"/"charts"
MODEL_DIR=BASE_DIR/"models"