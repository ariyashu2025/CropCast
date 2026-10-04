from pathlib import Path
BASE_DIR=Path(__file__).resolve().parent
DATA_PATH=BASE_DIR/"dataset"/"crop_yield.csv"
CHART_DIR=BASE_DIR/"static"/"charts"
MODEL_DIR=BASE_DIR/"models"