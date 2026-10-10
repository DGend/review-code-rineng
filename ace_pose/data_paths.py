"""Location of the Busan multi-domain HSE dataset (AI-Hub dataSetSn=71439).

Set BIOERA_DATA_ROOT to the folder that holds new_images/ and new_json/.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get(
    "BIOERA_DATA_ROOT",
    PROJECT_ROOT / "이미지 및 라벨링 데이터" / "부산시_작업부담_data_최신본" / "data_최신본",
))
IMG_DIR = DATA_ROOT / "new_images"
JSON_DIR = DATA_ROOT / "new_json"
