"""
Download the 'Give Me Some Credit' dataset from Kaggle via the Kaggle API.
Dataset: https://www.kaggle.com/c/GiveMeSomeCredit
Prerequisites: place kaggle.json in ~/.kaggle/kaggle.json (chmod 600)
or set KAGGLE_USERNAME & KAGGLE_KEY in .env
"""
import logging
import os
import zipfile
from pathlib import Path

from src.python.config import RAW_DIR, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

KAGGLE_COMPETITION = "GiveMeSomeCredit"


def _ensure_kaggle_credentials() -> None:
    """Set env vars so kaggle CLI/lib can authenticate without a kaggle.json file."""
    if settings.kaggle_username and settings.kaggle_key:
        os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
        os.environ["KAGGLE_KEY"] = settings.kaggle_key
        logger.info("Kaggle credentials loaded from .env")
    else:
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json.exists():
            raise FileNotFoundError(
                "Kaggle credentials not found. Fill KAGGLE_USERNAME/KAGGLE_KEY in .env "
                "or place kaggle.json in ~/.kaggle/kaggle.json"
            )
        logger.info("Kaggle credentials found in ~/.kaggle/kaggle.json")


def download_give_me_some_credit(dest_dir: Path = RAW_DIR) -> Path:
    """Download & extract the Give Me Some Credit dataset to data/raw/"""
    _ensure_kaggle_credentials()

    # Import inside function so the config module can be tested without authenticated kaggle lib
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading competition '{KAGGLE_COMPETITION}' to {dest_dir} ...")

    api.competition_download_files(KAGGLE_COMPETITION, path=str(dest_dir), quiet=False)

    zip_path = dest_dir / f"{KAGGLE_COMPETITION}.zip"
    if zip_path.exists():
        logger.info(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        zip_path.unlink()

    logger.info(f"Done. Files available at: {list(dest_dir.glob('*.csv'))}")
    return dest_dir


if __name__ == "__main__":
    download_give_me_some_credit()