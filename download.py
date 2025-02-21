"""Helper script to dowload any required data files for the tutorial."""
import argparse
import logging
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent / "data"

URLS = {
    "probes": "https://zenodo.org/records/5653517/files/probes_model_00745000.pt",
    "sdss": "https://zenodo.org/records/5653517/files/sdss_model_00745000.pt",
}

def download_weights(key):
    """Download GPT2 weights from Huggingface"""
    url = URLS[key]

    log.info(f"Downloading from {url}")
    response = requests.get(url, params={"download": True})

    path = DATA_PATH / "models" / Path(url).name
    path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Saving to {path}")
   
    with open(path, mode="wb") as file:
        file.write(response.content)

    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=download_weights.__doc__)

    parser.add_argument("key", choices=list(URLS), help="Model identifier")
    args = parser.parse_args()

    download_weights(args.key)