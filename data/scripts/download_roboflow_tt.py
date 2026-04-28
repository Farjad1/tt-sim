#!/usr/bin/env python3
"""Download Roboflow table-tennis ball detection dataset (YOLO format)."""
import json
import os
import sys
import zipfile
import tempfile
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = DATA_DIR / "roboflow_tt"
PROJECT_URL = "https://universe.roboflow.com/computer-vision-project-mjsdu/table-tennis-ball-um5tc"

# Roboflow API endpoint pattern
API_URL_TEMPLATE = (
    "https://universe.roboflow.com/ds/"
    "?key={api_key}&format=yolov8"
)


def _print_summary():
    n_files = sum(1 for f in OUTPUT_DIR.rglob("*") if f.is_file())
    size_mb = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file()) / 1e6
    print(f"  {n_files} files, {size_mb:.1f} MB")


def download():
    """Download the Roboflow TT dataset (idempotent). Requires ROBOFLOW_API_KEY."""
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        print(f"Already exists: {OUTPUT_DIR}")
        _print_summary()
        return

    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        print("=" * 60)
        print("ROBOFLOW_API_KEY environment variable is not set.")
        print()
        print("To download this dataset:")
        print("  1. Create a free Roboflow account at https://roboflow.com")
        print("  2. Get your API key from Settings -> API Keys")
        print(f"  3. Visit: {PROJECT_URL}")
        print("  4. Export as YOLOv8, then run:")
        print()
        print("     ROBOFLOW_API_KEY=your_key python data/scripts/download_roboflow_tt.py")
        print()
        print("  Alternatively, download manually and extract to:")
        print(f"     {OUTPUT_DIR}")
        print("=" * 60)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Roboflow TT dataset ...")
    print(f"  Project: {PROJECT_URL}")

    # Try the roboflow pip package approach via API
    # Since we can't rely on the roboflow package, use direct download URL
    # The user should export from the Roboflow UI and get a download link
    print()
    print("Automated Roboflow download requires the 'roboflow' pip package.")
    print("To download manually:")
    print(f"  1. Visit: {PROJECT_URL}")
    print("  2. Click 'Download Dataset' -> YOLOv8 format -> 'show download code'")
    print(f"  3. Extract to: {OUTPUT_DIR}")
    print()
    print("Or install roboflow and use:")
    print("  pip install roboflow")
    print("  from roboflow import Roboflow")
    print(f'  rf = Roboflow(api_key="{api_key[:4]}...")')
    print('  project = rf.workspace("computer-vision-project-mjsdu").project("table-tennis-ball-um5tc")')
    print("  dataset = project.version(1).download(\"yolov8\")")
    sys.exit(1)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print(f"\nDownloads to: {OUTPUT_DIR}")
        print(f"Source: {PROJECT_URL}")
        print("\nRequires ROBOFLOW_API_KEY environment variable.")
        sys.exit(0)
    download()
