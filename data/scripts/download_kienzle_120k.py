#!/usr/bin/env python3
"""Download Kienzle 120k trajectory dataset."""
import os
import sys
import zipfile
import tempfile
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = DATA_DIR / "kienzle_120k"
REPO_URL = "https://github.com/KieDani/UpliftingTableTennis"

# Known part URLs from the repo README (may change over time)
PART_URLS = [
    "https://mediastore.rz.uni-augsburg.de/get/JN7DjGb_qQ/",
    "https://mediastore.rz.uni-augsburg.de/get/BrOeqGhYx5/",
    "https://mediastore.rz.uni-augsburg.de/get/LiTkJbQf0p/",
]


def _print_summary():
    n_files = sum(1 for f in OUTPUT_DIR.rglob("*") if f.is_file())
    size_mb = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file()) / 1e6
    print(f"  {n_files} files, {size_mb:.1f} MB")


def download():
    """Download the Kienzle 120k dataset (idempotent)."""
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        print(f"Already exists: {OUTPUT_DIR}")
        _print_summary()
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Kienzle 120k dataset ...")
    print(f"  Source repo: {REPO_URL}")
    print(f"  Attempting to download {len(PART_URLS)} parts ...")

    any_failed = False
    for i, url in enumerate(PART_URLS, 1):
        print(f"  Part {i}/{len(PART_URLS)}: {url}")
        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        try:
            urllib.request.urlretrieve(url, tmp.name)
            if zipfile.is_zipfile(tmp.name):
                with zipfile.ZipFile(tmp.name, "r") as zf:
                    zf.extractall(OUTPUT_DIR)
                print(f"    Extracted.")
            else:
                dest = OUTPUT_DIR / f"part_{i}"
                os.rename(tmp.name, str(dest))
                print(f"    Saved as {dest}")
                tmp.name = None
        except Exception as e:
            print(f"    Failed: {e}", file=sys.stderr)
            any_failed = True
        finally:
            if tmp.name and os.path.exists(tmp.name):
                os.unlink(tmp.name)

    if any_failed:
        print(
            f"\nSome parts failed to download. The part URLs may have changed."
            f"\nPlease check the repo for current download links:"
            f"\n  {REPO_URL}"
            f"\nThen place the data in: {OUTPUT_DIR}",
            file=sys.stderr,
        )

    _print_summary()


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print(f"\nDownloads to: {OUTPUT_DIR}")
        print(f"Source repo: {REPO_URL}")
        print("\nNote: Part URLs may change. Check the repo README for current links.")
        sys.exit(0)
    download()
