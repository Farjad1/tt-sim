#!/usr/bin/env python3
"""Download BlurBall motion blur annotation dataset."""
import os
import sys
import zipfile
import tempfile
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = DATA_DIR / "blurball"
URL = "https://cloud.cs.uni-tuebingen.de/index.php/s/C3pJEPKWQAkono7/download"


def _print_summary():
    n_files = sum(1 for f in OUTPUT_DIR.rglob("*") if f.is_file())
    size_mb = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file()) / 1e6
    print(f"  {n_files} files, {size_mb:.1f} MB")


def download():
    """Download the BlurBall dataset (idempotent)."""
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        print(f"Already exists: {OUTPUT_DIR}")
        _print_summary()
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading BlurBall dataset from {URL} ...")

    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    try:
        urllib.request.urlretrieve(URL, tmp.name)
        print(f"  Downloaded to temp file.")

        if zipfile.is_zipfile(tmp.name):
            with zipfile.ZipFile(tmp.name, "r") as zf:
                zf.extractall(OUTPUT_DIR)
            print(f"Extracted to {OUTPUT_DIR}")
        else:
            # May be a tar or single file
            import tarfile

            if tarfile.is_tarfile(tmp.name):
                with tarfile.open(tmp.name) as tf:
                    tf.extractall(OUTPUT_DIR)
                print(f"Extracted to {OUTPUT_DIR}")
            else:
                dest = OUTPUT_DIR / "blurball_data"
                os.rename(tmp.name, str(dest))
                print(f"Saved to {dest}")
                tmp.name = None
    except Exception as e:
        print(f"Error downloading: {e}", file=sys.stderr)
        print(
            f"Please download manually from:\n"
            f"  https://cloud.cs.uni-tuebingen.de/index.php/s/C3pJEPKWQAkono7\n"
            f"Then extract to: {OUTPUT_DIR}",
            file=sys.stderr,
        )
        sys.exit(1)
    finally:
        if tmp.name and os.path.exists(tmp.name):
            os.unlink(tmp.name)

    print(f"Downloaded to {OUTPUT_DIR}")
    _print_summary()


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print(f"\nDownloads to: {OUTPUT_DIR}")
        print(f"Source: {URL}")
        sys.exit(0)
    download()
