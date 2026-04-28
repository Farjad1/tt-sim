#!/usr/bin/env python3
"""Download AIMY real launcher trajectory dataset (HDF5)."""
import os
import sys
import urllib.request
import html.parser
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = DATA_DIR / "aimy"
URL = "https://webdav.tuebingen.mpg.de/aimy/"


class _LinkParser(html.parser.HTMLParser):
    """Extract href links from an HTML directory listing."""

    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value and not value.startswith(".."):
                    self.links.append(value)


def _print_summary():
    n_files = sum(1 for f in OUTPUT_DIR.rglob("*") if f.is_file())
    size_mb = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file()) / 1e6
    print(f"  {n_files} files, {size_mb:.1f} MB")


def download():
    """Download the AIMY dataset (idempotent)."""
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        print(f"Already exists: {OUTPUT_DIR}")
        _print_summary()
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading AIMY dataset from {URL} ...")

    try:
        # Fetch directory listing
        with urllib.request.urlopen(URL) as resp:
            listing = resp.read().decode("utf-8", errors="replace")

        parser = _LinkParser()
        parser.feed(listing)

        # Filter for data files (hdf5, h5, csv, etc.)
        files = [
            link
            for link in parser.links
            if not link.startswith("?")
            and not link.startswith("/")
            and "." in link
        ]

        if not files:
            print("No files found in directory listing. Saving listing for inspection.")
            (OUTPUT_DIR / "_listing.html").write_text(listing)
            print(f"  Saved listing to {OUTPUT_DIR / '_listing.html'}")
            return

        print(f"  Found {len(files)} files to download.")
        for fname in files:
            file_url = URL.rstrip("/") + "/" + fname
            dest = OUTPUT_DIR / fname
            if dest.exists():
                print(f"  Skipping (exists): {fname}")
                continue
            print(f"  Downloading: {fname} ...")
            urllib.request.urlretrieve(file_url, str(dest))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        print(f"Please download manually from: {URL}", file=sys.stderr)
        sys.exit(1)

    print(f"Downloaded to {OUTPUT_DIR}")
    _print_summary()


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print(f"\nDownloads to: {OUTPUT_DIR}")
        print(f"Source: {URL}")
        sys.exit(0)
    download()
