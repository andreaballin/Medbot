"""Download free public medical sources into data/raw/.

Sources (all free):
  medlineplus : NIH/NLM, ~1000 patient-oriented topics, public domain (small, fast)
  statpearls  : NCBI Bookshelf, ~9000 peer-reviewed clinical articles, CC BY-NC-ND (~2 GB)
  textbooks   : put open-licensed book PDFs (OpenStax, open NCBI Bookshelf) in data/raw/textbooks/

Usage:
    python scripts/01_download_sources.py medlineplus
    python scripts/01_download_sources.py statpearls
"""
import re
import sys
import urllib.request
from pathlib import Path

RAW = Path("data/raw")
UA = {"User-Agent": "medbot-research/0.1"}
STATPEARLS = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz"


def get(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\r{dest.name}: {done >> 20} MB", end="", flush=True)
    print()


def medlineplus():
    html = urllib.request.urlopen(urllib.request.Request(
        "https://medlineplus.gov/xml.html", headers=UA), timeout=60).read().decode()
    # link to the latest uncompressed topics file (mplus_topics_YYYY-MM-DD.xml)
    links = sorted(set(re.findall(r'https://medlineplus\.gov/xml/mplus_topics_\d{4}-\d{2}-\d{2}\.xml', html)))
    get(links[-1], RAW / "medlineplus" / "topics.xml")


def statpearls():
    get(STATPEARLS, RAW / "statpearls.tar.gz")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "medlineplus"
    {"medlineplus": medlineplus, "statpearls": statpearls}[what]()
    (RAW / "textbooks").mkdir(parents=True, exist_ok=True)
