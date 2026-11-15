"""Scarica fonti mediche pubbliche e gratuite in data/raw/.

Fonti (tutte libere, nessun pagamento):
  medlineplus : NIH/NLM, ~1000 voci per pazienti, pubblico dominio (piccolo, veloce)
  statpearls  : NCBI Bookshelf, ~9000 articoli clinici peer-reviewed, CC BY-NC-ND (~2 GB)
  textbooks   : metti tu i PDF di libri open (OpenStax, NCBI Bookshelf open) in data/raw/textbooks/

Uso:
    python 01_download_sources.py medlineplus
    python 01_download_sources.py statpearls
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
    # link al file topics non compresso piu' recente (mplus_topics_YYYY-MM-DD.xml)
    links = sorted(set(re.findall(r'https://medlineplus\.gov/xml/mplus_topics_\d{4}-\d{2}-\d{2}\.xml', html)))
    get(links[-1], RAW / "medlineplus" / "topics.xml")


def statpearls():
    get(STATPEARLS, RAW / "statpearls.tar.gz")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "medlineplus"
    {"medlineplus": medlineplus, "statpearls": statpearls}[what]()
    (RAW / "textbooks").mkdir(parents=True, exist_ok=True)
