"""Converte le fonti grezze in chunk di testo pulito: data/chunks.jsonl

Gestisce: MedlinePlus XML, StatPearls NXML (tar.gz in streaming), PDF in data/raw/textbooks/.
Uso:
    pip install pymupdf
    python 02_chunk.py [--limit-statpearls 500]
Chunk: {chunk_id, source, title, text}
"""
import argparse
import html
import json
import re
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/chunks.jsonl")


def clean_html(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def split_words(text, size=350, overlap=50):
    w = text.split()
    for i in range(0, len(w), max(1, size - overlap)):
        piece = w[i:i + size]
        if len(piece) >= 40:
            yield " ".join(piece)


def medlineplus():
    p = RAW / "medlineplus" / "topics.xml"
    if not p.exists():
        return
    for t in ET.parse(p).getroot().iter("health-topic"):
        if t.get("language") != "English":
            continue
        yield "medlineplus", t.get("title"), clean_html(t.findtext("full-summary"))


def itertext(el):
    return re.sub(r"\s+", " ", " ".join(el.itertext())).strip()


def statpearls(limit):
    p = RAW / "statpearls.tar.gz"
    if not p.exists():
        return
    n = 0
    with tarfile.open(p, "r|gz") as tar:
        for m in tar:
            if not m.name.endswith(".nxml"):
                continue
            root = ET.parse(tar.extractfile(m)).getroot()
            title = itertext(root.find(".//article-title")) if root.find(".//article-title") is not None else m.name
            body = root.find(".//body")
            if body is None:
                continue
            yield "statpearls", title, itertext(body)
            n += 1
            if limit and n >= limit:
                return


def textbooks():
    d = RAW / "textbooks"
    if not d.exists():
        return
    try:
        import fitz
    except ImportError:
        print("pip install pymupdf per i PDF")
        return
    for pdf in d.glob("*.pdf"):
        with fitz.open(pdf) as doc:
            yield "textbook", pdf.stem, "\n".join(pg.get_text() for pg in doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-statpearls", type=int, default=0)
    a = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for gen in (medlineplus(), statpearls(a.limit_statpearls), textbooks()):
            for source, title, text in gen:
                for j, c in enumerate(split_words(text)):
                    f.write(json.dumps({"chunk_id": f"{source}:{title}#{j}", "source": source,
                                        "title": title, "text": c}, ensure_ascii=False) + "\n")
                    n += 1
    print(f"{n} chunk -> {OUT}")


if __name__ == "__main__":
    main()
