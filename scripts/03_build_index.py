"""Build the vector index (numpy, no database) from data/chunks.jsonl.

    pip install sentence-transformers numpy
    python scripts/03_build_index.py
Output: data/index_emb.npy, data/index_meta.jsonl, data/index_manifest.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402

chunks = [json.loads(l) for l in open(config.DATA / "chunks.jsonl", encoding="utf-8")]
model = SentenceTransformer(config.EMB_MODEL)
texts = [f"{c['title']}. {c['text']}" for c in chunks]
emb = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
np.save(config.INDEX_EMB, emb.astype("float16"))
with open(config.INDEX_META, "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
sources = {}
for c in chunks:
    sources[c["source"]] = sources.get(c["source"], 0) + 1
config.INDEX_MANIFEST.write_text(json.dumps(
    {"emb_model": config.EMB_MODEL, "chunks": len(chunks), "sources": sources,
     "built": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2), encoding="utf-8")
print(len(chunks), "chunks indexed")
