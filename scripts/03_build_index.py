"""Crea l'indice vettoriale (numpy, nessun DB) da data/chunks.jsonl.

    pip install sentence-transformers numpy
    python 03_build_index.py
Output: data/index_emb.npy, data/index_meta.jsonl
"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer

EMB_MODEL = "BAAI/bge-small-en-v1.5"

chunks = [json.loads(l) for l in open("data/chunks.jsonl", encoding="utf-8")]
model = SentenceTransformer(EMB_MODEL)
texts = [f"{c['title']}. {c['text']}" for c in chunks]
emb = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
np.save("data/index_emb.npy", emb.astype("float16"))
with open("data/index_meta.jsonl", "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
print(len(chunks), "chunk indicizzati")
