"""Dense retriever over the numpy index built by scripts/03_build_index.py."""
import json

import numpy as np

from . import config

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Retriever:
    def __init__(self):
        if not config.INDEX_EMB.exists():
            raise FileNotFoundError(f"{config.INDEX_EMB} not found: run python scripts/03_build_index.py")
        if config.INDEX_MANIFEST.exists():
            manifest = json.loads(config.INDEX_MANIFEST.read_text(encoding="utf-8"))
            if manifest["emb_model"] != config.EMB_MODEL:
                raise RuntimeError(f"index was built with {manifest['emb_model']}, config uses {config.EMB_MODEL}: "
                                   "rebuild it with python scripts/03_build_index.py")
        from sentence_transformers import SentenceTransformer
        self.emb = np.load(config.INDEX_EMB).astype("float32")
        self.meta = [json.loads(l) for l in open(config.INDEX_META, encoding="utf-8")]
        self.model = SentenceTransformer(config.EMB_MODEL)

    def embed(self, question):
        """Return (retrieval vector, plain sentence vector) for a question, in one batch."""
        v = self.model.encode([QUERY_PREFIX + question, question], normalize_embeddings=True)
        return v[0], v[1]

    def embed_plain(self, texts):
        return self.model.encode(list(texts), normalize_embeddings=True)

    def search(self, vec, k=None):
        """Return the top-k chunks as dicts with an added "score" (cosine similarity)."""
        k = k or config.TOP_K
        scores = self.emb @ vec
        idx = np.argsort(-scores)[:k]
        return [dict(self.meta[i], score=float(scores[i])) for i in idx]
