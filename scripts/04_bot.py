"""Medical databot: RAG over the sources + local LLM via Ollama.

    pip install sentence-transformers numpy requests
    ollama pull qwen3:4b          # then, after fine-tuning: ollama create medbot ...
    python scripts/04_bot.py [--model qwen3:4b] [--k 5]
"""
import argparse
import json
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

SYSTEM = """You are MedBot, an encyclopedic medical reference assistant.
Answer ONLY using the numbered context passages. Cite them like [1], [2].
If the context does not contain the answer, say you don't have enough information.
Write in clear English. You provide general educational information, not personal medical advice:
for questions about a specific person's symptoms, diagnosis or treatment, give general background
and tell the user to consult a qualified healthcare professional; for emergencies, tell them to call emergency services."""


class Retriever:
    def __init__(self):
        self.emb = np.load("data/index_emb.npy").astype("float32")
        self.meta = [json.loads(l) for l in open("data/index_meta.jsonl", encoding="utf-8")]
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def search(self, q, k):
        qv = self.model.encode(
            "Represent this sentence for searching relevant passages: " + q,
            normalize_embeddings=True)
        idx = np.argsort(-(self.emb @ qv))[:k]
        return [self.meta[i] for i in idx]


def ask(retriever, model, question, k, history):
    hits = retriever.search(question, k)
    ctx = "\n\n".join(f"[{i+1}] ({h['source']}: {h['title']}) {h['text']}" for i, h in enumerate(hits))
    msgs = [{"role": "system", "content": SYSTEM}] + history + [
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}]
    r = requests.post("http://localhost:11434/api/chat",
                      json={"model": model, "messages": msgs, "stream": False,
                            "think": False, "options": {"temperature": 0.2}}, timeout=600)
    r.raise_for_status()
    answer = r.json()["message"]["content"]
    sources = "\n".join(f"  [{i+1}] {h['source']}: {h['title']}" for i, h in enumerate(hits))
    return answer, sources


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:4b")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    ret = Retriever()
    print("MedBot ready (educational only, not medical advice). Ctrl+C to quit.")
    history = []
    while True:
        q = input("\n> ").strip()
        if not q:
            continue
        if q.startswith("/img "):  # skin classifier -> RAG explanation
            from importlib import import_module
            sk = import_module("07_predict_skin")
            res, ok = sk.predict(q[5:].strip().strip('"'))
            print("\nImage classifier (NOT a diagnosis):")
            for c, p in res:
                print(f"  {p:5.1%}  {c}")
            if not ok:
                print("Confidence too low: image not recognised as one of the supported skin lesions. "
                      "See a dermatologist for any concerning lesion.")
                continue
            q = (f"What is {sk.RAG_QUERY[res[0][0]]}? Describe it, how it is usually evaluated, "
                 "and when to see a doctor.")
            print(f"\n(explaining top class: {res[0][0]})")
        ans, src = ask(ret, a.model, q, a.k, history[-4:])
        print(f"\n{ans}\n\nSources:\n{src}")
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
