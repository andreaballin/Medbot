"""MedBot in the terminal: the same staged pipeline as the API (see app/pipeline.py).

    ollama pull qwen3:4b          # then, after fine-tuning: ollama create medbot ...
    python scripts/04_bot.py [--model qwen3:4b] [--k 5]
Commands: /img path.jpg classifies a skin photo and explains the top class.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config, generator  # noqa: E402
from app.pipeline import ANSWERED, Pipeline  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.MODEL)
    ap.add_argument("--k", type=int, default=config.TOP_K)
    a = ap.parse_args()
    config.MODEL, config.TOP_K = a.model, a.k
    pipe = Pipeline()
    print(f"MedBot ready ({config.MODEL}; reference and study tool, not medical advice). Ctrl+C to quit.")
    history = []
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.startswith("/img "):  # skin classifier -> RAG explanation
            from app import skin
            classes, recognised = skin.predict(q[5:].strip().strip('"'))
            print("\nImage classifier (NOT a diagnosis):")
            for c, p in classes:
                print(f"  {p:5.1%}  {c}")
            if not recognised:
                print("Confidence too low: image not recognised as one of the supported skin lesions.")
                continue
            q = skin.explain_question(classes[0][0])
            print(f"\n(explaining top class: {classes[0][0]})")
        try:
            res = pipe.answer(q, history[-4:])
        except generator.LLMUnavailable as e:
            print(f"\n{e}")
            continue
        print(f"\n[{res.decision}] {res.answer}")
        if res.decision == ANSWERED:
            print("\nSources:\n" + "\n".join(
                f"  [{s['n']}] {s['source']}: {s['title']} ({s['score']:.2f})" for s in res.sources))
            history += [{"role": "user", "content": q}, {"role": "assistant", "content": res.answer}]
