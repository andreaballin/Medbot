"""Build the exam question bank (Italian questions from the English source chunks).

One question per chunk, generated with the local LLM. Slow on CPU (about a minute per question),
so it is resumable: chunks already in the bank are skipped and each question is saved as soon as it is ready.
    python scripts/10_build_question_bank.py --n 20
    python scripts/10_build_question_bank.py --n 10 --topic "Diabetes"     # titles containing the text
Output: data/question_bank.jsonl
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config, exam  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="questions to add")
    ap.add_argument("--topic", default="", help="only chunks whose title contains this text")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    chunks = [json.loads(l) for l in open(config.DATA / "chunks.jsonl", encoding="utf-8")]
    done = set()
    if exam.BANK_FILE.exists():
        done = {json.loads(l)["id"] for l in open(exam.BANK_FILE, encoding="utf-8") if l.strip()}
    # the first chunk of each topic holds the summary, so prefer it
    pool = [c for c in chunks if c["chunk_id"] not in done and a.topic.lower() in c["title"].lower()]
    random.Random(a.seed).shuffle(pool)
    pool.sort(key=lambda c: not c["chunk_id"].endswith("#0"))

    added = 0
    with open(exam.BANK_FILE, "a", encoding="utf-8") as f:
        for c in pool:
            if added >= a.n:
                break
            t0 = time.time()
            q = exam.make_question(c)
            if not q:
                print(f"skip (unusable output): {c['chunk_id']}")
                continue
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
            f.flush()
            added += 1
            print(f"{added}/{a.n} [{time.time() - t0:.0f}s] {q['topic']}: {q['question']}", flush=True)
    print(f"{added} questions added, {len(done) + added} in {exam.BANK_FILE}")
