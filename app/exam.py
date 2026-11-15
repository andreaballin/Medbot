"""Exam mode: an Italian-speaking professor asks questions from a pre-built bank and grades the answers.

The question bank (data/question_bank.jsonl) is built offline by scripts/10_build_question_bank.py:
each question comes from one source passage and stores an Italian model answer and key points.
At exam time the LLM only has to grade one answer against one passage, which keeps it fast.
Source passages are in English; everything shown to the student is in Italian.
"""
import json
import random
import threading
import time

from . import config, generator

BANK_FILE = config.DATA / "question_bank.jsonl"
PROGRESS_FILE = config.DATA / "progress.json"

CORRECT, PARTIAL, WRONG = "corretta", "parziale", "sbagliata"
SCORE = {CORRECT: 2, PARTIAL: 1, WRONG: 0}

GEN_SYSTEM = """You are a professor of medicine preparing an oral exam for Italian medical students.
From the English passage you are given, write ONE exam question in Italian that can be answered from the passage alone.
Do not mention the passage in the question. Then write the model answer in Italian (2-4 sentences, only facts stated
in the passage) and 2 to 5 short key points in Italian that a correct answer must contain.
Use correct Italian medical terminology.
Return ONLY a JSON object: {"domanda": "...", "risposta": "...", "punti_chiave": ["...", "..."]}"""

GRADE_SYSTEM = """Sei un professore di medicina che interroga uno studente italiano. Sei rigoroso ma incoraggiante.
Valuta la risposta dello studente confrontandola SOLO con il brano e con i punti chiave forniti: non usare altre conoscenze.
- "corretta": contiene la sostanza di tutti i punti chiave e nessun errore importante.
- "parziale": contiene solo alcuni punti chiave, oppure contiene un errore accanto a cose giuste.
- "sbagliata": non contiene i punti chiave o afferma cose in contrasto con il brano.
Conta il contenuto, non la forma: sinonimi e parole diverse vanno bene.
Nella spiegazione rivolgiti allo studente dandogli del tu, in 2-4 frasi: di' cosa ha detto bene, cosa mancava e correggi
gli errori spiegando il perche'. Scrivi tutto in italiano.
Rispondi SOLO con un oggetto JSON:
{"voto": "corretta|parziale|sbagliata", "punti_mancanti": ["..."], "errori": ["..."], "spiegazione": "..."}"""


def make_question(chunk, llm_json=None):
    """Generate one bank entry from a source chunk. Returns None when the model output is unusable."""
    llm_json = llm_json or generator.generate_json
    data = llm_json([{"role": "system", "content": GEN_SYSTEM},
                     {"role": "user", "content": f"Topic: {chunk['title']}\n\nPassage:\n{chunk['text']}"}])
    if not data:
        return None
    q, a, points = data.get("domanda"), data.get("risposta"), data.get("punti_chiave")
    if not (isinstance(q, str) and isinstance(a, str) and isinstance(points, list)):
        return None
    points = [p.strip() for p in points if isinstance(p, str) and p.strip()]
    if len(q.strip()) < 10 or len(a.strip()) < 20 or not points:
        return None
    return {"id": chunk["chunk_id"], "topic": chunk["title"], "source": chunk["source"],
            "question": q.strip(), "reference": a.strip(), "key_points": points, "passage": chunk["text"]}


def _as_list(v):
    return [x.strip() for x in v if isinstance(x, str) and x.strip()] if isinstance(v, list) else []


class Exam:
    def __init__(self, bank_file=BANK_FILE, progress_file=PROGRESS_FILE, llm_json=None, seed=None):
        self.bank_file, self.progress_file = bank_file, progress_file
        self.llm_json = llm_json or generator.generate_json
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.questions = {}
        if bank_file.exists():
            for line in open(bank_file, encoding="utf-8"):
                if line.strip():
                    q = json.loads(line)
                    self.questions[q["id"]] = q
        self.progress = json.loads(progress_file.read_text(encoding="utf-8")) if progress_file.exists() else {}

    def topics(self):
        """[(topic, number of questions)] sorted by name."""
        counts = {}
        for q in self.questions.values():
            counts[q["topic"]] = counts.get(q["topic"], 0) + 1
        return sorted(counts.items())

    def _weight(self, qid):
        """Questions answered badly come back more often; correct ones rarely."""
        p = self.progress.get(qid)
        if not p:
            return 3.0
        return {0: 4.0, 1: 2.0, 2: 0.5}[p["last"]]

    def next_question(self, topic=None, exclude=None):
        pool = [q for q in self.questions.values() if not topic or q["topic"] == topic]
        if len(pool) > 1 and exclude:
            pool = [q for q in pool if q["id"] != exclude]
        if not pool:
            return None
        return self.rng.choices(pool, weights=[self._weight(q["id"]) for q in pool])[0]

    def grade(self, qid, answer):
        """Grade a student answer. Returns a dict with verdict, score, feedback and the reference answer."""
        q = self.questions[qid]
        answer = (answer or "").strip()
        out = {"question_id": qid, "topic": q["topic"], "reference": q["reference"], "key_points": q["key_points"],
               "source": f"{q['source']}: {q['topic']}", "missing": [], "errors": []}
        if len(answer) < 3:
            out.update(verdict=WRONG, missing=q["key_points"],
                       feedback="Non hai dato una risposta. Leggi la risposta attesa e riprova più tardi.")
        else:
            user = (f"Brano (in inglese):\n{q['passage']}\n\nDomanda: {q['question']}\n\n"
                    f"Risposta attesa: {q['reference']}\n\nPunti chiave:\n- " + "\n- ".join(q["key_points"])
                    + f"\n\nRisposta dello studente: {answer}")
            data = self.llm_json([{"role": "system", "content": GRADE_SYSTEM}, {"role": "user", "content": user}])
            verdict = str((data or {}).get("voto", "")).strip().lower()
            if verdict not in SCORE:
                out.update(verdict=None, score=None,
                           feedback="Non sono riuscito a valutare la risposta. Confrontala tu con la risposta attesa.")
                return out
            out.update(verdict=verdict, missing=_as_list(data.get("punti_mancanti")),
                       errors=_as_list(data.get("errori")), feedback=str(data.get("spiegazione", "")).strip())
        out["score"] = SCORE[out["verdict"]]
        self._record(qid, out["score"])
        return out

    def _record(self, qid, score):
        with self.lock:
            p = self.progress.setdefault(qid, {"attempts": 0, "scores": []})
            p["attempts"] += 1
            p["last"] = score
            p["scores"] = (p["scores"] + [score])[-10:]
            p["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.progress_file.parent.mkdir(parents=True, exist_ok=True)
            self.progress_file.write_text(json.dumps(self.progress, ensure_ascii=False, indent=1), encoding="utf-8")

    def summary(self):
        """Per-topic progress: questions in the bank, questions tried, last-attempt score in percent."""
        rows = {}
        for q in self.questions.values():
            r = rows.setdefault(q["topic"], {"topic": q["topic"], "questions": 0, "tried": 0, "points": 0})
            r["questions"] += 1
            p = self.progress.get(q["id"])
            if p:
                r["tried"] += 1
                r["points"] += p["last"]
        for r in rows.values():
            r["percent"] = round(100 * r["points"] / (2 * r["tried"])) if r["tried"] else None
        tried = sum(r["tried"] for r in rows.values())
        points = sum(r["points"] for r in rows.values())
        return {"questions": len(self.questions), "tried": tried,
                "percent": round(100 * points / (2 * tried)) if tried else None,
                "topics": sorted(rows.values(), key=lambda r: r["topic"])}
