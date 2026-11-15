"""Staged question-answering pipeline.

    question -+-> [1a] rule gate          (in parallel)
              +-> [1b] question embedding
              [2] semantic gate  -> fixed message for patient-specific requests
              [3] retrieval      -> fixed message when the best passage is too dissimilar (out of scope)
              [4] generation     -> local LLM, answers only from the passages
              [5] citation check
"""
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from . import config, gate, generator

ANSWERED, OUT_OF_SCOPE = "answered", "out_of_scope"


@dataclass
class Result:
    decision: str
    answer: str = ""
    sources: list = field(default_factory=list)      # [{n, source, title, score, text}]
    citations: dict = field(default_factory=dict)    # {cited: [...], invalid: [...]}
    top_score: float | None = None
    patient_score: float | None = None
    timings: dict = field(default_factory=dict)      # seconds per stage
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self):
        return asdict(self)


class Pipeline:
    def __init__(self, retriever=None, llm=None, llm_stream=None, log=True):
        if retriever is None:
            from .retriever import Retriever
            retriever = Retriever()
        self.retriever = retriever
        self.llm = llm or generator.generate
        self.llm_stream = llm_stream or generator.generate_stream
        self.log = log
        self.prototypes = retriever.embed_plain(gate.PATIENT_PROTOTYPES)
        self.pool = ThreadPoolExecutor(max_workers=4)

    # stages 1-3: everything that happens before the LLM is called
    def prepare(self, question):
        """Return (result, hits). hits is None when the gate already produced the final result."""
        t = {}
        question = (question or "").strip()
        t0 = time.perf_counter()
        f_rule = self.pool.submit(gate.rule_gate, question)
        f_emb = self.pool.submit(self.retriever.embed, question or " ")
        decision, msg = f_rule.result()
        if decision != gate.OK:                       # no need to wait for the embedding
            f_emb.cancel()
            t["gate"] = time.perf_counter() - t0
            return Result(decision, msg, timings=t), None
        qvec, plain = f_emb.result()
        decision, msg, pscore = gate.semantic_gate(plain, self.prototypes)
        t["gate"] = time.perf_counter() - t0
        if decision != gate.OK:
            return Result(decision, msg, patient_score=pscore, timings=t), None

        t0 = time.perf_counter()
        hits = self.retriever.search(qvec, config.TOP_K)
        t["retrieval"] = time.perf_counter() - t0
        top = hits[0]["score"] if hits else 0.0
        res = Result(ANSWERED, top_score=top, patient_score=pscore, timings=t)
        if top < config.SCOPE_MIN:
            res.decision, res.answer = OUT_OF_SCOPE, generator.NOINFO_MSG
            return res, None
        res.sources = [{"n": i + 1, "source": h["source"], "title": h["title"],
                        "score": round(h["score"], 3), "text": h["text"]} for i, h in enumerate(hits)]
        return res, hits

    def _finish(self, res, n_hits=0):
        if res.decision == ANSWERED:
            res.citations = generator.check_citations(res.answer, n_hits)
        res.timings = {k: round(v, 3) for k, v in res.timings.items()}
        if self.log:
            config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(config.LOG_FILE, "a", encoding="utf-8") as f:   # no question text: may contain patient data
                f.write(json.dumps({"id": res.request_id, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "model": config.MODEL, "decision": res.decision, "top_score": res.top_score,
                                    "patient_score": res.patient_score, "citations": res.citations,
                                    "timings": res.timings}) + "\n")
        return res

    def answer(self, question, history=()):
        res, hits = self.prepare(question)
        if hits is None:
            return self._finish(res)
        t0 = time.perf_counter()
        res.answer = self.llm(generator.build_messages(question.strip(), hits, list(history)))
        res.timings["generation"] = time.perf_counter() - t0
        return self._finish(res, len(hits))

    def answer_stream(self, question, history=()):
        """Yield Result objects with a growing answer; the last one is final (citations filled in)."""
        res, hits = self.prepare(question)
        if hits is None:
            yield self._finish(res)
            return
        t0 = time.perf_counter()
        for piece in self.llm_stream(generator.build_messages(question.strip(), hits, list(history))):
            res.answer += piece
            yield res
        res.timings["generation"] = time.perf_counter() - t0
        yield self._finish(res, len(hits))
