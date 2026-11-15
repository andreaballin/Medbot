"""HTTP API and web UI.

    uvicorn app.api:app            # API on http://localhost:8000, chat UI on http://localhost:8000/ui
"""
import json
import threading
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from . import config, exam, generator, skin
from .pipeline import Pipeline

_pipeline = None
_lock = threading.Lock()


def get_pipeline():
    """The single warm pipeline (index, embedding model and gate prototypes loaded once)."""
    global _pipeline
    with _lock:
        if _pipeline is None:
            _pipeline = Pipeline()
    return _pipeline


@asynccontextmanager
async def lifespan(app):
    get_pipeline()                      # load at startup, not on the first request
    yield


app = FastAPI(title="MedBot", version="0.1",
              description="Medical reference assistant for study and consultation. Not a medical device.",
              lifespan=lifespan)


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[Message] = []


def _ask(question, history=()):
    try:
        return get_pipeline().answer(question, history).to_dict()
    except generator.LLMUnavailable as e:
        raise HTTPException(503, str(e))


@app.post("/ask")
def ask(req: AskRequest):
    """Run the staged pipeline. `decision` is one of: answered, out_of_scope, patient_specific, emergency, invalid."""
    return _ask(req.question, [m.model_dump() for m in req.history[-4:]])


@app.post("/classify")
def classify(image: UploadFile = File(...), explain: bool = True):
    """Classify a skin lesion photo (demo, NOT a diagnosis) and explain the top class from the sources."""
    try:
        img = Image.open(image.file)
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "not a readable image")
    classes, recognised = skin.predict(img)
    out = {"classes": [{"name": c, "probability": round(p, 4)} for c, p in classes], "recognised": recognised,
           "disclaimer": "Image classifier demo, not a diagnosis. Melanoma recall on the test set is 64%."}
    if recognised and explain:
        out["explanation"] = _ask(skin.explain_question(classes[0][0]))
    return out


@app.get("/health")
def health():
    p = get_pipeline()
    manifest = json.loads(config.INDEX_MANIFEST.read_text(encoding="utf-8")) if config.INDEX_MANIFEST.exists() else None
    ollama, model_present = False, False
    try:
        tags = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=3).json()
        names = [m["name"] for m in tags.get("models", [])]
        ollama = True
        model_present = any(n == config.MODEL or n.split(":")[0] == config.MODEL for n in names)
    except requests.RequestException:
        pass
    return {"status": "ok" if ollama and model_present else "degraded", "chunks": len(p.retriever.meta),
            "index": manifest, "model": config.MODEL, "ollama_reachable": ollama, "model_present": model_present,
            "skin_model_present": config.SKIN_MODEL.exists(),
            "thresholds": {"scope_min": config.SCOPE_MIN, "patient_min": config.PATIENT_MIN}}


_exam, _exam_mtime = None, None


def get_exam():
    """The exam session; reloaded when the question bank file changes."""
    global _exam, _exam_mtime
    with _lock:
        mtime = exam.BANK_FILE.stat().st_mtime if exam.BANK_FILE.exists() else None
        if _exam is None or mtime != _exam_mtime:
            _exam, _exam_mtime = exam.Exam(), mtime
    return _exam


class GradeRequest(BaseModel):
    question_id: str
    answer: str


@app.get("/exam/topics")
def exam_topics():
    return [{"topic": t, "questions": n} for t, n in get_exam().topics()]


@app.get("/exam/question")
def exam_question(topic: str | None = None, exclude: str | None = None):
    """Next question (in Italian). Questions answered badly before are more likely to come back."""
    q = get_exam().next_question(topic, exclude)
    if q is None:
        raise HTTPException(404, "no questions for this topic: run python scripts/10_build_question_bank.py")
    return {"question_id": q["id"], "topic": q["topic"], "question": q["question"]}


@app.post("/exam/grade")
def exam_grade(req: GradeRequest):
    """Grade an answer: verdict (corretta / parziale / sbagliata), explanation, model answer and source."""
    e = get_exam()
    if req.question_id not in e.questions:
        raise HTTPException(404, "unknown question_id")
    try:
        return e.grade(req.question_id, req.answer)
    except generator.LLMUnavailable as err:
        raise HTTPException(503, str(err))


@app.get("/exam/progress")
def exam_progress():
    return get_exam().summary()


from .ui import mount_ui  # noqa: E402

app = mount_ui(app, get_pipeline, get_exam)
