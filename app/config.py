"""Central settings. Override any of them with MEDBOT_<NAME> environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name, default, cast=str):
    return cast(os.environ.get(f"MEDBOT_{name}", default))


MODEL = _env("MODEL", "qwen3:4b-instruct")            # Ollama model; set to "medbot" after fine-tuning
OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434")
EMB_MODEL = _env("EMB_MODEL", "BAAI/bge-small-en-v1.5")
TOP_K = _env("TOP_K", 5, int)
TEMPERATURE = _env("TEMPERATURE", 0.2, float)
MAX_QUESTION_CHARS = _env("MAX_QUESTION_CHARS", 1000, int)

# Gate thresholds (cosine similarity). Calibrate with: python scripts/09_eval.py
SCOPE_MIN = _env("SCOPE_MIN", 0.60, float)            # best passage below this -> out of scope
PATIENT_MIN = _env("PATIENT_MIN", 0.88, float)        # near-paraphrase of a patient-specific prototype

DATA = ROOT / "data"
INDEX_EMB = DATA / "index_emb.npy"
INDEX_META = DATA / "index_meta.jsonl"
INDEX_MANIFEST = DATA / "index_manifest.json"
SKIN_MODEL = ROOT / "models" / "skin_model.pt"
SKIN_MIN_CONF = _env("SKIN_MIN_CONF", 0.50, float)    # below this: "not recognised"
LOG_FILE = ROOT / "logs" / "requests.jsonl"
