"""Answer generation with a local LLM served by Ollama, plus citation checking."""
import json
import re

import requests

from . import config

SYSTEM = """You are MedBot, an encyclopedic medical reference assistant for clinicians and medical students.
Answer ONLY using the numbered context passages. Cite them like [1], [2].
If the context does not contain the answer, say you don't have enough information.
Write in clear English. You provide reference information, not decisions about an individual patient:
state general facts (definitions, criteria, usual management, dosing as written in the sources) and leave
the application to a specific person to the treating clinician."""

NOINFO_MSG = "The indexed sources do not contain enough information to answer this question."


class LLMUnavailable(RuntimeError):
    """Ollama is not reachable or the model is missing."""


def build_messages(question, hits, history=()):
    ctx = "\n\n".join(f"[{i + 1}] ({h['source']}: {h['title']}) {h['text']}" for i, h in enumerate(hits))
    return [{"role": "system", "content": SYSTEM}, *history,
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}]


def _post(messages, stream):
    try:
        r = requests.post(f"{config.OLLAMA_URL}/api/chat", stream=stream, timeout=600,
                          json={"model": config.MODEL, "messages": messages, "stream": stream,
                                "options": {"temperature": config.TEMPERATURE}})
        r.raise_for_status()
        return r
    except requests.RequestException as e:
        raise LLMUnavailable(f"Ollama request failed ({config.MODEL} at {config.OLLAMA_URL}): {e}") from e


def strip_reasoning(text):
    """Drop model reasoning that leaked into the answer (<think>...</think>, or text before a stray </think>).

    Ollama normally returns reasoning in a separate field, which is ignored here; this is the fallback.
    While a <think> block is still open (streaming), everything from the tag on is hidden.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.split("<think>")[0].lstrip()


def generate(messages):
    return strip_reasoning(_post(messages, stream=False).json()["message"]["content"])


def generate_stream(messages):
    """Yield the answer piece by piece (reasoning removed)."""
    raw, sent = "", 0
    for line in _post(messages, stream=True).iter_lines():
        if not line:
            continue
        raw += json.loads(line).get("message", {}).get("content", "")
        clean = strip_reasoning(raw)
        if len(clean) > sent and clean.startswith(clean[:sent]):
            yield clean[sent:]
            sent = len(clean)


def check_citations(answer, n_passages):
    """Which [n] markers the answer uses, and which of them do not match a passage."""
    cited = sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer)})
    return {"cited": [c for c in cited if 1 <= c <= n_passages],
            "invalid": [c for c in cited if not 1 <= c <= n_passages]}
