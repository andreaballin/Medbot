"""Gate rules and pipeline decisions (no index, no LLM needed).  Run: pytest"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config, gate, generator  # noqa: E402
from app.pipeline import ANSWERED, OUT_OF_SCOPE, Pipeline  # noqa: E402


@pytest.mark.parametrize("q", ["", "  ", "??", "12345", "a" * (config.MAX_QUESTION_CHARS + 1)])
def test_invalid(q):
    assert gate.rule_gate(q)[0] == gate.INVALID


@pytest.mark.parametrize("q", [
    "I think I'm having a stroke, my face is drooping.",
    "My husband is having a heart attack",
    "I can't breathe",
    "My baby is not breathing",
    "I want to kill myself",
    "I just swallowed the whole bottle of pills",
    "I have sudden chest pain and my arm is numb",
])
def test_emergency(q):
    assert gate.rule_gate(q)[0] == gate.EMERGENCY


@pytest.mark.parametrize("q", [
    "What dose of antibiotics should I take for my sore throat?",
    "Should I stop taking my blood pressure medication?",
    "Can you diagnose me?",
    "What disease do I have?",
    "Do I have diabetes?",
    "Is it safe for me to take ibuprofen?",
    "What should my patient take for his hypertension?",
    "Should my child go to the hospital?",
])
def test_patient_specific(q):
    assert gate.rule_gate(q)[0] == gate.PATIENT


@pytest.mark.parametrize("q", [
    "What is type 2 diabetes?",
    "What are the warning signs of a stroke?",
    "What is the usual adult dose of amoxicillin for streptococcal pharyngitis?",
    "What are the diagnostic criteria for sepsis?",
    "How is a heart attack treated?",
    "How do beta blockers lower blood pressure?",
    "What causes chest pain?",
    "What are the risk factors for suicide?",
    "When should antibiotics be given for otitis media in children?",
    "How is an opioid overdose managed?",
])
def test_general_questions_pass(q):
    assert gate.rule_gate(q)[0] == gate.OK


class FakeRetriever:
    """Returns a fixed top score and a fixed similarity to the patient prototypes."""

    def __init__(self, top=0.9, patient=0.1):
        self.top, self.patient = top, patient

    def embed_plain(self, texts):
        return np.eye(len(list(texts)), 4, dtype="float32")

    def embed(self, question):
        return np.zeros(4, "float32"), np.array([self.patient, 0, 0, 0], "float32")

    def search(self, vec, k=None):
        return [{"source": "test", "title": "Diabetes", "text": "Type 2 diabetes is ...", "score": self.top}]


def make(top=0.9, patient=0.1, reply="It is a chronic condition [1]."):
    calls = []

    def llm(messages):
        calls.append(messages)
        return reply

    return Pipeline(FakeRetriever(top, patient), llm=llm, log=False), calls


def test_answered_with_sources_and_citations():
    p, calls = make()
    r = p.answer("What is type 2 diabetes?")
    assert r.decision == ANSWERED and len(calls) == 1
    assert r.sources[0]["title"] == "Diabetes" and r.citations == {"cited": [1], "invalid": []}
    assert calls[0][0]["content"] == generator.SYSTEM and "[1] (test: Diabetes)" in calls[0][-1]["content"]


def test_out_of_scope_skips_llm():
    p, calls = make(top=config.SCOPE_MIN - 0.05)
    r = p.answer("What is the capital of France?")
    assert r.decision == OUT_OF_SCOPE and r.answer == generator.NOINFO_MSG and not calls and not r.sources


@pytest.mark.parametrize("q,decision", [
    ("I can't breathe", gate.EMERGENCY),
    ("Should I stop taking my blood pressure medication?", gate.PATIENT),
    ("", gate.INVALID),
])
def test_gate_decisions_skip_llm(q, decision):
    p, calls = make()
    r = p.answer(q)
    assert r.decision == decision and r.answer and not calls


def test_semantic_gate_catches_what_rules_miss():
    p, calls = make(patient=config.PATIENT_MIN + 0.05)
    r = p.answer("Please look at these results for the gentleman in bed four and decide the treatment")
    assert r.decision == gate.PATIENT and not calls


def test_invalid_citation_is_reported():
    p, _ = make(reply="See [1] and [7].")
    assert p.answer("What is type 2 diabetes?").citations == {"cited": [1], "invalid": [7]}


@pytest.mark.parametrize("raw,clean", [
    ("Plain answer [1].", "Plain answer [1]."),
    ("<think>let me see</think>\n\nAnswer [1].", "Answer [1]."),
    ("reasoning without opening tag\n</think>\n\nAnswer [1].", "Answer [1]."),
    ("<think>still thinking", ""),
])
def test_strip_reasoning(raw, clean):
    assert generator.strip_reasoning(raw) == clean


def test_stream_ends_with_final_result():
    p = Pipeline(FakeRetriever(), llm_stream=lambda m: iter(["It is ", "chronic [1]."]), log=False)
    out = list(p.answer_stream("What is type 2 diabetes?"))
    assert out[-1].answer == "It is chronic [1]." and out[-1].citations["cited"] == [1]
