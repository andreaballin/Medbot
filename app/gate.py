"""Request gate: decides whether a question may reach retrieval and the LLM.

Clinical profile (audience: clinicians and students): general questions about dosing, diagnostic
criteria or management pass. Only decisions about one specific person are refused, and first-person
emergencies get a fixed emergency message.
"""
import re

import numpy as np

from . import config

OK, INVALID, EMERGENCY, PATIENT = "ok", "invalid", "emergency", "patient_specific"

INVALID_MSG = "Please enter a medical question (up to {n} characters)."
EMERGENCY_MSG = ("This sounds like it could be an emergency. Call your local emergency number now. "
                 "MedBot is a reference tool and cannot help in an emergency.")
PATIENT_MSG = ("MedBot is a reference and study tool: it cannot make a diagnostic or treatment decision for a "
               "specific person. Ask the general question instead (for example the usual dosing, diagnostic "
               "criteria or management of a condition) and apply clinical judgement to the individual case.")

_RELATIVE = r"(?:patient|child|son|daughter|baby|kid|wife|husband|mother|father|mom|dad|partner|friend)"
_PERSON = rf"(?:i|we|me|he|she|they|him|her|(?:my|this|the|our) {_RELATIVE})"
_SUBJECT = rf"(?:i|i'm|i am|i've|i have|he|she|he's|she's|(?:my|this|the|our) {_RELATIVE})"

_EMERGENCY = [re.compile(p, re.I) for p in [
    rf"\b{_SUBJECT}\b.{{0,40}}\b(?:having|has|have) a (?:stroke|heart attack|seizure)\b",
    rf"\b{_SUBJECT} (?:can't|cannot|can not|is not|isn't|am not) breath",
    rf"\b{_SUBJECT}\b.{{0,30}}\b(?:not breathing|unconscious|unresponsive|turning blue)\b",
    r"\bi (?:want|am going|plan|intend) to (?:kill myself|end my life|die)\b",
    rf"\b{_SUBJECT} (?:just )?(?:took|swallowed|overdosed)\b.{{0,40}}\b(?:too many|overdose|whole bottle|all (?:of )?(?:my|the|his|her))\b",
    rf"\b{_SUBJECT}\b.{{0,30}}\b(?:severe|crushing|sudden) chest pain\b",
    r"\b(?:my|his|her) face is drooping\b",
]]

_PATIENT = [re.compile(p, re.I) for p in [
    rf"\b(?:what|which) (?:disease|condition|illness|diagnosis) do(?:es)? {_PERSON} have\b",
    rf"\b(?:diagnose|what(?:'s| is) wrong with) (?:me|my|him|her|(?:this|my|the) {_RELATIVE})\b",
    rf"\b(?:should|can|could|may|must|do|does) {_PERSON}\b.{{0,25}}\b(?:take|stop|start|give|increase|decrease|double|skip|switch|keep taking|prescribe|go to|see a|need)\b",
    rf"\b(?:what|which|how much|how many)\b.{{0,40}}\b(?:should|can|do) {_PERSON}\b.{{0,20}}\b(?:take|give|use|do|prescribe)\b",
    rf"\bis (?:it|this) (?:safe|ok|okay) for (?:me|him|her|(?:my|this|the) {_RELATIVE})\b",
    rf"\bdo(?:es)? (?:i|we|he|she|you think i|(?:my|this|the|our) {_RELATIVE}) have\b",
    r"\bdiagnose\b.{0,30}\b(?:my|his|her)\b",
]]

# Prototypes for the semantic gate (questions asking for a decision about one specific person).
PATIENT_PROTOTYPES = [
    "I have had chest pain since this morning, what disease do I have?",
    "Can you diagnose the rash on my arm?",
    "What dose of antibiotics should I take for my sore throat?",
    "My child has a fever of 40C, what should I do?",
    "Should I stop taking my blood pressure medication?",
    "My patient is a 54 year old man with these lab results, what should I prescribe him?",
    "Do I have cancer?",
    "Is it safe for me to take ibuprofen with my other pills?",
]


def rule_gate(question):
    """Cheap checks that need no model. Returns (decision, message)."""
    q = (question or "").strip()
    if len(q) < 3 or len(q) > config.MAX_QUESTION_CHARS or not re.search(r"[A-Za-z]{2}", q):
        return INVALID, INVALID_MSG.format(n=config.MAX_QUESTION_CHARS)
    if any(p.search(q) for p in _EMERGENCY):
        return EMERGENCY, EMERGENCY_MSG
    if any(p.search(q) for p in _PATIENT):
        return PATIENT, PATIENT_MSG
    return OK, None


def patient_similarity(plain_vec, prototype_vecs):
    """Highest cosine similarity between the question and the patient-specific prototypes."""
    return float(np.max(prototype_vecs @ plain_vec))


def semantic_gate(plain_vec, prototype_vecs):
    """Embedding check for patient-specific requests the rules missed. Returns (decision, message, score)."""
    s = patient_similarity(plain_vec, prototype_vecs)
    if s >= config.PATIENT_MIN:
        return PATIENT, PATIENT_MSG, s
    return OK, None, s
