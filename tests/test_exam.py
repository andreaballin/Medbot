"""Exam mode: question generation, grading and progress (no LLM needed).  Run: pytest"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import exam  # noqa: E402

CHUNK = {"chunk_id": "medlineplus:Asthma#0", "source": "medlineplus", "title": "Asthma",
         "text": "Asthma is a chronic disease that affects the airways in your lungs."}
GOOD = {"domanda": "Che cos'è l'asma e quali strutture colpisce?",
        "risposta": "L'asma è una malattia cronica che colpisce le vie aeree dei polmoni.",
        "punti_chiave": ["malattia cronica", "vie aeree dei polmoni"]}


def test_make_question():
    q = exam.make_question(CHUNK, llm_json=lambda m: GOOD)
    assert q["id"] == CHUNK["chunk_id"] and q["topic"] == "Asthma" and q["passage"] == CHUNK["text"]
    assert q["question"] == GOOD["domanda"] and q["key_points"] == GOOD["punti_chiave"]


@pytest.mark.parametrize("bad", [None, {}, {"domanda": "Che cos'è l'asma?"}, dict(GOOD, punti_chiave=[]),
                                 dict(GOOD, risposta="x"), dict(GOOD, punti_chiave="malattia cronica")])
def test_make_question_rejects_unusable_output(bad):
    assert exam.make_question(CHUNK, llm_json=lambda m: bad) is None


def make_exam(tmp_path, reply, n=3):
    bank = tmp_path / "bank.jsonl"
    with open(bank, "w", encoding="utf-8") as f:
        for i in range(n):
            q = exam.make_question(dict(CHUNK, chunk_id=f"t:Asthma#{i}"), llm_json=lambda m: GOOD)
            f.write(json.dumps(dict(q, topic="Asthma" if i < 2 else "Stroke")) + "\n")
    calls = []

    def llm(messages):
        calls.append(messages)
        return reply

    return exam.Exam(bank, tmp_path / "progress.json", llm_json=llm, seed=0), calls


def test_grade_records_progress_and_returns_reference(tmp_path):
    e, calls = make_exam(tmp_path, {"voto": "Parziale", "punti_mancanti": ["vie aeree dei polmoni"],
                                    "errori": [], "spiegazione": "Hai detto che è cronica, ma non cosa colpisce."})
    r = e.grade("t:Asthma#0", "È una malattia cronica")
    assert r["verdict"] == exam.PARTIAL and r["score"] == 1 and r["missing"] == ["vie aeree dei polmoni"]
    assert r["reference"] == GOOD["risposta"] and r["source"] == "medlineplus: Asthma"
    assert "Risposta dello studente: È una malattia cronica" in calls[0][-1]["content"]
    assert exam.Exam(e.bank_file, e.progress_file).progress["t:Asthma#0"]["last"] == 1   # persisted


def test_empty_answer_is_wrong_without_llm(tmp_path):
    e, calls = make_exam(tmp_path, None)
    r = e.grade("t:Asthma#0", "  ")
    assert r["verdict"] == exam.WRONG and r["score"] == 0 and not calls


@pytest.mark.parametrize("reply", [None, {"voto": "ottimo"}, {"spiegazione": "..."}])
def test_ungradable_reply_is_not_recorded(tmp_path, reply):
    e, _ = make_exam(tmp_path, reply)
    r = e.grade("t:Asthma#0", "È una malattia cronica")
    assert r["verdict"] is None and r["reference"] and "t:Asthma#0" not in e.progress


def test_topics_and_topic_filter(tmp_path):
    e, _ = make_exam(tmp_path, None)
    assert e.topics() == [("Asthma", 2), ("Stroke", 1)]
    assert all(e.next_question("Stroke")["id"] == "t:Asthma#2" for _ in range(5))
    assert e.next_question("Nothing") is None
    assert e.next_question("Asthma", exclude="t:Asthma#0")["id"] == "t:Asthma#1"


def test_wrong_answers_come_back_more_often(tmp_path):
    e, _ = make_exam(tmp_path, {"voto": "corretta", "spiegazione": "Bene."})
    e.grade("t:Asthma#0", "risposta completa")      # correct -> rare
    e.grade("t:Asthma#1", "")                        # wrong -> frequent
    picks = [e.next_question("Asthma")["id"] for _ in range(300)]
    assert picks.count("t:Asthma#1") > 4 * picks.count("t:Asthma#0")


def test_summary(tmp_path):
    e, _ = make_exam(tmp_path, {"voto": "corretta", "spiegazione": "Bene."})
    e.grade("t:Asthma#0", "risposta completa")
    e.grade("t:Asthma#1", "")
    s = e.summary()
    assert s["questions"] == 3 and s["tried"] == 2 and s["percent"] == 50
    assert s["topics"][0] == {"topic": "Asthma", "questions": 2, "tried": 2, "points": 2, "percent": 50}
    assert s["topics"][1]["percent"] is None
