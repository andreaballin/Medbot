"""Gradio chat UI, mounted on the FastAPI app at /ui."""
import gradio as gr

from . import generator, skin
from .pipeline import ANSWERED

DISCLAIMER = ("**Reference and study tool only.** MedBot answers from its indexed sources and can be wrong. "
              "It does not make decisions about individual patients and is not a medical device.")
_SOURCES_MARK = "\n\n---\n**Sources**"
_LABELS = {"out_of_scope": "Not covered by the indexed sources", "patient_specific": "Patient-specific request",
           "emergency": "Possible emergency", "invalid": "Invalid question"}


def _text(content):
    """Plain text of a chat message, without the sources footer added by format_result."""
    if not isinstance(content, str):
        content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return content.split(_SOURCES_MARK)[0]


def format_result(res, final=True):
    if res.decision != ANSWERED:
        return f"**{_LABELS.get(res.decision, res.decision)}.** {res.answer}"
    out = res.answer
    if final:
        out += _SOURCES_MARK + "\n" + "\n".join(
            f"- [{s['n']}] {s['source']}: {s['title']} (similarity {s['score']:.2f})" for s in res.sources)
        if not res.citations.get("cited"):
            out += "\n\n*Warning: this answer does not cite any passage; check it against the sources.*"
        if res.citations.get("invalid"):
            out += f"\n\n*Warning: the answer cites passages that do not exist: {res.citations['invalid']}.*"
    return out


def _stream(pipeline, question, history=(), prefix=""):
    try:
        last = None
        for res in pipeline.answer_stream(question, history):
            last = res
            yield prefix + format_result(res, final=False)
        if last is not None:
            yield prefix + format_result(last, final=True)
    except generator.LLMUnavailable as e:
        yield prefix + f"**The language model is not available.** {e}"


ALL_TOPICS = "Tutti gli argomenti"
_VERDICT = {"corretta": "Corretta (2/2)", "parziale": "Parziale (1/2)", "sbagliata": "Sbagliata (0/2)"}


def format_grade(g):
    """Italian feedback shown to the student after an answer."""
    out = [f"### {_VERDICT.get(g['verdict'], 'Non valutata')}", g["feedback"]]
    if g["missing"]:
        out.append("**Cosa mancava:**\n" + "\n".join(f"- {m}" for m in g["missing"]))
    if g["errors"]:
        out.append("**Errori:**\n" + "\n".join(f"- {m}" for m in g["errors"]))
    out.append(f"**Risposta attesa:** {g['reference']}")
    out.append(f"*Fonte: {g['source']}*")
    return "\n\n".join(out)


def format_progress(s):
    if not s["questions"]:
        return ("La banca domande è vuota. Generala con "
                "`python scripts/10_build_question_bank.py --n 20`.")
    head = f"**Domande provate:** {s['tried']} su {s['questions']}"
    if s["percent"] is not None:
        head += f" — **punteggio:** {s['percent']}%"
    rows = [f"| {t['topic']} | {t['tried']}/{t['questions']} | {t['percent']}% |"
            for t in s["topics"] if t["tried"]]
    if rows:
        head += "\n\n| Argomento | Provate | Punteggio |\n|---|---|---|\n" + "\n".join(rows)
    return head


def mount_ui(app, get_pipeline, get_exam, path="/ui"):
    def topic_choices():
        return [ALL_TOPICS] + [t for t, _ in get_exam().topics()]

    def new_question(topic, current):
        q = get_exam().next_question(None if topic in (None, ALL_TOPICS) else topic, exclude=current)
        if q is None:
            return None, "Nessuna domanda disponibile per questo argomento.", "", "", format_progress(get_exam().summary())
        return q["id"], f"**{q['topic']}**\n\n### {q['question']}", "", "", format_progress(get_exam().summary())

    def submit(qid, answer):
        if not qid:
            return "Premi prima **Nuova domanda**.", format_progress(get_exam().summary())
        try:
            g = get_exam().grade(qid, answer)
        except generator.LLMUnavailable as e:
            return f"**Il modello non è disponibile.** {e}", format_progress(get_exam().summary())
        except KeyError:
            return "Questa domanda non è più nella banca. Premi **Nuova domanda**.", format_progress(get_exam().summary())
        return format_grade(g), format_progress(get_exam().summary())

    def chat(message, history):
        hist = [{"role": m["role"], "content": _text(m["content"])} for m in history][-4:]
        yield from _stream(get_pipeline(), message, hist)

    def classify(image):
        if image is None:
            yield "Upload a photo first."
            return
        classes, recognised = skin.predict(image)
        head = "**Image classifier demo, NOT a diagnosis** (melanoma recall on the test set is only 64%).\n\n" + \
               "\n".join(f"- {p:.1%} {c}" for c, p in classes)
        if not recognised:
            yield head + "\n\nConfidence too low: the image is not recognised as one of the supported lesion types."
            return
        head += f"\n\n**About the top class ({classes[0][0]}):**\n\n"
        yield head
        yield from _stream(get_pipeline(), skin.explain_question(classes[0][0]), prefix=head)

    with gr.Blocks(title="MedBot") as demo:
        gr.Markdown("# MedBot\n" + DISCLAIMER)
        with gr.Tab("Esame"):
            gr.Markdown("Il professore ti interroga: scegli un argomento, rispondi con parole tue e ricevi la "
                        "correzione. Le domande sbagliate tornano più spesso. La correzione richiede circa un minuto.")
            qid = gr.State(None)
            with gr.Row():
                topic = gr.Dropdown(choices=[ALL_TOPICS], value=ALL_TOPICS, label="Argomento", scale=3)
                next_btn = gr.Button("Nuova domanda", variant="primary", scale=1)
            question = gr.Markdown("Premi **Nuova domanda** per cominciare.")
            answer = gr.Textbox(label="La tua risposta", lines=4)
            submit_btn = gr.Button("Consegna")
            feedback = gr.Markdown()
            progress = gr.Markdown()
            next_btn.click(new_question, [topic, qid], [qid, question, answer, feedback, progress])
            submit_btn.click(submit, [qid, answer], [feedback, progress])
            demo.load(lambda: (gr.Dropdown(choices=topic_choices(), value=ALL_TOPICS),
                               format_progress(get_exam().summary())), None, [topic, progress])
        with gr.Tab("Ask"):
            gr.ChatInterface(chat, examples=[
                "What are the warning signs of a stroke?",
                "How is pneumonia diagnosed?",
                "What is the difference between Crohn's disease and ulcerative colitis?"])
        with gr.Tab("Skin image"):
            img = gr.Image(type="pil", label="Skin lesion photo")
            btn = gr.Button("Classify and explain")
            out = gr.Markdown()
            btn.click(classify, img, out)
    return gr.mount_gradio_app(app, demo, path=path)
