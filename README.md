# MedBot

An English-language medical reference assistant for **study and consultation** by clinicians and medical students. It answers only from its indexed sources and cites them. It is not a diagnostic tool, does not make decisions about individual patients, and is not a medical device.

It combines three free components:

1. **RAG** over public medical sources (MedlinePlus, optionally StatPearls and open textbooks), with a `bge-small-en-v1.5` vector index stored as plain numpy files.
2. **A local LLM** served through [Ollama](https://ollama.com): Qwen3-4B, optionally fine-tuned (Unsloth QLoRA, on free Colab) to answer only from the retrieved context and cite sources like `[1]`.
3. **A skin lesion classifier** (EfficientNet-B0 on DermaMNIST, 7 classes) whose top prediction is explained through the same pipeline.

## Exam mode (Italian)

The main use is a professor that examines you. The interface and the professor speak **Italian**; the sources stay in English.

1. Build a question bank once: each question is generated from one source passage and stored with a model answer and key points.
2. In the **Esame** tab pick a topic and press *Nuova domanda*. Answer in your own words.
3. The professor grades the answer against the passage (`corretta`, `parziale`, `sbagliata`), explains what was missing or wrong, and shows the model answer and the source.
4. Progress is saved in `data/progress.json`; questions answered badly come back more often.

```bash
python scripts/10_build_question_bank.py --n 20                    # about one minute per question on CPU, resumable
python scripts/10_build_question_bank.py --n 10 --topic "Diabetes" # only topics containing the text
```

API: `GET /exam/topics`, `GET /exam/question?topic=`, `POST /exam/grade`, `GET /exam/progress`.

Grading is done by a small local model: treat the verdict as a study aid, and compare with the model answer shown below it.

## How a free question is processed

The **Ask** tab (English only for now) answers free questions from the sources:

```
question -+-> [1a] rule gate            (in parallel)
          +-> [1b] question embedding
          [2] semantic gate   -> fixed message for patient-specific requests
          [3] retrieval       -> fixed message when no passage is similar enough (out of scope)
          [4] generation      -> local LLM, answers only from the retrieved passages
          [5] citation check  -> answer + sources + per-stage timings
```

Every response carries a `decision`:

| Decision | Meaning | LLM called |
|---|---|---|
| `answered` | Answer generated from the passages, with sources | yes |
| `out_of_scope` | Best passage below the similarity threshold | no |
| `patient_specific` | Asks for a decision about one specific person | no |
| `emergency` | First-person emergency; fixed "call emergency services" message | no |
| `invalid` | Empty, too long or not text | no |

The gate uses a **clinical profile**: general questions about dosing, diagnostic criteria or management pass; only decisions about a specific person are refused.

## Repository layout

| Path | Purpose |
|---|---|
| `app/pipeline.py` | The staged pipeline (`Pipeline.answer`, `Pipeline.answer_stream`) |
| `app/gate.py` | Rule gate and semantic gate |
| `app/retriever.py`, `app/generator.py` | Dense retrieval; Ollama generation and citation check |
| `app/api.py`, `app/ui.py` | FastAPI service and Gradio chat UI |
| `app/exam.py` | Exam mode: question bank, grading, progress |
| `scripts/10_build_question_bank.py` | Generate the Italian question bank |
| `app/skin.py` | Skin classifier inference |
| `app/config.py` | All settings, overridable with `MEDBOT_*` environment variables |
| `scripts/01_download_sources.py` | Download MedlinePlus / StatPearls into `data/raw/` |
| `scripts/02_chunk.py` | Clean and split sources into `data/chunks.jsonl` |
| `scripts/03_build_index.py` | Build the embedding index and its manifest |
| `scripts/04_bot.py` | Terminal client for the pipeline |
| `scripts/06_train_skin.py`, `scripts/07_predict_skin.py` | Train / run the skin classifier |
| `scripts/09_eval.py`, `eval/golden.jsonl` | Gate evaluation and threshold calibration |
| `notebooks/08_finetune_colab.ipynb` | Generate the SFT dataset, fine-tune Qwen3-4B, export GGUF (Colab T4) |
| `tests/` | pytest suite for the gate and the pipeline decisions |

`data/`, `logs/` and model weights are not tracked in git (size and licenses).

## Quick start

Run everything from the repository root.

```bash
pip install -r requirements.txt

python scripts/01_download_sources.py medlineplus
python scripts/02_chunk.py
python scripts/03_build_index.py

ollama pull qwen3:4b-instruct
uvicorn app.api:app
```

- Chat UI: <http://localhost:8000/ui>
- API docs: <http://localhost:8000/docs>
- Terminal client: `python scripts/04_bot.py`

```bash
curl localhost:8000/health
curl localhost:8000/ask -H "content-type: application/json" -d '{"question": "What is type 2 diabetes?"}'
curl localhost:8000/classify -F image=@photo.jpg
```

### Settings

Set `MEDBOT_<NAME>` environment variables (see `app/config.py`). The most useful ones:

| Variable | Default | Meaning |
|---|---|---|
| `MEDBOT_MODEL` | `qwen3:4b-instruct` | Ollama model (use `medbot` after fine-tuning) |
| `MEDBOT_TOP_K` | `5` | Passages given to the LLM |
| `MEDBOT_SCOPE_MIN` | `0.60` | Minimum similarity of the best passage |
| `MEDBOT_PATIENT_MIN` | `0.88` | Similarity to a patient-specific prototype question |

### Tests and evaluation

```bash
pytest                         # gate rules and pipeline decisions, no model needed
python scripts/09_eval.py      # gate accuracy on eval/golden.jsonl and suggested thresholds
```

With the MedlinePlus index and the default thresholds the gate classifies 48 of the 50 questions in `eval/golden.jsonl` correctly. The two misses are patient-specific requests phrased in ways the rules do not cover, so the gate must not be treated as complete: the system prompt is the second line of defence. Re-run the evaluation and recalibrate whenever the sources or the embedding model change.

### Fine-tuning (optional, free)

1. Open `notebooks/08_finetune_colab.ipynb` in Google Colab, select a **T4 GPU** runtime.
2. Upload `data/chunks.jsonl` when prompted. The notebook generates the dataset with Qwen3 itself, trains a LoRA adapter and exports a GGUF. Progress is saved to Google Drive and resumes after disconnects.
3. Put the `.gguf` next to a `Modelfile` containing `FROM ./your-file.gguf`, then:

```bash
ollama create medbot -f Modelfile
MEDBOT_MODEL=medbot uvicorn app.api:app
```

### Skin classifier (optional)

```bash
python scripts/06_train_skin.py --size 128 --epochs 6   # CPU-friendly, downloads DermaMNIST (~1 GB)
python scripts/07_predict_skin.py photo.jpg
```

Results with the default CPU run (test set): accuracy 84.6%, balanced accuracy 77.5%. Melanoma recall is only 64% — the classifier is a demo and **must not** be used to assess real lesions.

## Limitations

- The index currently holds patient-level material (MedlinePlus). Clinical depth needs StatPearls, drug labels and textbooks.
- A 4B-parameter model makes mistakes even with good retrieval; answers are not verified sentence by sentence against the sources.
- On a CPU-only machine an answer takes tens of seconds to minutes.
- Requests are logged to `logs/requests.jsonl` without the question text. Do not enter patient-identifying data.

## Licenses and data

- MedlinePlus: public domain. StatPearls: CC BY-NC-ND. DermaMNIST / HAM10000: CC BY-NC 4.0 (non-commercial use only).
- Check the license of any source you add before redistributing data or models.
