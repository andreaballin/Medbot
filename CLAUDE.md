# MedBot - medical reference assistant (staged RAG + Qwen3 + skin classifier)

## Goal
English-language medical reference for **study and consultation by clinicians and medical students** (NOT diagnostic, not a medical device), built solely from free public sources.
Components: staged RAG pipeline served as API + web UI, Qwen3-4B (optionally fine-tuned) via Ollama, skin lesion image classifier.

## Constraints
- **Everything free**: no paid APIs. Training on free Colab/Kaggle.
- The dev PC only has an Intel Iris Xe (no CUDA): LLM fine-tuning runs on Colab only; everything else runs locally on CPU (generation ~9 tokens/s).
- The bot answers ONLY from retrieved context, cites [n], says it lacks information when context is missing.
- **Clinical gate profile**: general dosing / criteria / management questions pass; decisions about one specific person are refused; first-person emergencies get a fixed message.
- Image data (DermaMNIST/HAM10000) is CC BY-NC 4.0 (non-commercial only). Check the license of every new source.

## Layout
- `app/`: `config.py` (settings, `MEDBOT_*` env overrides), `retriever.py`, `gate.py`, `generator.py`, `pipeline.py`, `skin.py`, `api.py` (FastAPI), `ui.py` (Gradio at `/ui`).
- `scripts/`: data preparation and CLI (01-04, 06, 07, 09). Run from the repo root: `python scripts/NN_name.py`.
- `notebooks/08_finetune_colab.ipynb`, `eval/golden.jsonl`, `tests/`, `data/` and `logs/` (gitignored), `models/` (`*.pt` gitignored).

## Exam mode (app/exam.py) - the main product direction
A professor of medicine that examines the student, **in Italian**. `scripts/10_build_question_bank.py` builds
`data/question_bank.jsonl` offline (one Italian question + model answer + key points per English chunk, ~70 s each on CPU).
At exam time the LLM only grades one answer against one passage (JSON verdict: corretta / parziale / sbagliata).
Progress in `data/progress.json`; wrong answers are re-asked more often. UI tab "Esame", API under `/exam/*`.
Planned next: voice in/out (Whisper + browser speech synthesis), clinical sources (StatPearls), Italian free questions
(needs a multilingual embedding model and a re-index).

## Pipeline stages (app/pipeline.py)
rule gate + question embedding in parallel -> semantic gate -> retrieval + scope threshold -> generation -> citation check.
Decisions: `answered`, `out_of_scope`, `patient_specific`, `emergency`, `invalid`. Only `answered` calls the LLM.

## Commands
- Build data: `01_download_sources.py medlineplus|statpearls` -> `02_chunk.py` -> `03_build_index.py` (writes `data/index_manifest.json`).
- Serve: `uvicorn app.api:app` -> `/ask`, `/classify`, `/health`, `/ui`, `/docs`. CLI: `python scripts/04_bot.py`.
- Check: `pytest` (no model needed); `python scripts/09_eval.py [--verbose]` to evaluate the gate and calibrate `SCOPE_MIN` / `PATIENT_MIN`.
- Skin: `06_train_skin.py --size 128 --epochs 6`, `07_predict_skin.py photo.jpg`.
- Fine-tuned model: put the `.gguf` next to a `Modelfile` (`FROM ./file.gguf`), `ollama create medbot -f Modelfile`, then `MEDBOT_MODEL=medbot`.

## Gotchas
- Ollama's `qwen3:4b` is a thinking-only build: it ignores `think: false` and leaks reasoning into the answer. Use `qwen3:4b-instruct`. `generator.strip_reasoning` is the fallback.
- Embedding similarity separates in-scope from out-of-scope well, but NOT patient-specific from general questions (overlap around 0.75-0.82). The rules do the real work there; the semantic gate only catches near-paraphrases of the prototypes.
- The notebook 08 dataset teaches the old "public" refusal (refuse all personal/dosing questions); regenerate it if the fine-tuned model must match the clinical profile.

## Status (keep updated)
- Done: MedlinePlus index (1635 chunks), staged pipeline + API + UI, 42 tests passing, gate 48/50 on `eval/golden.jsonl`, skin classifier (test acc 84.6%, balanced acc 77.5%, melanoma recall 64%).
- Running: notebook 08 on Colab (fine-tuned GGUF not yet tested).
- Next (phase 2): StatPearls + drug labels, section-aware chunking with URL/date, hybrid retrieval + reranker.
- Later (phase 3): sentence-level answer verification, study mode (quiz/flashcards), PII masking, feedback button.

## Conventions
- Python 3.13 on Windows. Code, comments and docs in English. What the student sees in exam mode (UI text, questions, feedback) is in Italian.
