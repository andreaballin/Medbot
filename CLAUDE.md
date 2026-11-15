# MedBot - medical databot (fine-tuned Qwen3 + RAG + skin classifier)

## Goal
English-language medical encyclopedia for **educational use only** (NOT diagnostic), built solely from free public sources.
Components: RAG over medical sources + fine-tuned Qwen3-4B (style/citations/refusals) + skin lesion image classifier.

## Constraints
- **Everything free**: no paid APIs. Training on free Colab/Kaggle.
- The dev PC only has an Intel Iris Xe (no CUDA): LLM fine-tuning runs on Colab only; chunking, indexing, the bot and low-resolution skin training run locally on CPU.
- The bot answers ONLY from retrieved context, cites [n], says it lacks information when context is missing, and refuses personal diagnoses.
- Image data (DermaMNIST/HAM10000) is CC BY-NC 4.0 (non-commercial only). Check the license of every new source.

## Layout
`scripts/` (01-04, 06, 07), `notebooks/` (08), `data/` (gitignored: sources, chunks, index, datasets), `models/` (skin_model.pt gitignored, skin_metrics.json).
Run every script from the repo root: `python scripts/NN_name.py`.

## Pipeline
1. `01_download_sources.py medlineplus|statpearls` -> `data/raw/` (StatPearls ~2 GB; open textbooks: PDFs in `data/raw/textbooks/`)
2. `02_chunk.py [--limit-statpearls N]` -> `data/chunks.jsonl`
3. `03_build_index.py` -> `data/index_emb.npy`, `data/index_meta.jsonl` (bge-small-en-v1.5)
4. `04_bot.py [--model qwen3:4b|medbot]` -> CLI chat with Ollama; `/img path.jpg` classifies a photo
5. `06_train_skin.py --size 128 --epochs 6` -> `models/skin_model.pt`, `models/skin_metrics.json` (CPU ~30 min)
6. `07_predict_skin.py photo.jpg` -> top-3 classes, 50% threshold otherwise "not recognised"
7. `notebooks/08_finetune_colab.ipynb` -> on Colab T4: generates the dataset from `chunks.jsonl`, Unsloth QLoRA fine-tuning, GGUF export. Saves to Google Drive (`MyDrive/medbot`) and resumes after disconnects.

## After fine-tuning
Put the `.gguf` in a folder with a `Modelfile` (`FROM ./file.gguf`), then `ollama create medbot -f Modelfile`, then `python scripts/04_bot.py --model medbot`.

## Status (keep updated)
- Done: MedlinePlus download + chunking (1635 chunks), all scripts written, skin classifier trained (128 px, 6 epochs: test acc 84.6%, balanced acc 77.5%, melanoma recall 64% -> educational only).
- Running: notebook 08 on Colab.
- Not yet tried: index + bot end-to-end, quality of the auto-generated dataset, StatPearls, textbooks.

## Conventions
- Python 3.13 on Windows; numbered scripts run from the repo root. Everything (code, comments, docs, bot output) in English.
