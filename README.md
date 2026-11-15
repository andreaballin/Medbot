# MedBot

An English-language medical reference chatbot for **educational purposes only** — it is not a diagnostic tool and does not give personal medical advice.

It combines three free components:

1. **RAG** over public medical sources (MedlinePlus, optionally StatPearls and open textbooks), with a `bge-small-en-v1.5` vector index stored as plain numpy files.
2. **Qwen3-4B fine-tuned** (Unsloth QLoRA, on free Colab) to answer only from the retrieved context, cite sources like `[1]`, say when information is missing, and refuse personal diagnoses. Served locally through [Ollama](https://ollama.com).
3. **Skin lesion classifier** (EfficientNet-B0 on DermaMNIST, 7 classes) whose top prediction is explained through the RAG pipeline.

> **Disclaimer:** MedBot provides general educational information. It never replaces a qualified healthcare professional. For emergencies call your local emergency number.

## Repository layout

| Path | Purpose |
|---|---|
| `scripts/01_download_sources.py` | Download MedlinePlus / StatPearls into `data/raw/` |
| `scripts/02_chunk.py` | Clean and split sources into `data/chunks.jsonl` |
| `scripts/03_build_index.py` | Build the embedding index |
| `scripts/04_bot.py` | Interactive CLI bot (RAG + Ollama, `/img photo.jpg` for the classifier) |
| `scripts/06_train_skin.py` | Train the skin classifier |
| `scripts/07_predict_skin.py` | Run the skin classifier on an image |
| `notebooks/08_finetune_colab.ipynb` | Generate the SFT dataset, fine-tune Qwen3-4B, export GGUF (Colab T4) |

`data/` and model weights are not tracked in git (size and licenses).

## Quick start

Run everything from the repository root.

```bash
pip install sentence-transformers numpy requests pymupdf

python scripts/01_download_sources.py medlineplus
python scripts/02_chunk.py
python scripts/03_build_index.py

ollama pull qwen3:4b
python scripts/04_bot.py            # --model medbot after fine-tuning
```

### Fine-tuning (optional, free)

1. Open `notebooks/08_finetune_colab.ipynb` in Google Colab, select a **T4 GPU** runtime.
2. Upload `data/chunks.jsonl` when prompted. The notebook generates the dataset with Qwen3 itself, trains a LoRA adapter and exports a GGUF. Progress is saved to Google Drive and resumes after disconnects.
3. Put the `.gguf` next to a `Modelfile` containing `FROM ./your-file.gguf`, then:

```bash
ollama create medbot -f Modelfile
python scripts/04_bot.py --model medbot
```

### Skin classifier (optional)

```bash
pip install torch torchvision pillow
python scripts/06_train_skin.py --size 128 --epochs 6   # CPU-friendly, downloads DermaMNIST (~1 GB)
python scripts/07_predict_skin.py photo.jpg
```

Results with the default CPU run (test set): accuracy 84.6%, balanced accuracy 77.5%. Melanoma recall is only 64% — the classifier is a demo and **must not** be used to assess real lesions.

## Licenses and data

- MedlinePlus: public domain. StatPearls: CC BY-NC-ND. DermaMNIST / HAM10000: CC BY-NC 4.0 (non-commercial use only).
- Check the license of any source you add before redistributing data or models.
