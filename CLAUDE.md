# MedBot - databot medico (Qwen3 finetunato + RAG + classificatore pelle)

## Obiettivo
Enciclopedia medica in **inglese**, usata a scopo educativo (NON diagnostico), costruita solo con fonti pubbliche e **gratuite**.
Componenti: RAG su fonti mediche + Qwen3-4B finetunato (stile/citazioni/rifiuti) + classificatore di immagini per lesioni cutanee.

## Vincoli
- **Tutto gratis**: niente API a pagamento (no Anthropic API per il dataset). Training su Colab/Kaggle gratuiti.
- Il PC ha solo Intel Iris Xe (no CUDA): finetuning dell'LLM solo su Colab; il resto (chunking, indice, bot, training classificatore a bassa risoluzione) gira in locale su CPU.
- Il bot risponde SOLO dal contesto recuperato, cita [n], dice "non ho abbastanza informazioni" se manca, rifiuta diagnosi personali.
- Dati immagini: DermaMNIST/HAM10000 e' CC BY-NC 4.0 (solo uso non commerciale). Controllare la licenza di ogni nuova fonte.

## Struttura
`scripts/` (01-04, 06, 07), `notebooks/` (08), `data/` (dataset, anche dermamnist_224.npz scaricato da 06), `models/` (skin_model.pt, skin_metrics.json).
Ogni script si lancia dalla root medbot/: `python scripts/NN_nome.py`.

## Pipeline
1. `01_download_sources.py medlineplus|statpearls` -> `data/raw/` (StatPearls ~2 GB; libri open: PDF in `data/raw/textbooks/`)
2. `02_chunk.py [--limit-statpearls N]` -> `data/chunks.jsonl`
3. `03_build_index.py` -> `data/index_emb.npy`, `data/index_meta.jsonl` (bge-small-en-v1.5)
4. `04_bot.py [--model qwen3:4b|medbot]` -> chat CLI con Ollama; `/img percorso.jpg` classifica una foto
5. (rimosso: dataset a pagamento; il dataset si genera nel notebook 08)
6. `06_train_skin.py --size 128 --epochs 6` -> `skin_model.pt`, `skin_metrics.json` (CPU ~30 min)
7. `07_predict_skin.py foto.jpg` -> top-3 classi, soglia 50% altrimenti "non riconosciuto"
8. `08_finetune_colab.ipynb` -> su Colab T4: genera dataset da `chunks.jsonl`, fine-tuning Unsloth QLoRA, export GGUF.
   Salva tutto su Google Drive (`MyDrive/medbot`) e riprende da solo se Colab si disconnette.
   Il notebook e' GENERATO da uno script (non modificare l'.ipynb a mano senza ricordarsene).

## Dopo il finetuning
Mettere il `.gguf` in una cartella con un `Modelfile` (`FROM ./file.gguf`), poi `ollama create medbot -f Modelfile`, poi `python 04_bot.py --model medbot`.

## Stato (aggiornare)
- Fatto: scarico+chunk MedlinePlus (1635 chunk), tutti gli script scritti e con sintassi controllata, notebook 08 con resume su Drive.
- Non ancora provato: indice+bot end-to-end, notebook su Colab, qualita' del dataset auto-generato, StatPearls, libri di testo.
- Classificatore pelle addestrato (128 px, 6 epoche): test acc 84.6%, balanced acc 77.5%, recall melanoma 64% -> solo educativo. Notebook 08 in esecuzione su Colab; poi GGUF -> Ollama.

## Convenzioni
- Python 3.13 su Windows; script numerati, eseguiti dalla root medbot/. `data/` e `*.pt`, `*.gguf` non vanno condivisi (grandi/licenze).
- Risposte all'utente in italiano; contenuti del bot e dei dati in inglese.
