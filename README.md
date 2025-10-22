# Underlying Property Matching (UPM) — Long‑Text Entity Matching

This repository contains the code accompanying the paper/thesis on **Underlying Property Matching (UPM)** for long textual data (scientific articles and news). The core idea is to model *matching by an unobserved underlying property* (e.g., scientific field, news topic) and to learn **representations and assignments jointly** via an iterative **paragraph‑selection** algorithm and lightweight aggregation, with optional transformer‑based aggregation and contrastive fine‑tuning.

> **Main entry point:** `res_run_through.ipynb` (end‑to‑end walkthrough of the pipeline and experiments).

---

## Table of Contents
- [Concept](#concept)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Embedding & Models](#embedding--models)
- [Paragraph Selection & Centroids](#paragraph-selection--centroids)
- [Training (Optional) — Transformer Aggregation](#training-optional--transformer-aggregation)
- [Evaluation](#evaluation)
- [Reproducing Results (Recommended Order)](#reproducing-results-recommended-order)
- [Configuration Notes & Placeholders](#configuration-notes--placeholders)
- [Troubleshooting](#troubleshooting)
- [Citing](#citing)
- [License](#license)

---

## Concept

**Underlying Property Matching (UPM)** defines a matching criterion where two items match iff they share the same **underlying property**. Because long documents contain heterogeneous paragraphs (some more indicative than others), the pipeline:
1. Learns/uses initial embeddings.
2. Infers **centroids** for each property from training labels (or via transitivity/cluster assignments).
3. **Selects** per‑item the paragraph **closest** to its predicted centroid, iteratively recomputing centroids until convergence.
4. Uses the selected paragraphs as compact, strong representations for matching and retrieval tasks.

---

## Repository Structure

```
.
├─ res_run_through.ipynb              # End-to-end walkthrough (main file)
├─ matching_metrics.py                # Macro-F1 (per property) for pairwise predictions
├─ data_refining_pipeline.py          # Centroids, assignments, paragraph-selection alg., voting
├─ loading_paper_utils.py             # I/O, HTML parsing glue, embedding helpers
├─ html_parsing_utils.py              # Robust arXiv (ar5iv) HTML parsing utilities
├─ extracting_papers_html.py          # arXiv ID lookup + ar5iv HTML downloader
├─ news_data_scraping.py              # Example: scrape paragraphs from news links
├─ dataloadrer_and_prep.py            # Pair generation, splits for contrastive finetuning
├─ finetune_hf_model_for_reps.py      # Sentence-Transformers contrastive finetuning
├─ transformer_agg.py                 # Transformer-based paragraph aggregation (optional)
├─ generating_summary.py              # (Optional) LLM summaries for a summary baseline
└─ (notebooks/*.ipynb)                # Additional exploration / experiments
```

**Highlights**:
- `data_refining_pipeline.py`: contains both **train** and **inference** versions of the **iterative paragraph‑selection** algorithm and voting utilities for k‑paragraph selection.
- `transformer_agg.py`: builds training pairs from per‑paragraph embeddings, defines a small Transformer encoder that aggregates sequences of paragraph vectors, and trains with cosine embedding loss.
- `finetune_hf_model_for_reps.py`: SBERT‑style contrastive finetuning using `SentenceTransformerTrainer`.
- `loading_paper_utils.py` & `html_parsing_utils.py`: end‑to‑end arXiv HTML ➜ paragraphs/sections ➜ embeddings.
- `matching_metrics.py`: macro‑F1 computed *per underlying property* and averaged.

---

## Setup

Tested with Python 3.10+ and CUDA‑enabled PyTorch.

```bash
# (recommended) create a fresh environment
conda create -n upm_env python=3.10 -y
conda activate upm_env

# core deps
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ML / NLP
pip install sentence-transformers transformers datasets pytorch-metric-learning scikit-learn faiss-cpu

# data & parsing
pip install pandas numpy beautifulsoup4 requests tqdm regex

# optional
pip install arxiv openai matplotlib
```

> Set `CUDA_VISIBLE_DEVICES` if needed (several scripts include a local default).

---

## Data Preparation

### A) Scientific Articles (arXiv via ar5iv)
1. **Collect HTML** (per category directory):
   - Use `extracting_papers_html.py` to find arXiv IDs by title and download ar5iv HTML pages to `HTML_ROOT/<category>/*.html`.
2. **Build a DataFrame**:
   - Use `loading_paper_utils.py`:
     - `load_data_from_html_dir(HTML_ROOT)` → DataFrame with `id, title, abstract, category`.
     - `add_path_to_html(df, HTML_ROOT)` to create `html_path` per row.
     - `add_paragraphs(df)` to extract `pars` (and `sections`); filters empty rows.
3. **Save** the resulting DataFrame to `*.pkl` for downstream steps.

### B) News Articles
- If using the HuffPost News Category dataset (or similar with links), `news_data_scraping.py` includes an example for fetching and parsing article paragraphs from URLs.
- Save a unified DataFrame with at least: `id, category, pars` (list[str]).

> Train/val/test splits per category are supported in `dataloadrer_and_prep.py`.

---

## Embedding & Models

You can embed **titles/abstracts**, **concatenations**, or **per‑paragraph lists**:

- `loading_paper_utils.embed_col(...)` for single text columns.
- `loading_paper_utils.embed_text_list_col(...)` for `pars` (list[str]) to produce per‑paragraph embeddings.
- `loading_paper_utils.embed_multi_col(...)` to concatenate multiple columns before embedding.
- Transformer‑tokenizer variants (`*_transformer`) are available for custom HF backbones.
- **Normalization** flags are available where appropriate.

> Suggested base encoders: `intfloat/e5-base` / `intfloat/e5-base-unsupervised` or `sentence-transformers/all-mpnet-base-v2`.

---

## Paragraph Selection & Centroids

- **Centroid inference** (train): group by `category` and average embeddings to get class centroids.
- **Assignments** (inference): assign each item to the **closest centroid** by cosine similarity.
- **Iterative paragraph selection**:
  1. For each item, compute similarity of its paragraph embeddings to the assigned centroid.
  2. Pick the **closest paragraph** as the current representation.
  3. Recompute centroids from chosen paragraphs.
  4. Stop on **no change** in centroids or chosen paragraphs, or on iteration cap.
- Includes **debug outputs** (e.g., silhouette / CH scores) and **consistency checks** (e.g., whether selected paragraphs “vote” for their own centroid).

---

## Training (Optional) — Transformer Aggregation

`transformer_agg.py` implements a compact Transformer encoder that consumes a sequence of paragraph vectors and outputs an aggregated representation.

Workflow:
1. Produce per‑paragraph embeddings for all items.
2. Run `text_list_embedded_df_to_data_pairs(...)` to create **positive/negative** pairs with padding masks.
3. Train `transformer_agg_net(d_model=768, nhead=12, n_transformer_layers=2, dropout=0.1)` using cosine embedding loss.
4. Save the model and use it to encode documents from paragraph sequences.

> This is optional; the paper’s core results can be reproduced with selection + mean pooling baselines.

---

## Evaluation

- **Pairwise macro‑F1 (per property)**: `matching_metrics.get_macro_f1(...)` on a cross‑join of predicted vs. true property agreements.
- **Clustering quality**: silhouette / Calinski‑Harabasz on selected paragraphs or means.
- **Downstream retrieval**: evaluate `precision@k` with nearest‑neighbors over representations.

---

## Reproducing Results (Recommended Order)

1. **(Notebook) Run the end‑to‑end demo**: `res_run_through.ipynb`  
   - Shows the pipeline on your prepared data.
2. **Prepare data** (if starting from raw):
   - arXiv HTML ➜ DataFrame ➜ paragraphs (`loading_paper_utils.py`, `html_parsing_utils.py`).
   - (Optional) news scraping (`news_data_scraping.py`).
3. **Embeddings**:
   - Encode `title`, `abstract`, and `pars` with SBERT/HF models.
   - (Optional) contrastive **finetune** with `finetune_hf_model_for_reps.py` using pairs from `dataloadrer_and_prep.py`.
4. **Centroids & paragraph selection**:
   - Compute train centroids; run **train/inference** selection loops in `data_refining_pipeline.py`.
5. **Evaluation**:
   - Macro‑F1, silhouette/CH, and retrieval metrics.
6. **(Optional)** Train and compare **Transformer aggregation** (`transformer_agg.py`).

---

## Configuration Notes & Placeholders

Several scripts include **placeholders** you’ll need to set before running:
- Paths like `PATH/TO/...`, `YOUR/TEST/SET/PATH.pkl`, output directories, and checkpoints.
- Model names (e.g., `'intfloat/e5-base'`, `'sentence-transformers/all-mpnet-base-v2'`).
- GPU visibility via `CUDA_VISIBLE_DEVICES` (or remove the line if unnecessary).
- OpenAI API key for `generating_summary.py` (optional baseline).

> Many functions are written as utilities; call them from `res_run_through.ipynb` or a driver script in the order shown above.

---

## Troubleshooting

- **CUDA OOM**: lower batch sizes; ensure paragraph batches are chunked (see `*_transformer` helpers).
- **Empty/short documents after parsing**: confirm `ar5iv` HTML parsing; tweak cleaning thresholds in `html_parsing_utils.py`.
- **Class imbalance**: sampling is per‑category where relevant; adjust negative:positive ratio in pair generation.
- **Convergence**: the iterative selection loop stops if centroids or chosen paragraphs stop changing (cap iterations otherwise).

---

## Citing

If you use this code or ideas from the paper/thesis, please cite the repository and manuscript (BibTeX TBD by the author).

---

## License

[MIT](LICENSE) unless specified otherwise by included datasets’ terms.

---

### Acknowledgments

- arXiv’s **ar5iv** project for HTML rendering.
- Sentence‑Transformers and Hugging Face for embedding backbones and training utilities.
