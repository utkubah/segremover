# SegRemover

Weakly-supervised, document-aware segment ranker for YouTube transcripts. A three-head hierarchical Transformer (RoBERTa-base + inter-segment Transformer) outputs a calibrated removal probability `p_remove ∈ [0,1]` per topic segment — no manual training labels required.

**Paper:** [SegRemover (ACL 2025)](https://github.com/utkubah/segremover)  
**Key results:** ROC-AUC 0.871 (95% CI 0.868–0.874), ECE 0.057, gold accuracy 0.773 vs weak-label baseline 0.750 (n = 1,989 segments).

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Reproduce

### 1. Collect transcripts

```bash
python src/scrape.py --sources data/source_registry.csv
```

### 2. Segment

```bash
python src/segment.py
```

### 3. Generate weak labels

```bash
# Core 6 LFs
python src/weak_labels.py

# Optional 7th LF (summary alignment — downloads ~1.6 GB BART on first run)
python src/summary_align.py
python src/weak_labels.py   # re-run to fold in LF 7
```

### 4. Auxiliary labels (Heads B and C)

```bash
python src/function_labels.py       # 6-class function (Head B)
python src/disfluency_labels.py     # 5-class disfluency (Head C)
```

### 5. Baseline scores

```bash
python src/baselines.py
```

### 6. Train

```bash
python src/train.py
# Saves models/best.pt and models/config.json (includes temperature T)
```

### 7. Evaluate

```bash
# Smoke test (CPU, no model — checks the pipeline)
python src/eval.py --max-docs 5 --no-model

# Full paper eval (GPU required)
python src/eval.py --checkpoint models/best.pt --paper-mode \
    --eval-sample 500 --transcript-sample 10 \
    --ablate-cascade --ablate-no-doc models/best_no_doc.pt \
    --pareto
```

Outputs go to `results/`.

---

## HPC (Bocconi SLURM — `stud` partition)

```bash
sbatch hpc/run_scrape.sh
sbatch hpc/run_labels.sh
sbatch hpc/run_train.sh
sbatch hpc/run_eval.sh          # full paper eval, ~16h, 48 GB RAM, 1 GPU
```

Pull results locally:

```bash
scp -r username@hpc.host:/path/to/project/results/ ./results/
```

---

## Repository structure

```
src/
  scrape.py               # yt-dlp scraper, 6 genres, checkpoint resume
  segment.py              # spaCy sentences + SBERT topic boundaries
  weak_labels.py          # 7 LFs → Snorkel LabelModel → p_remove per segment
  summary_align.py        # LF 7: BART summary alignment
  function_labels.py      # Head B labels (6-class function)
  disfluency_labels.py    # Head C labels (5-class disfluency)
  baselines.py            # Random / Heuristic / TF-IDF / SBERT
  train.py                # three-head BERTSum, temperature calibration
  eval.py                 # full evaluation suite
hpc/
  run_*.sh                # SLURM job scripts
data/
  source_registry.csv     # 6 genres, 50+ channel sources
  processed/              # generated: segments, labels, baselines
  gold/videos/            # manual annotations (not in repo)
models/
  best.pt                 # trained checkpoint (not in repo — too large)
  config.json             # encoder, temperature T, hyperparameters
results/
  plots/                  # all paper figures (PNG)
  segment_metrics.csv     # AUC-ROC, PR-AUC, ECE, F1
  gold_results.csv        # Spearman ρ, NDCG@20, gold accuracy
  transcript_eval.csv     # compression vs SBERT cosine per threshold
  ablation_table.csv      # cascade / inter-seg ablation
  evaluation_report.md    # full eval report
  qualitative_examples.md # per-genre FP/FN examples
notebooks/
  eda.ipynb               # corpus analysis, weak label statistics
```

---

## Model

RoBERTa-base encodes each segment independently (Stage 1). A 4-layer inter-segment Transformer with 8 attention heads lets each segment attend over the full document (Stage 2). Three heads share the encoder:

- **Head A** — removability, `sigmoid → p_remove` (primary)
- **Head B** — function class (6-class: new\_information, clarification, useful\_repetition, redundant\_repetition, discourse\_filler, off\_topic)
- **Head C** — disfluency (5-class: clean, filled\_pause, repetition, revision, restart)

Head A receives `[h ‖ logit_B ‖ logit_C]` (cascade input). Loss: `BCE_A + 0.5·CE_B + 0.5·CE_C`. Temperature scaling (T = 1.1) applied on dev set.

---

## Weak labels

Seven labeling functions aggregated via Snorkel `LabelModel`:

| LF | Signal | Vote |
|---|---|---|
| `LF_filler` | filler word density > 3% | REMOVE |
| `LF_repetition` | max SBERT cosine to earlier segment > 0.70 | REMOVE |
| `LF_short` | < 5 content words | REMOVE |
| `LF_low_tfidf` | bottom-20% TF-IDF in document | REMOVE |
| `LF_named_entity` | ≥ 3 spaCy named entities | KEEP |
| `LF_question_or_def` | contains `?` or definition phrase | KEEP |
| `LF_summary_align` | cosine alignment to local BART summary | REMOVE / KEEP |

---

## Gold annotation format

Manual annotations stored in `data/gold/videos/{video_id}.json`:

```json
{
  "video_id": "abc123",
  "segments": [
    {
      "seg_idx": 0,
      "gold_removability": "probably_keep",
      "gold_function": "new_information",
      "gold_disfluency": "clean"
    }
  ]
}
```

Removability maps: `definitely_remove → 1.0`, `probably_remove → 0.75`, `unsure → 0.5`, `probably_keep → 0.25`, `definitely_keep → 0.0`.

---

## Limitations

- Transcript-only: visual/deictic references ("this graph") are invisible to the model.
- English-only.
- Single annotator; no inter-annotator κ reported.
- Weak-label ceiling: LF noise propagates to training signal.
