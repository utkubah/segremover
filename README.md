# segremover — Measuring Information Loss in Video Transcripts

## 1. Research Question

Can a transcript-only Transformer model identify low-contribution segments in video transcripts such that removing them causes minimal loss of factual and semantic content, and does this performance vary across video length and genre?

## 2. Hypotheses

- **H1.** Longer videos contain proportionally more removable content than shorter ones.
- **H2.** Removing low-scoring segments preserves most of the transcript's core content up to a moderate compression level.
- **H3.** Performance varies by genre — filler, repetition, and useful elaboration function differently in lectures, TED talks, and commentary.
- **H4.** A transcript-only model handles obvious repetition well, but struggles where importance depends on visual context, pedagogy, or rhetorical emphasis.

---

## 3. The Task

**Input:** one transcript segment (3–25 sentences), with optional left/right context.  
**Output:** a calibrated probability `p_remove ∈ [0, 1]` — *how confident are we that this segment can be removed without losing information?*

That single number is the deliverable. Everything else in the plan exists to produce it, calibrate it, or evaluate it.

### Confidence buckets (the user-facing interpretation)

| Bucket | Meaning |
|---|---|
| `p ≥ 0.95` | Definitely remove |
| `0.7 ≤ p < 0.95` | Probably remove |
| `p < 0.7` | Keep |

Calibration matters: `p = 0.9` must mean ~90% precision on average. We enforce this with temperature scaling on the dev set ([Guo et al., 2017](https://arxiv.org/abs/1706.04599)).

### Definitions

| Term | Definition |
|---|---|
| Segment | Topic-segmented span of 3–25 sentences (sentence-level kept for ablation) |
| Removability | Calibrated probability output of the trained classifier |
| Information loss | 1 − SBERT cosine(full, reduced transcript); secondary: ROUGE-L recall |
| Genre | Channel-level coarse label assigned at scrape time |

---

## 4. Corpus

- **Target: ~5 000 English-only YouTube transcripts**, across 6 genres.
- **Genres:** commentary, entertainment, lectures, podcasts, TED talks, TV series.
- **Filters:** English `en` caption track, ≥500 words, manual captions preferred over auto.
- **Splits (full corpus):** 4 000 train / 500 dev / 500 test, stratified by genre × length.

---

## 5. Pipeline

```
scrape ──► segment ──► weak labels ──► train ──► calibrate ──► evaluate
  │           │            │              │          │             │
sources.csv  spaCy +    Snorkel        roberta-base  fit T      gold set
yt-dlp       embedding  LabelModel     3 heads       on dev     per-video
             boundaries 7 rule LFs                              JSON files
```

### Implementation status

| File | Status | Notes |
|---|---|---|
| `src/scrape.py` | ✅ Done | yt-dlp, 6 genres, sharding, checkpoint resume |
| `src/segment.py` | ✅ Done | spaCy sentences + sentence-embedding topic boundaries |
| `src/weak_labels.py` | ✅ Done | 7 LFs → Snorkel LabelModel → `weak_labels.jsonl` (Head A) |
| `src/summary_align.py` | ✅ Done | 7th LF using local BART summarization |
| `src/function_labels.py` | ✅ Done | 13 rule LFs + optional LLM → `function_labels.jsonl` (Head B) |
| `src/disfluency_labels.py` | ✅ Done | 7 rule LFs + optional LLM → `disfluency_labels.jsonl` (Head C) |
| `src/baselines.py` | ✅ Done | Random / Heuristic / TF-IDF / SBERT baselines → `baselines.jsonl` |
| `src/train.py` | ✅ Done | BERTSum 3-head model, temperature calibration, Slurm-ready |
| `src/eval.py` | ✅ Done | Full 10-section evaluation suite (see §10) |
| `notebooks/eda.ipynb` | ✅ Done | §1–11, corpus features + weak label analysis |
| `data/source_registry.csv` | ✅ Done | 6 genres, 50+ sources, 5 000 target |
| `slurm/run_scrape.sh` | ✅ Done | Array job, 4 shards |
| `slurm/run_labels.sh` | ✅ Done | Full label pipeline (segment → all 3 heads) |
| `slurm/run_train.sh` | ✅ Done | Wired to `src/train.py` |

---

## 6. Quickstart

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 1. Collect transcripts
python src/scrape.py --sources sources.csv

# 2. Segment transcripts into topic chunks
python src/segment.py

# 3. Generate weak labels (6 local LFs)
python src/weak_labels.py

# 4. Optional: add summary-alignment LF (downloads ~1.6 GB BART on first run)
python src/summary_align.py
python src/weak_labels.py   # re-run to fold in the 7th LF

# 5. Function labels (Head B — 13 rule LFs, no LLM needed)
python src/function_labels.py
# With open-source LLM for 14th LF:
# python src/function_labels.py --llm-model microsoft/Phi-3.5-mini-instruct

# 6. Disfluency labels (Head C — 7 rule LFs)
python src/disfluency_labels.py

# 7. Baseline scores (required before eval)
python src/baselines.py

# 8. Train
python src/train.py

# 9. Evaluate (smoke test — CPU OK)
python src/eval.py --max-docs 5 --no-model

# 9. Evaluate (full — GPU required)
python src/eval.py --checkpoint models/best.pt

# ── On HPC with Slurm ──────────────────────────────────────────────
# sbatch slurm/run_scrape.sh
# sbatch slurm/run_labels.sh microsoft/Phi-3.5-mini-instruct
# sbatch slurm/run_train.sh
```

---

## 7. Annotation Scheme (gold set only — 150–300 segments)

Three orthogonal layers per segment. Annotator marks all three; *removability is derived*, not annotated directly. Stratified across genre × length, sampled from the test split only — never train or dev.

**Layer 1 — Function** (6-class):

| Label | Meaning | Maps to |
|---|---|---|
| `new_information` | Introduces a fact not yet stated | keep |
| `useful_repetition` | Repeats for emphasis or pedagogy | keep |
| `redundant_repetition` | Restates with no added value | remove |
| `clarification` | Elaborates prior content | keep |
| `discourse_filler` | Hedges, transitions, "um/you know" | remove |
| `off_topic` | Tangents, banter | remove |

**Layer 2 — Disfluency** (5-class, surface-level): `clean`, `filled_pause`, `repetition`, `revision`, `restart`.

**Layer 3 — Confidence** (3-point): `definitely` / `probably` / `unsure`. Provides ground truth for calibration evaluation directly.

Two-annotator κ on Function targeted at ≥ 0.6 (single-annotator with documented limitation if no co-annotator).

### Gold annotation format

Each annotated video is stored as a separate JSON file in `data/gold/videos/{video_id}.json`:

```json
{
  "video_id": "abc123",
  "removability_to_soft_target": {
    "definitely_remove": 1.0,
    "probably_remove":   0.75,
    "unsure":            0.5,
    "probably_keep":     0.25,
    "definitely_keep":   0.0
  },
  "function_to_binary_remove": {
    "redundant_repetition": 1,
    "discourse_filler":     1,
    "off_topic":            1,
    "new_information":      0,
    "useful_repetition":    0,
    "clarification":        0
  },
  "segments": [
    {
      "seg_idx": 0,
      "gold_removability": "probably_keep",
      "gold_function":     "new_information",
      "gold_disfluency":   "clean",
    }
  ]
}
```

---

## 8. Model: Multi-head BERTSum (hierarchical, document-aware)

Architecture follows [Liu & Lapata, BERTSum (EMNLP 2019)](https://aclanthology.org/D19-1387.pdf): insert `[CLS]` before each segment, get a `[CLS]` embedding per segment, feed those embeddings through a second-stage Transformer so each segment carries full-document context, and stack three heads on top.

The hierarchical design is essential: a 20-minute video is far longer than `roberta-base`'s 512-token window, and a *redundant repetition* often refers back to something said 5 minutes earlier. Without document-level awareness the model literally cannot see what's being repeated.

```
                          Document (60 segments)
                                  ↓
          ┌──────────────────────────────────────────┐
Stage 1   │  roberta-base encoder, applied per chunk │
(local)   │  Output: one [CLS] embedding per segment │
          └──────────────────────────────────────────┘
                                  ↓
          ┌──────────────────────────────────────────┐
Stage 2   │  Inter-segment Transformer (2–4 layers)  │
(global)  │  All segments attend to each other       │
          │  Each [CLS] now sees the whole document  │
          └──────────────────────────────────────────┘
                                  ↓
                  ┌───────────────┼───────────────┐
                  ↓               ↓               ↓
               Head A          Head B          Head C
            Removability      Function       Disfluency
            sigmoid → [0,1]   6 classes      5 classes
            ◄── PRIMARY ──►
```

**Why three heads:** one encoder, three label streams per segment. Heads B and C regularize the encoder and let us trace *why* a segment received a high `p_remove`. Reference: [Ruder, multi-task learning (2017)](https://arxiv.org/abs/1706.05098).

**Repetition detection.** Stage 2's self-attention over segment embeddings acts as a learned similarity function — `Q·K` dot products catch "segment 50 echoes segment 5." We trust attention to learn this from the data. SBERT cosine appears separately as a weak-labeling function (`LF_repetition`), where noisy scalable supervision is its appropriate role.

**Loss:** `L = 1.0 · BCE(p_remove) + 0.5 · CE(function) + 0.5 · CE(disfluency)`.

---

## 9. Training with Weak Supervision

Hand-labeling 5 000 documents is infeasible in a semester. We label automatically with noisy rules and aggregate them via a probabilistic model, train on the result, and use hand-labeled gold only for evaluation. Standard methodology — [Ratner et al., Snorkel (VLDB 2018)](https://arxiv.org/abs/1711.10160).

### The labeling functions

A **labeling function (LF)** votes: `REMOVE (1)`, `KEEP (0)`, or `ABSTAIN (-1)`. No single LF is accurate enough to trust alone, but aggregating 7 LFs that each capture a different signal gives a reliable soft label per segment.

| LF | Signal | Vote | Threshold |
|---|---|---|---|
| `LF_filler` | filler word density | remove | > 3% |
| `LF_repetition` | max SBERT cosine to any earlier segment | remove | > 0.70 |
| `LF_short` | content words after stopword removal | remove | < 5 words |
| `LF_low_tfidf` | TF-IDF sum within document | remove | bottom 20% |
| `LF_named_entity` | named entities (spaCy NER) | keep | ≥ 3 entities |
| `LF_question_or_def` | contains `?` or definition phrase | keep | any match |
| `LF_summary_align` | cosine alignment to local BART summary | remove / keep | bottom / top 30% |

Snorkel's `LabelModel` estimates each LF's accuracy from inter-LF agreement patterns — no gold labels required. Output: a soft `p_remove ∈ [0, 1]` per segment.

### Training procedure

1. Generate weak labels offline (`src/weak_labels.py`, optionally `src/summary_align.py` first).
2. Fine-tune `roberta-base`. AdamW, lr 2e-5, 3 epochs, batch 16. Class-weighted loss.
3. Early-stop on dev-set Head-A AUC against weak labels.
4. **Calibrate:** fit a single temperature `T` on the dev set minimising ECE. Save `T` alongside the checkpoint in `config.json`.
5. **Critical invariant:** the gold set is touched only at evaluation. No early-stopping on it, no inspection during training.

---

## 10. Evaluation

Two tracks, both stratified by genre and length bucket — that is how H1, H3, and H4 are tested. The evaluation suite is `src/eval.py` (10 sections, all outputs go to `outputs/eval/`).

### 10.1 Segment-level (intrinsic — gold subset when available, weak-label proxy otherwise)

| Metric | Description |
|---|---|
| AUC-ROC, AUC-PR | Ranking quality with 95% bootstrap CI |
| ECE (10 bins) | Expected Calibration Error |
| F1 / P / R @ 0.5 | Classification at the default operating threshold |
| Reliability diagram | Overall + per-genre small multiples |
| Calibration before/after T | Side-by-side to show temperature-scaling effect |
| Confidence-bucket precision | Precision at p ≥ 0.95 / 0.70 / < 0.70 |

### 10.2 Transcript-level (extrinsic — full test set) — the headline

Sweep `p_remove` threshold from 0 → 1. At each step compute:

1. **Compression ratio** (% words kept)
2. **SBERT cosine** (centroid of full vs reduced transcript) — semantic preservation proxy
3. **ROUGE-1-R and ROUGE-L-R** (recall vs original) — lexical preservation

**Headline figure:** compression vs SBERT-cosine curve, one line per genre. Tests H1, H2, H3 in a single plot.

*Note: QA-preservation (QAGS-style) is not computed — SBERT cosine serves as the semantic proxy. This avoids LLM API dependency and is sufficient for the preservation–compression trade-off analysis.*

### 10.3 Baseline comparison

| Tier | Baseline |
|---|---|
| Sanity | Random (biased coin matching positive rate) |
| Heuristic | Filler-word + repetition rule |
| Lexical | TF-IDF importance |
| Unsupervised neural | SBERT-Scorer (1 − cosine to centroid) |

All baselines shown with 95% bootstrap CI error bars, both overall and per-genre grouped bars.

### 10.4 4-way agreement matrix

Pairwise Spearman ρ between four operationalisations of "removable":

| Definition | Signal |
|---|---|
| Model `p_remove` | Calibrated classifier output |
| Centroid-deviation | SBERT distance from document centroid |
| Surface heuristics | Filler density + low TF-IDF + low NER |
| Weak-label proxy | Snorkel LabelModel output |

Shown overall (heatmap) and per-genre (bar chart of ρ values).

### 10.5 Gold evaluation (activates when `data/gold/videos/` is populated)

| Metric | Description |
|---|---|
| Spearman ρ | Model p_remove vs human gold soft target |
| NDCG@20 | Ranking quality against gold ordinal |
| Binary accuracy / F1 | Model and weak labels vs gold binary label |
| LF accuracy | Per-labeling-function accuracy against gold (requires `lf_votes.jsonl`) |

### 10.6 Caption-type robustness

AUC-ROC and PR-AUC split by `caption_type` (manual vs auto-generated). Quantifies sensitivity to transcription quality — addresses limitation §11.2 directly.

### 10.7 Qualitative examples

Per-genre failure mode table: one false positive and one false negative per genre, with function label and text snippet.

---

## 11. Data Directory Structure

```
data/
  processed/
    segments_topic.jsonl        # topic-segmented transcripts
    weak_labels.jsonl           # Snorkel LabelModel output (p_remove per segment)
    function_labels.jsonl       # Head B labels (6-class function)
    disfluency_labels.jsonl     # Head C labels (5-class disfluency)
    baselines.jsonl             # random / heuristic / tfidf / sbert scores
    corpus_summary.csv          # per-video metadata (genre, length_bucket, etc.)
    lf_votes.jsonl              # (optional) per-LF vote matrix for LF accuracy analysis
  gold/
    videos/
      {video_id}.json           # one file per annotated video (fill locally, scp to HPC)
models/
  best.pt                       # trained checkpoint
  config.json                   # encoder name, temperature T, hyperparameters
outputs/
  eval/
    plots/                      # all PNG + PDF figures
    audit.md
    transcript_eval.csv
    qualitative_examples.md
    gold_eval.md                # populated only when gold data exists
    evaluation_report.md
```

---

## 12. Limitations

1. **Transcript-only.** No audio prosody, no video frames. A diagram referenced as "this graph" is invisible to the model. Directly affects H4.
2. **ASR noise.** Auto-generated captions contain transcription errors that can appear as disfluencies. Tracked via the `caption_type` field and the robustness evaluation (§10.6).
3. **Weak label noise.** Summary alignment occasionally marks an important detail as removable when the summary glosses over it. Evaluated on gold annotations only — never used as training signal for the gold set.
4. **English only.** No multilingual claims are made.
5. **Annotation subjectivity.** Useful vs redundant repetition is a judgment call. Mitigated with the confidence layer and κ reporting.
6. **Genre coverage.** Six genres, channel-curated. Generalisation to news, sports, or domain-specific content is unsupported.
7. **Segment granularity.** Topic boundaries are derived from cosine-similarity dips — a heuristic. Sentence-level ablation partially controls for this (§8, ablation note).
8. **Single-annotator risk.** If no co-annotator is available, inter-annotator κ is undefined. Documented as a limitation where applicable.
9. **QA-preservation.** SBERT cosine serves as the semantic preservation proxy. Full QAGS-style evaluation would require LLM API access (estimated ~$20–40 for 500 documents).

---

## 13. What the Article Will Need

### Article checklist

| Item | Status |
|---|---|
| Explicit definitions section (§3 table verbatim) | ✅ In §3 |
| 4-way agreement matrix as a figure, broken down by genre | ✅ `agreement_heatmap.png` + `agreement_by_genre.png` |
| Reliability diagrams for model + per-genre small multiples | ✅ `model_reliability.png` + `reliability_by_genre.png` |
| Calibration before/after temperature scaling | ✅ `calibration_before_after_T.png` |
| Per-genre, per-length stratified results | ✅ `model_genre_auc.png`, `transcript_by_length.png`, `baseline_by_genre.png` |
| Caption-type robustness check | ✅ `caption_type_comparison.png` |
| Table of failure modes — one annotated example per genre | ✅ `qualitative_examples.md` per-genre table |
| Gold evaluation: Spearman ρ, NDCG@20, weak-label accuracy | ✅ `gold_eval.md` (activates on annotation) |
| Bootstrap CI error bars on all key bar charts | ✅ 95% CI, 500 resamples |
| Reproducibility appendix — seed, splits, T, LF config | ✅ `evaluation_report.md` appendix |


### Pipeline improvements (future work)

1. **Annotator-confidence weighting in the loss.** Weight each gold-eval point by annotator certainty so calibration is judged on consensus cases more strictly.
2. **Cross-segment coherence.** A CRF or BiLSTM head over the segment sequence would penalise choices that produce jumpy cuts.
3. **Beyond binary.** Ranking objective for Head A; NDCG / Spearman evaluation already implemented (§10.5).
4. **LongFormer and LLM-as-classifier baselines.** LongFormer-base (4 096 tokens) and Claude Haiku few-shot — planned as upper-bound baselines, excluded from current evaluation due to compute cost.
5. **Listener-aware compression.** Personalise `p_remove` per user — beginner vs expert.
6. **Multimodal extension.** Add audio prosody and visual frames. Natural fix for H4 failure cases.
7. **Multilingual transfer.** Swap `roberta-base` for `xlm-roberta`; test transfer to non-English.

---

## 14. Datasets and Precedents

| Dataset / paper | What it gives the article |
|---|---|
| [TalkSumm (Lev et al., ACL 2019)](https://aclanthology.org/P19-1204/) | Methodological precedent for weak-label strategy |
| [VT-SSum (2021)](https://arxiv.org/abs/2106.05606) | Closest direct precedent — 9 096 video transcripts with annotations |
| [How2 (Sanabria et al., ACL 2019)](https://aclanthology.org/P19-1659.pdf) | Cross-domain robustness test |
| [TED-LIUM](https://huggingface.co/datasets/LIUM/tedlium) | TED transcripts with audio — useful for multimodal extension |
| [Miller, BERT extractive on lectures (2019)](https://arxiv.org/abs/1906.04165) | Most directly comparable architecture on lecture content |
| [SummEval (Fabbri et al., TACL 2021)](https://arxiv.org/abs/2007.12626) | Justifies metric choices |
| [AMI annotation schema](https://groups.inf.ed.ac.uk/ami/corpus/annotation.shtml) | Reference for the 6-class Function label set |

---

## 15. Reading List

**Read these three first:**
1. [Ratner et al., Snorkel (VLDB 2018)](https://arxiv.org/abs/1711.10160) — how weak labeling works
2. [Liu & Lapata, BERTSum (EMNLP 2019)](https://aclanthology.org/D19-1387.pdf) — the architecture
3. [Guo et al., Calibration (ICML 2017)](https://arxiv.org/abs/1706.04599) — temperature scaling

**Direct domain precedents:**
- [TalkSumm (Lev et al., ACL 2019)](https://aclanthology.org/P19-1204/) — the weak-label trick we adapt
- [VT-SSum (2021)](https://arxiv.org/abs/2106.05606) — video transcript segmentation + summarisation
- [How2 (Sanabria et al., ACL 2019)](https://aclanthology.org/P19-1659.pdf) — instructional video transcripts
- [Miller, BERT extractive on lectures (2019)](https://arxiv.org/abs/1906.04165)

**Evaluation:**
- [BERTScore (Zhang et al., ICLR 2020)](https://arxiv.org/abs/1904.09675)
- [SummEval (Fabbri et al., TACL 2021)](https://arxiv.org/abs/2007.12626)

**Multi-task and disfluency:**
- [Ruder, multi-task survey (2017)](https://arxiv.org/abs/1706.05098)
- [Disfluency Detection via LLMs (STIL 2024)](https://aclanthology.org/2024.stil-1.16.pdf)
- [AMI annotation schema](https://groups.inf.ed.ac.uk/ami/corpus/annotation.shtml)

**Datasets hub:** [Awesome-Summarization-Datasets](https://github.com/edahanoam/Awesome-Summarization-Datasets).

---

## 16. Open Decisions

1. **Encoder size.** `roberta-base` (125M) on local GPU vs `roberta-large` (355M) on HPC. Default: base first.
2. **Gold size.** 200 segments default, expand to 300 if time allows.
3. **Co-annotator.** Two-annotator κ vs single-annotator with documented limitation.
4. **Multi-task loss weights.** Default `1.0 / 0.5 / 0.5`, tunable on dev.
5. **LLM baselines.** LongFormer and Claude Haiku excluded from current eval due to compute cost; noted as future work.
