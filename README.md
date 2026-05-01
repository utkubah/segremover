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

### Confidence buckets (the user-facing story)

| Bucket | Meaning |
|---|---|
| `p ≥ 0.95` | Definitely remove |
| `0.7 ≤ p < 0.95` | Probably remove |
| `p < 0.7` | Keep |

Calibration matters: `p = 0.9` must actually mean ~90% accuracy on average. We enforce this with temperature scaling on the dev set ([Guo et al., 2017](https://arxiv.org/abs/1706.04599)).

### Definitions used in the report

| Term | Definition |
|---|---|
| Segment | Topic-segmented span of 3–25 sentences (sentence-level kept for ablation) |
| Removability | Calibrated probability output of the trained classifier |
| Information loss | 1 − SBERT cosine(full, reduced); secondary: drop in QA-preservation |
| Genre | Channel-level coarse label assigned at scrape time |

---

## 4. Corpus

- **Target: ~5000 English-only YouTube transcripts**, 3 genres × ~1666 each. Currently scraped: **79 videos** (development phase).
- **Genres:** MIT lectures (MIT 18.06 Linear Algebra), TED talks (@TED), Commentary (@Veritasium).
- **Length stratification within each genre:** ~33% short (<5 min), ~33% medium (5–20 min), ~33% long (>20 min).
- **Filters:** English `en` caption track, ≥500 words, manual captions preferred over auto.
- **Splits (full corpus):** 4000 train / 500 dev / 500 test, stratified by genre × length.

| Genre | Source | Target | Scraped |
|---|---|---|---|
| Lectures | MIT 18.06 Linear Algebra | 34 | 25 |
| TED talks | @TED | 14 | 20 |
| Commentary | @Veritasium | 34 | 34 |

---

## 5. Pipeline

```
scrape ──► segment ──► weak labels ──► train ──► calibrate ──► evaluate
  │           │            │              │          │             │
sources.csv  spaCy +    summary       roberta-base  fit T      gold set
yt-dlp       embedding  alignment +   3 heads       on dev     + 500-doc
             boundaries rule LFs                                test set
                                                                QA preservation
```

### Implementation status

| File | Status |
|---|---|
| `src/scrape.py` | ✅ Done — yt-dlp, checkpoint resume, cookie support |
| `src/segment.py` | ✅ Done — spaCy sentences + sentence-embedding topic boundaries |
| `notebooks/eda.ipynb` | ✅ Done — §1–11, corpus features + weak label analysis, saves `corpus_summary.csv` |
| `sources.csv` | ✅ Done — all 3 genres |
| `src/weak_labels.py` | ✅ Done — 6 local LFs → Snorkel LabelModel → `weak_labels.jsonl` |
| `src/summary_align.py` | ✅ Done — optional 7th LF (local BART, ~1.6 GB download) |
| `src/train.py` | ⬜ Next |
| `src/baselines.py` | ⬜ Next |
| `src/eval_segment.py`, `src/eval_transcript.py` | ⬜ Last |

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

# 5. Explore results
jupyter notebook notebooks/eda.ipynb
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

**Layer 3 — Confidence** (3-point): definitely / probably / unsure. Provides ground truth for the calibration evaluation directly.

Two-annotator κ on Function targeted at ≥ 0.6 (single-annotator with documented limitation if no co-annotator).

---

## 8. Model: Multi-head BERTSum (hierarchical, document-aware)

Architecture follows [Liu & Lapata, BERTSum (EMNLP 2019)](https://aclanthology.org/D19-1387.pdf): insert `[CLS]` before each segment, get a `[CLS]` embedding per segment, feed those embeddings through a second-stage Transformer so each segment carries full-document context, and stack three heads on top.

The hierarchical design is essential: a 20-minute video is far longer than `roberta-base`'s 512-token window, and a *redundant repetition* often refers back to something said 5 minutes earlier. Without document-level awareness the model literally cannot see what's being repeated.

```
                              Document (60 segments)
                                      ↓
              ┌──────────────────────────────────────────┐
 Stage 1      │  roberta-base encoder, applied per chunk │
 (local)      │  Output: one [CLS] embedding per segment │
              └──────────────────────────────────────────┘
                                      ↓
              ┌──────────────────────────────────────────┐
 Stage 2      │  Inter-segment Transformer (2–4 layers)  │
 (global)     │  All segments attend to each other       │
              │  Each [CLS] now sees the whole doc       │
              └──────────────────────────────────────────┘
                                      ↓
                      ┌───────────────┼───────────────┐
                      ↓               ↓               ↓
                   Head A          Head B          Head C
                Removability      Function       Disfluency
                sigmoid → [0,1]   6 classes      5 classes
                ◄── PRIMARY ──►
```

**Why three heads:** one encoder, three label streams per segment. Heads B and C regularize the encoder and let us trace *why* a segment got a high `p_remove`. Reference: [Ruder, multi-task learning (2017)](https://arxiv.org/abs/1706.05098).

**Repetition detection.** Stage 2's self-attention over segment embeddings *is* a learned similarity function — `Q·K` dot products are exactly what would catch "segment 50 echoes segment 5". We trust attention to learn this from the data rather than patching it with hand-crafted similarity features. SBERT cosine still appears in the pipeline as a weak-labeling function (§9, `LF_repetition`), where noisy scalable supervision is its right job.

**Loss:** `L = 1.0 · BCE(p_remove) + 0.5 · CE(function) + 0.5 · CE(disfluency)`.

---

## 9. Training with Weak Supervision

Hand-labeling 5000 documents is impossible in a semester. We label the 5000 *automatically* with noisy rules and an LLM, train on those, and only use hand-labeled gold for evaluation. Standard methodology — [Ratner et al., Snorkel (VLDB 2018)](https://arxiv.org/abs/1711.10160).

### The labeling functions

A **labeling function (LF)** is a simple rule that looks at one segment and votes: `REMOVE (1)`, `KEEP (0)`, or `ABSTAIN (-1)`. No single LF is accurate enough to trust alone, but aggregating 6–7 LFs that each capture a different signal gives a reliable soft label for every segment.

| LF | Signal | Vote | Threshold |
|---|---|---|---|
| `LF_filler` | filler word density | remove | > 3% |
| `LF_repetition` | max SBERT cosine to any earlier segment in doc | remove | > 0.70 |
| `LF_short` | content words after stopword removal | remove | < 5 words |
| `LF_low_tfidf` | TF-IDF sum within document | remove | bottom 20% |
| `LF_named_entity` | named entities (spaCy NER) | keep | ≥ 3 entities |
| `LF_question_or_def` | contains `?` or definition phrase | keep | any match |
| `LF_summary_align` | cosine alignment to local BART summary | remove / keep | bottom / top 30% |

Each LF abstains when it lacks signal. Snorkel's `LabelModel` estimates each LF's accuracy from inter-LF agreement patterns — no gold labels required. Output: a soft `p_remove ∈ [0, 1]` per segment.

```
Segment X  →  LF_filler:        remove  (1)
           →  LF_repetition:    abstain (-1)
           →  LF_short:         abstain (-1)
           →  LF_low_tfidf:     remove  (1)
           →  LF_named_entity:  keep    (0)
           →  LF_summary_align: remove  (1)
                                      ↓
                          LabelModel aggregation
                                      ↓
                            p_remove = 0.72
```

**Training procedure:**

1. Generate weak labels offline (`src/weak_labels.py`, optionally `src/summary_align.py` first).
2. Fine-tune `roberta-base`. AdamW, lr 2e-5, 3 epochs, batch 16. Class-weighted loss.
3. Early-stop on dev-set Head-A AUC against weak labels.
4. **Calibrate:** fit a single temperature `T` on dev minimizing ECE. Save `T` next to checkpoint.
5. **Critical invariant:** the gold set is touched only at evaluation. No early-stopping on it, no inspection during training.

---

## 10. Evaluation

Two tracks. Both stratified by genre and length bucket — that's how H1, H3, H4 get tested.

### 10.1 Segment-level (intrinsic, on the gold subset)

- F1 / Precision / Recall at calibrated 0.5
- AUC-ROC, AUC-PR
- ECE (Expected Calibration Error, 10 bins)
- Confidence-bucket precision at thresholds {0.95, 0.7}

### 10.2 Transcript-level (extrinsic, on the 500-doc test set) — the headline

Sweep `p_remove` threshold from 0 → 1, compute at each step:

1. **Compression ratio** (% words kept)
2. **SBERT cosine** (full vs reduced)
3. **ROUGE-L** vs LLM-generated summary of the full transcript
4. **QA preservation (QAGS-style)** — LLM generates ~10 factual questions from full transcript; answers from full *and* reduced; BERTScore between answer sets. [Wang et al., QAGS (ACL 2020)](https://aclanthology.org/2020.acl-main.450/).

**Headline figure:** compression-vs-QA-preservation curve, one per genre. Tests H1, H2, H3 in a single plot.

### 10.3 The 4-way agreement matrix

Compares four operationalizations of "removable" head-to-head:

| Definition | Signal |
|---|---|
| Model `p_remove` | Calibrated classifier output |
| Centroid-deviation | SBERT distance from doc centroid |
| Surface heuristics | Filler density + low TF-IDF + low NER |
| Human gold | Function ∈ {`redundant_repetition`, `discourse_filler`, `off_topic`} |

### 10.4 Baselines

| Tier | Baseline |
|---|---|
| Sanity | Random (biased coin matching positive rate) |
| Heuristic | Filler-word + repetition rule |
| Lexical | TF-IDF importance |
| Unsupervised neural | SBERT-Scorer (1 − cosine to centroid) |
| Long context | LongFormer-base (4096 tok) |
| Zero-shot | LLM-as-classifier (Claude Haiku, few-shot) |

---

## 11. Limitations

1. **Transcript-only.** No audio prosody, no video frames. A diagram referenced as "this graph" is invisible to us. Affects H4 directly.
2. **ASR noise.** Auto-generated captions contain transcription errors that can look like disfluencies. Tracked via `caption_type` field.
3. **Weak label noise.** Summary alignment occasionally marks an important detail as removable when the summary glosses over it. Evaluated on gold, not on training labels.
4. **English only.** No multilingual claims.
5. **Annotation subjectivity.** Useful vs redundant repetition is a judgment call. Mitigated with the confidence layer + κ reporting.
6. **Genre coverage.** Three genres, channel-curated. Generalization to news, sports, podcasts is unsupported.
7. **Segment granularity.** Topic boundaries are derived from cosine-similarity dips — a heuristic. Sentence-level ablation partially controls for this.
8. **Single-annotator risk.** If no co-annotator, κ is undefined.
9. **LLM evaluator dependency.** QAGS-style QA preservation inherits any biases of the evaluating LLM.

---

## 12. Improvements & What the Article Will Need

### Pipeline improvements

1. **Annotator-confidence weighting in the loss.** Weight each gold-eval point by annotator certainty so calibration is judged on consensus cases more strictly.
2. **Cross-segment coherence.** A CRF or BiLSTM head over the segment sequence would penalize choices that produce jumpy cuts.
3. **Gold expansion via the model itself.** Once calibrated, pre-label thousands of segments; humans verify only high-confidence outputs.
4. **Beyond binary.** Replace Head A's sigmoid with a ranking objective; evaluate against NDCG / Spearman against gold.
5. **Listener-aware compression.** Personalize `p_remove` per user — beginner vs expert.
6. **Multimodal extension.** Add audio prosody and visual frames. Natural fix for H4 failure cases.
7. **Multilingual transfer.** Swap `roberta-base` for `xlm-roberta`; test transfer to non-English.
8. **Real-time variant.** Causal version that scores segments as they arrive.

### Datasets and precedents

| Dataset / paper | What it gives the article |
|---|---|
| [TalkSumm (Lev et al., ACL 2019)](https://aclanthology.org/P19-1204/) | Methodological precedent for our Head A weak labels |
| [VT-SSum (2021)](https://arxiv.org/abs/2106.05606) | Closest direct precedent — 9,096 video transcripts with annotations |
| [How2 (Sanabria et al., ACL 2019)](https://aclanthology.org/P19-1659.pdf) | Cross-domain robustness test |
| [TED-LIUM](https://huggingface.co/datasets/LIUM/tedlium) | TED transcripts with audio — useful for multimodal extension |
| [Miller, BERT extractive on lectures (2019)](https://arxiv.org/abs/1906.04165) | Most directly comparable architecture on lecture content |
| [SummEval (Fabbri et al., TACL 2021)](https://arxiv.org/abs/2007.12626) | Justifies metric choices (BERTScore + QAGS over ROUGE-only) |
| [AMI annotation schema](https://groups.inf.ed.ac.uk/ami/corpus/annotation.shtml) | Reference for the 6-class Function label set |

### Article checklist

- **Explicit definitions section** reproducing the §3 table verbatim
- **4-way agreement matrix as a figure** (not just a table) — visual disagreement across genres
- **Reliability diagrams** for primary model + two strongest baselines
- **Per-genre, per-length stratified results** everywhere (H1 and H3 are only testable this way)
- **Caption-type robustness check** — manual vs auto split on test set
- **Table of failure modes** — one annotated example per genre
- **Honest cost ledger** — LLM call counts and dollar cost
- **Reproducibility appendix** — channel list, scrape date, split hashes, weak-label config, training seed, calibration `T`

---

## 13. Open Decisions

1. **Encoder size.** `roberta-base` (125M) on local GPU vs `roberta-large` (355M) on HPC. Default: base first.
2. **LLM for weak labels / evaluation.** Claude Haiku vs GPT-4o-mini. Estimated cost: ~$5–10 for 5000 summaries, ~$20–40 for QA evaluation.
3. **Gold size.** 200 segments default, expand to 300 if time allows.
4. **Co-annotator.** Two-annotator κ vs single-annotator with documented limitation.
5. **Multi-task loss weights.** Default `1.0 / 0.5 / 0.5`, tunable on dev.

---

## 14. Reading List

**Read these three first:**
1. [Ratner et al., Snorkel (VLDB 2018)](https://arxiv.org/abs/1711.10160) — how weak labeling works
2. [Liu & Lapata, BERTSum (EMNLP 2019)](https://aclanthology.org/D19-1387.pdf) — the architecture
3. [Guo et al., Calibration (ICML 2017)](https://arxiv.org/abs/1706.04599) — temperature scaling

**Direct domain precedents:**
- [TalkSumm (Lev et al., ACL 2019)](https://aclanthology.org/P19-1204/) — the weak-label trick we adapt
- [VT-SSum (2021)](https://arxiv.org/abs/2106.05606) — video transcript segmentation + summarization
- [How2 (Sanabria et al., ACL 2019)](https://aclanthology.org/P19-1659.pdf) — instructional video transcripts
- [Miller, BERT extractive on lectures (2019)](https://arxiv.org/abs/1906.04165)

**Evaluation:**
- [QAGS (Wang et al., ACL 2020)](https://aclanthology.org/2020.acl-main.450/) — QA-preservation methodology
- [BERTScore (Zhang et al., ICLR 2020)](https://arxiv.org/abs/1904.09675)
- [SummEval (Fabbri et al., TACL 2021)](https://arxiv.org/abs/2007.12626)

**Multi-task and disfluency:**
- [Ruder, multi-task survey (2017)](https://arxiv.org/abs/1706.05098)
- [Disfluency Detection via LLMs (STIL 2024)](https://aclanthology.org/2024.stil-1.16.pdf)
- [AMI annotation schema](https://groups.inf.ed.ac.uk/ami/corpus/annotation.shtml)

**Datasets hub:** [Awesome-Summarization-Datasets](https://github.com/edahanoam/Awesome-Summarization-Datasets).
