# Project Plan: Measuring Information Loss in Video Transcripts

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

- **5000 English-only YouTube transcripts**, 3 genres × ~1666 each.
- **Genres:** MIT lectures (MIT OCW, CrashCourse), TED talks, English commentary (Lex Fridman / Veritasium / 3Blue1Brown — TBD).
- **Length stratification within each genre:** ~33% short (<5 min), ~33% medium (5–20 min), ~33% long (>20 min).
- **Filters:** English `en` caption track, ≥500 words, manual captions preferred over auto.
- **Splits:** 4000 train / 500 dev / 500 test, stratified by genre × length.

---

## 5. Pipeline

```
scrape ──► segment ──► weak labels ──► train ──► calibrate ──► evaluate
  │           │            │              │          │             │
  │           │            │              │          │             │
sources.csv  spaCy +    summary       roberta-base  fit T      gold set
yt-dlp       embedding  alignment +   3 heads       on dev     + 500-doc
youtube-     boundaries rule LFs                                test set
transcript-                                                     QA preservation
api
```

### Implementation status

| File | Status |
|---|---|
| `src/scrape.py` | Done — yt-dlp + youtube-transcript-api, checkpoint resume |
| `src/segment.py` | Done — spaCy sentences + sentence-embedding topic boundaries |
| `notebooks/eda.ipynb` | Done — ties every figure to H1/H3/H4, saves `corpus_summary.csv` |
| `sources.csv` | Lectures + TED filled in; third genre TBD |
| `src/weak_labels.py` | Next |
| `src/train.py` | Next |
| `src/eval_segment.py`, `src/eval_transcript.py` | Last |

---

## 6. Annotation Scheme (gold set only — 150–300 segments)

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

## 7. Model: Multi-head BERTSum (hierarchical, document-aware)

Architecture follows [Liu & Lapata, BERTSum (EMNLP 2019)](https://aclanthology.org/D19-1387.pdf): insert `[CLS]` before each segment, get a `[CLS]` embedding per segment, feed those embeddings through a second-stage Transformer so each segment carries full-document context, and stack three heads on top.

The hierarchical design is essential: a 20-minute video is far longer than `roberta-base`'s 512-token window, and a *redundant repetition* often refers back to something said 5 minutes earlier. Without document-level awareness the model literally cannot see what's being repeated.

```
                                  Document (60 segments)
                                          ↓
                  ┌──────────────────────────────────────────┐
   Stage 1        │  roberta-base encoder, applied per chunk │
   (local)        │  Output: one [CLS] embedding per segment │
                  └──────────────────────────────────────────┘
                                          ↓
                  ┌──────────────────────────────────────────┐
   Stage 2        │  Inter-segment Transformer (2–4 layers)  │
   (global)       │  All segments attend to each other       │
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

**Repetition detection.** Stage 2's self-attention over segment embeddings *is* a learned similarity function — `Q·K` dot products are exactly what would catch "segment 50 echoes segment 5". We trust attention to learn this from the data rather than patching it with hand-crafted similarity features. SBERT cosine still appears in the pipeline as a weak-labeling function (§8, `LF_repetition`), where noisy scalable supervision is its right job. If gold-set evaluation shows the model misses long-range repetitions, we add `max_cosine_to_earlier_segments` as an explicit Head-A feature *then* and report it as an ablation result.

**Loss:** `L = 1.0 · BCE(p_remove) + 0.5 · CE(function) + 0.5 · CE(disfluency)`.

**Long-document fallback.** For videos that exceed Stage-1's chunked context (rare — most segments fit in 512 tokens individually), we keep `LongFormer-base` as both a baseline (§9.4) and a drop-in replacement for Stage 1 if hierarchical roberta underperforms on long lectures.

---

## 8. Training with Weak Supervision

Hand-labeling 5000 documents is impossible in a semester. We label the 5000 *automatically* with noisy rules and an LLM, train on those, and only use hand-labeled gold for evaluation. Standard methodology — [Ratner et al., Snorkel (VLDB 2018)](https://arxiv.org/abs/1711.10160).

**Three label sources, one per head:**

| Head | Source | Method |
|---|---|---|
| A. Removability | Summary alignment | LLM generates a summary per video; segments scored by max ROUGE-L / BERTScore against summary sentences. Top 30% → keep, bottom 30% → remove, middle → soft. Borrows from [TalkSumm (Lev et al., ACL 2019)](https://aclanthology.org/P19-1204/) |
| B. Function | Rule LFs + LLM zero-shot | High TF-IDF → `new_information`; high filler density → `discourse_filler`; etc. LLM resolves ambiguous segments. Aggregate via Snorkel `LabelModel` |
| C. Disfluency | Rules only | Disfluencies are surface-defined: filler word lists, exact-token repetition, edit-distance for revisions |

**Training procedure:**

1. Generate weak labels offline (one LLM summary per training+dev doc, then align segments).
2. Fine-tune `roberta-base`. AdamW, lr 2e-5, 3 epochs, batch 16. Class-weighted loss.
3. Early-stop on dev-set Head-A AUC against weak labels.
4. **Calibrate:** fit a single temperature `T` on dev minimizing ECE. Save `T` next to checkpoint.
5. **Critical invariant:** the gold set is touched only at evaluation. No early-stopping on it, no inspection during training.

---

## 9. Evaluation

Two tracks. Both stratified by genre and length bucket — that's how H1, H3, H4 get tested.

### 9.1 Segment-level (intrinsic, on the gold subset)

Metrics for the primary model + 6 baselines:

- F1 / Precision / Recall at calibrated 0.5
- AUC-ROC, AUC-PR
- ECE (Expected Calibration Error, 10 bins)
- Confidence-bucket precision at thresholds {0.95, 0.7}
- Annotator-confidence × model-confidence alignment — does annotator "definitely" map to model `p > 0.95`?

Output: one results table + one reliability diagram per model.

### 9.2 Transcript-level (extrinsic, on the 500-doc test set) — the headline

Sweep `p_remove` threshold from 0 → 1, compute four properties of the reduced transcript at each step:

1. **Compression ratio** (% words kept)
2. **SBERT cosine** (full vs reduced) — primary information-loss metric
3. **ROUGE-L** vs an LLM-generated summary of the full transcript
4. **QA preservation (QAGS-style)** — LLM generates ~10 factual questions from the full transcript; same LLM answers them from full *and* reduced; BERTScore between answer sets. [Wang et al., QAGS (ACL 2020)](https://aclanthology.org/2020.acl-main.450/).

**Headline figure:** compression-vs-QA-preservation curve, **one per genre**. Tests H1, H2, H3 in a single plot.

### 9.3 The 4-way agreement matrix (analysis, not training signal)

Compares four operationalizations of "removable" head-to-head:

| Definition | Signal |
|---|---|
| Model `p_remove` (primary) | Calibrated classifier output |
| Centroid-deviation | SBERT distance from doc centroid |
| Surface heuristics | Filler density + low TF-IDF + low NER |
| Human gold | Function ∈ {`redundant_repetition`, `discourse_filler`, `off_topic`} |

A 4×4 agreement matrix, computed per genre, becomes a central report table. Cases where definitions disagree are the most informative for error analysis.

### 9.4 Baselines (six)

| Tier | Baseline |
|---|---|
| Sanity | Random (biased coin matching positive rate) |
| Heuristic | Filler-word + repetition rule |
| Lexical | TF-IDF importance |
| Unsupervised neural | SBERT-Scorer (1 − cosine to centroid) |
| Long context | LongFormer-base (4096 tok) |
| Zero-shot | LLM-as-classifier (Claude Haiku, few-shot) |

---

## 10. Limitations

To be acknowledged explicitly in the report — not as apologies, but as scope.

1. **Transcript-only.** No audio prosody, no video frames, no speaker identity. A diagram referenced as "this graph" or a beat of silence used for emphasis is invisible to us. This affects H4 directly.
2. **ASR noise.** Auto-generated captions contain transcription errors that can look like disfluencies. We track this with the `caption_type` field but don't fully correct for it. Manual-vs-auto split is reported.
3. **Weak label noise.** Summary alignment occasionally marks an important detail as removable when the summary glosses over it. We expect noise; we measure model performance on *gold*, not on the noisy training labels.
4. **English only.** No multilingual claims. Generalization to other languages is unsupported.
5. **Annotation subjectivity.** Useful vs redundant repetition is a judgment call. Mitigated with the confidence layer + κ reporting, but never fully resolved.
6. **Genre coverage.** Three genres, channel-curated. We can't claim generalization to genres we didn't sample (news, sports, music vlogs, podcast interviews).
7. **Segment granularity.** Topic boundaries are derived from cosine-similarity dips — a heuristic. Different segmentation would shift results. Sentence-level ablation partially controls for this.
8. **Single-annotator risk.** If we can't recruit a co-annotator, κ is undefined and inter-annotator agreement claims are weakened.
9. **LLM evaluator dependency.** QAGS-style QA preservation uses an LLM to generate and answer questions. The metric inherits any biases of that LLM.

---

## 11. Improvements & What the Article Will Need

Notes on where this work can be sharpened — for the §Discussion and §Future Work sections of the write-up, and for prioritizing follow-up effort if time allows.

### Improvements to the current pipeline

1. **Annotator-confidence weighting in the loss.** Right now the model is trained on weak labels and *evaluated* against gold confidence. The next step is to fold gold confidence directly in — e.g. weight each gold-eval point by annotator certainty so calibration is judged on consensus cases more strictly than on borderline ones.
2. **Cross-segment coherence.** Segments are scored largely independently. A CRF or BiLSTM head over the segment sequence would penalize choices that produce jumpy cuts and break narrative flow. Cheap to add, likely improves transcript-level QA preservation.
3. **Gold expansion via the model itself.** Once calibrated, the model can pre-label thousands of segments; humans verify only its high-confidence outputs. An order-of-magnitude cheaper than de-novo labeling — and a clean Discussion-section result on its own.
4. **Beyond binary.** Replace Head A's sigmoid with a ranking objective and evaluate against NDCG / Spearman against gold. Better matches the actual user task ("which 30% should I cut?") than a binary cutoff.
5. **Listener-aware compression.** Personalize `p_remove` per user — a beginner and an expert want different content kept. Strong fit for the Discussion section as a thought experiment, even if not implemented.
6. **Multimodal extension.** Add audio prosody (pause length, pitch contour) and visual frames (slide changes). Most likely fixes the H4 failure cases — a clean follow-up paper hook.
7. **Multilingual transfer.** Swap `roberta-base` for `xlm-roberta`; test transfer to a non-English subset. Closes the original Turkish-corpus idea without reintroducing the translation confound.
8. **Real-time variant.** Current model is offline. A causal version that scores segments as they arrive would enable live caption summarization — natural extension for an applied follow-up.

### Datasets and precedents to cite (and possibly compare against) in the article

These each give us either a comparison point, a methodology to credit, or a robustness test.

| Dataset / paper | What it gives the article |
|---|---|
| [TalkSumm (Lev et al., ACL 2019)](https://aclanthology.org/P19-1204/) | The methodological precedent for our Head A weak labels. Cite as the inspiration for summary-alignment supervision. Their alignment scoring is a directly comparable baseline for us |
| [VT-SSum (2021)](https://arxiv.org/abs/2106.05606) | 9,096 video transcripts with segment + summary annotations. Closest direct precedent. Could serve as a held-out cross-domain test if time permits |
| [How2 (Sanabria et al., ACL 2019)](https://aclanthology.org/P19-1659.pdf) | ~80,000 instructional videos with transcripts and abstractive summaries. Good cross-domain robustness test — train on YouTube, evaluate on How2, report the gap |
| [TED-LIUM](https://huggingface.co/datasets/LIUM/tedlium) | TED transcripts with audio. Useful if we extend to multimodal, even just for the Future Work paragraph |
| [Miller, BERT extractive on lectures (2019)](https://arxiv.org/abs/1906.04165) | Most directly comparable architecture on lecture content. A natural baseline to cite in the Related Work section |
| [MCIF (FBK-MT, 2024)](https://huggingface.co/datasets/FBK-MT/MCIF) | Newer scientific-talk corpus with multimodal annotations. Cite as evidence the field is moving toward multimodal — supports our Limitations and Future Work framing |
| [SummEval (Fabbri et al., TACL 2021)](https://arxiv.org/abs/2007.12626) | Meta-evaluation of summarization metrics. Useful for justifying our metric choices (BERTScore + QAGS over ROUGE-only) |
| [AMI annotation schema](https://groups.inf.ed.ac.uk/ami/corpus/annotation.shtml) | Reference for our 6-class Function label set. Cite as the schema heritage; explicitly *don't* train on it (domain mismatch is its own discussion point) |

### Things to make sure end up in the article

- **An explicit definitions section.** §2 of the report should reproduce the table from §3 of this plan verbatim. The professor asked for this directly.
- **The 4-way agreement matrix as a figure** (§9.3), not just a table — visual disagreement patterns across genres are the most legible single result.
- **Reliability diagrams** (predicted vs actual probability) for the primary model and the strongest two baselines — the visual proof that calibration worked.
- **Per-genre, per-length stratified results everywhere.** Hypotheses H1 and H3 are *only* testable via stratified reporting; aggregate numbers hide the effect.
- **Caption-type robustness check.** Manual vs auto-caption split on the test set, reported separately. ASR noise is a known confound; showing it doesn't dominate is a credibility-builder.
- **A table of failure modes** drawn from §13 error analysis — segments where Model says remove but Gold says keep, with one annotated example per genre.
- **An honest cost ledger.** LLM call counts and dollar cost for the QA-preservation pipeline. Reproducibility detail reviewers ask for.
- **Reproducibility appendix.** Channel list, scrape date, splits file hashes, weak-label config, training seed, calibration `T`. Without these, no one can replicate, and reviewers will dock for it.

---

## 12. Open Decisions (before further coding)

1. **Encoder size.** `roberta-base` (125M) on local GPU vs `roberta-large` (355M) on Bocconi HPC. Default: base first.
2. **LLM for weak labels.** Claude Haiku 4.5 vs GPT-4o-mini. Estimated cost: ~$5–10 for 5000 summaries, ~$20–40 for QA evaluation.
3. **Gold size.** 200 segments default, expand to 300 if time allows.
4. **Co-annotator.** Two-annotator κ vs single-annotator with documented limitation.
5. **Third genre channels.** Pick from Lex Fridman / Huberman / Veritasium / 3Blue1Brown.
6. **Multi-task loss weights.** Default `1.0 / 0.5 / 0.5`, tunable on dev.

---

## 13. Reading List

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
- [AMI annotation schema](https://groups.inf.ed.ac.uk/ami/corpus/annotation.shtml) — used as reference, not as data

**Datasets hub:** [Awesome-Summarization-Datasets](https://github.com/edahanoam/Awesome-Summarization-Datasets).
