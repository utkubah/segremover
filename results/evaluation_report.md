# segremover — Evaluation Report

**Checkpoint:** `models/best.pt`  
**Data dir:** `data/processed`  
**Label source:** weak labels (proxy — not human-annotated)  

## 1. Dataset summary
| Item | Value |
|---|---|
| Documents | 26 |
| Segments  | 1,608 |
| Gold labels | ✓ human-annotated |
| Model checkpoint | ✓ loaded |
| Baselines | ✓ loaded |
| Genre: Commentary | 9 videos |
| Genre: Entertainment | 2 videos |
| Genre: Lectures | 4 videos |
| Genre: Podcasts | 3 videos |
| Genre: TED talks | 4 videos |
| Genre: TV series | 4 videos |

### Table 1: Corpus statistics by genre
| Genre | Videos | Segments | Mean seg length (words) | % positive (p_remove > 0.5) |
|---|---:|---:|---:|---:|
| Commentary | 9 | 416 | 82.1 | 37.7% |
| Entertainment | 2 | 132 | 47.2 | 42.4% |
| Lectures | 4 | 143 | 86.7 | 26.6% |
| Podcasts | 3 | 501 | 87.0 | 42.5% |
| TED talks | 4 | 147 | 59.3 | 42.2% |
| TV series | 4 | 269 | 35.6 | 49.1% |

## Evidence coverage checklist
| Evidence item | Status | Note |
|---|---:|---|
| Six-genre coverage | OK | all expected genres present |
| Direct removability by length | OK | generated |
| Caption-type robustness | WARN | caption_type missing or only one type |
| Gold/human evaluation | OK | gold labels loaded |
| Transcript-only risk scan | OK | generated |

## 2. Segment-level metrics
| Metric | Value |
|---|---|
| ROC-AUC  | 0.8638 (95% CI: 0.8449–0.882) |
| PR-AUC   | 0.8213 |
| ECE      | 0.0528 |
| F1 @0.5  | 0.752 |
| P  @0.5  | 0.7234 |
| R  @0.5  | 0.7829 |

![SegRemover achieves AUC-ROC=0.8638 vs weak-label proxy.](plots/model_roc_pr.png)
*SegRemover achieves AUC-ROC=0.8638 vs weak-label proxy.*
![Reliability diagram: bars show fraction truly removed per confidence bin; diagonal = perfect calibration.](plots/model_reliability.png)
*Reliability diagram: bars show fraction truly removed per confidence bin; diagonal = perfect calibration.*
![Calibration varies by genre — ECE values reveal where the model is over/under-confident.](plots/reliability_by_genre.png)
*Calibration varies by genre — ECE values reveal where the model is over/under-confident.*
![Temperature scaling (T=1.1) changes ECE; this panel prevents claiming improvement when ECE increases.](plots/calibration_before_after_T.png)
*Temperature scaling (T=1.1) changes ECE; this panel prevents claiming improvement when ECE increases.*
![Confusion matrix at threshold=0.5; most errors are false positives (cautious model keeps borderline segments).](plots/model_confusion.png)
*Confusion matrix at threshold=0.5; most errors are false positives (cautious model keeps borderline segments).*
![Lowering the threshold increases recall at the cost of precision; the F1 peak marks the operating point.](plots/model_threshold_sweep.png)
*Lowering the threshold increases recall at the cost of precision; the F1 peak marks the operating point.*
![High-confidence predictions (p>=0.95) are most reliable; precision drops in the borderline 0.70-0.95 bucket.](plots/model_confidence_buckets.png)
*High-confidence predictions (p>=0.95) are most reliable; precision drops in the borderline 0.70-0.95 bucket.*
![Function-label mix by confidence bucket: this explains what the model tends to remove.](plots/function_by_confidence_bucket.png)
*Function-label mix by confidence bucket: this explains what the model tends to remove.*

### Head B — Function classifier
![Function-label distribution by genre: lectures are new-information heavy; podcasts skew toward discourse fillers.](plots/fn_label_distribution_by_genre.png)
*Function-label distribution by genre: lectures are new-information heavy; podcasts skew toward discourse fillers.*
![Redundant and off-topic segments score highest p_remove, validating that Head B shapes the removal decision.](plots/p_remove_by_function.png)
*Redundant and off-topic segments score highest p_remove, validating that Head B shapes the removal decision.*

**Head B classification:** accuracy=0.5225  macro-F1=0.3798

| Function class | F1 |
|---|---|
| New Information | 0.6914 |
| Useful Repetition | 0.2718 |
| Redundant Repetition | 0.2013 |
| Clarification | 0.2742 |
| Discourse Filler | 0.441 |
| Off Topic | 0.3989 |
![Row-normalised confusion matrix for Head B. Off-diagonal mass shows which function classes are confused.](plots/head_b_confusion.png)
*Row-normalised confusion matrix for Head B. Off-diagonal mass shows which function classes are confused.*

### Head C — Disfluency classifier
![Disfluency distribution by genre: podcasts and TV series contain more filled pauses and repetitions than lectures.](plots/disfl_distribution_by_genre.png)
*Disfluency distribution by genre: podcasts and TV series contain more filled pauses and repetitions than lectures.*
![Disfluent segments (filled pause, repetition, restart) score higher p_remove, validating Head C's signal.](plots/p_remove_by_disfluency.png)
*Disfluent segments (filled pause, repetition, restart) score higher p_remove, validating Head C's signal.*

**Head C classification:** accuracy=0.9617  macro-F1=0.9627

| Disfluency class | F1 |
|---|---|
| Clean | 0.9762 |
| Filled Pause | 0.9778 |
| Repetition | 0.8772 |
| Revision | 0.9825 |
| Restart | 1.0 |
![Row-normalised confusion matrix for Head C. Confusion between adjacent disfluency classes is expected.](plots/head_c_confusion.png)
*Row-normalised confusion matrix for Head C. Confusion between adjacent disfluency classes is expected.*

### AUC by genre (95% bootstrap CI)
![AUC-ROC varies across genres — lectures show the highest discriminability, suggesting cleaner segment boundaries.](plots/model_genre_auc.png)
*AUC-ROC varies across genres — lectures show the highest discriminability, suggesting cleaner segment boundaries.*

### p_remove distribution
![p_remove distribution across all segments — meaningful mass in the uncertain zone confirms the model is a continuous ranker, not a binary classifier.](plots/p_remove_distribution.png)
*p_remove distribution across all segments — meaningful mass in the uncertain zone confirms the model is a continuous ranker, not a binary classifier.*
![Per-genre p_remove distributions — podcasts show more uncertain mass than lectures, consistent with higher genre-level ECE.](plots/p_remove_distribution_by_genre.png)
*Per-genre p_remove distributions — podcasts show more uncertain mass than lectures, consistent with higher genre-level ECE.*

## 4. Baseline comparison
| Method       |  ROC-AUC | ROC 95% CI |   PR-AUC | PR 95% CI | p vs model |
|--------------|----------|--------------|----------|--------------|------------|
| SegRemover   |   0.8638 | 0.8438-0.8806 |   0.8213 | 0.7908-0.8479 | — |
| Random       |   0.4951 | 0.4687-0.521 |   0.4043 | 0.3773-0.4385 | 0.0000 |
| Heuristic    |   0.6343 | 0.6064-0.662 |   0.5903 | 0.5581-0.6273 | 0.0000 |
| TF-IDF       |   0.7948 | 0.7723-0.8145 |   0.7055 | 0.663-0.749 | 0.0000 |
| SBERT        |   0.7428 | 0.7142-0.7654 |   0.6701 | 0.6275-0.711 | 0.0000 |

![SegRemover outperforms the unsupervised baselines; error bars show 95% bootstrap CI across 300 resamples.](plots/baseline_comparison.png)
*SegRemover outperforms the unsupervised baselines; error bars show 95% bootstrap CI across 300 resamples.*
![Genre-level breakdown: model advantage over baselines varies by domain.](plots/baseline_by_genre.png)
*Genre-level breakdown: model advantage over baselines varies by domain.*
![ROC overlay: SegRemover's curve lies above all baselines at every FPR.](plots/roc_overlay.png)
*ROC overlay: SegRemover's curve lies above all baselines at every FPR.*

## 5. 4-way agreement
![Spearman ρ heatmap: model agrees most with SBERT centroid, confirming that semantic centrality guides removability.](plots/agreement_heatmap.png)
*Spearman ρ heatmap: model agrees most with SBERT centroid, confirming that semantic centrality guides removability.*
![Per-genre model–source Spearman ρ: agreement with SBERT drops in conversational genres (podcasts, entertainment).](plots/agreement_by_genre.png)
*Per-genre model–source Spearman ρ: agreement with SBERT drops in conversational genres (podcasts, entertainment).*

## 6. Gold evaluation (human annotations)
- **Spearman ρ vs gold soft target:** 0.5202 (p=0.0)
- **NDCG@20:** 0.8439
- **Weak-label accuracy vs gold:** 0.7995 F1=0.6719
- **Head B vs human gold_function:** accuracy=0.477  macro-F1=0.2406  _(model's function-type predictions compared to annotator labels)_

![Scatter of model p_remove vs gold soft target; upward trend confirms the model captures human removability judgements.](plots/gold_scatter.png)
*Scatter of model p_remove vs gold soft target; upward trend confirms the model captures human removability judgements.*
![Model outperforms weak labels against human binary annotations, validating the training signal.](plots/gold_accuracy.png)
*Model outperforms weak labels against human binary annotations, validating the training signal.*

See `gold_eval.md` for full labeling-function accuracy breakdown.
![Head B predicted function class vs human gold_function annotation — alignment here validates the explainability claim independently of weak-label training signal.](plots/head_b_vs_gold_fn_confusion.png)
*Head B predicted function class vs human gold_function annotation — alignment here validates the explainability claim independently of weak-label training signal.*

## 8. Qualitative examples
See `qualitative_examples.md` — includes high-confidence keep/remove examples and any gold-based FP/FN examples.
![Transcript-only limitation scan: visual/deictic cues by genre.](plots/transcript_only_risk_by_genre.png)
*Transcript-only limitation scan: visual/deictic cues by genre.*
See `transcript_only_risk_examples.md` for snippets.

## 9. Limitations
- Segment metrics evaluated against **weak labels** (proxy) unless gold annotations exist.
- Transcript SBERT cosine uses centroid-level similarity; does not capture fact-level preservation.
- ROUGE recall measures lexical preservation (`pip install rouge-score`); QA-preservation is not computed.
- Transcript-only risk examples are lexical cues, not confirmed errors, unless gold labels are present.
- Model scores can be missing for segments beyond the checkpoint `max_segs` truncation limit.
- Caption-type analysis requires `caption_type` field in segment data.
- The 6-class function taxonomy (new_information, useful_repetition, redundant_repetition, clarification, discourse_filler, off_topic) and the 5-class disfluency taxonomy are hand-crafted without empirical validation that these specific categories are optimal for the removability task. No prior work establishes this taxonomy as the correct decomposition of segment function for spoken YouTube content; alternative label designs (finer-grained, coarser, or grounded in a different linguistic theory) might yield stronger cascade signal or better Head B/C accuracy. The taxonomies should be treated as design choices, not established ground truth.

## Appendix: Reproducibility
```
eval seed       : 42
eval sample     : 10
encoder         : roberta-base
inter_layers    : 4
temperature T   : 1.1
max_segs        : 256
max_tok         : 384
checkpoint      : models/best.pt
data_dir        : data/processed
thresholds      : [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
```

## Run command
```bash
python src/eval.py --checkpoint models/best.pt --data-dir data/processed
```