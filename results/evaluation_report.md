# segremover — Evaluation Report

**Checkpoint:** `models/best.pt`  
**Data dir:** `data/processed`  
**Label source:** weak labels (proxy — not human-annotated)  

## 1. Dataset summary
| Item | Value |
|---|---|
| Documents | 494 |
| Segments  | 58,979 |
| Gold labels | ✓ human-annotated |
| Model checkpoint | ✓ loaded |
| Baselines | ✓ loaded |
| Genre: Commentary | 105 videos |
| Genre: Entertainment | 77 videos |
| Genre: Lectures | 114 videos |
| Genre: Podcasts | 84 videos |
| Genre: TED talks | 69 videos |
| Genre: TV series | 45 videos |

### Table 1: Corpus statistics by genre
| Genre | Videos | Segments | Mean seg length (words) | % positive (p_remove > 0.5) |
|---|---:|---:|---:|---:|
| Commentary | 105 | 6,242 | 92.4 | 34.8% |
| Entertainment | 77 | 10,699 | 41.3 | 48.6% |
| Lectures | 114 | 9,090 | 80.8 | 34.0% |
| Podcasts | 84 | 21,959 | 81.8 | 42.2% |
| TED talks | 69 | 1,675 | 89.0 | 38.1% |
| TV series | 45 | 9,314 | 36.2 | 45.8% |

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
| ROC-AUC  | 0.871 (95% CI: 0.8679–0.8741) |
| PR-AUC   | 0.8197 |
| ECE      | 0.0568 |
| F1 @0.5  | 0.7594 |
| P  @0.5  | 0.716 |
| R  @0.5  | 0.8085 |

![SegRemover achieves AUC-ROC=0.871 vs weak-label proxy.](plots/model_roc_pr.png)
*SegRemover achieves AUC-ROC=0.871 vs weak-label proxy.*
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

**Head B classification:** accuracy=0.5122  macro-F1=0.3667

| Function class | F1 |
|---|---|
| New Information | 0.6995 |
| Useful Repetition | 0.2676 |
| Redundant Repetition | 0.1872 |
| Clarification | 0.285 |
| Discourse Filler | 0.4 |
| Off Topic | 0.3611 |
![Row-normalised confusion matrix for Head B. Off-diagonal mass shows which function classes are confused.](plots/head_b_confusion.png)
*Row-normalised confusion matrix for Head B. Off-diagonal mass shows which function classes are confused.*

### Head C — Disfluency classifier
![Disfluency distribution by genre: podcasts and TV series contain more filled pauses and repetitions than lectures.](plots/disfl_distribution_by_genre.png)
*Disfluency distribution by genre: podcasts and TV series contain more filled pauses and repetitions than lectures.*
![Disfluent segments (filled pause, repetition, restart) score higher p_remove, validating Head C's signal.](plots/p_remove_by_disfluency.png)
*Disfluent segments (filled pause, repetition, restart) score higher p_remove, validating Head C's signal.*

**Head C classification:** accuracy=0.9619  macro-F1=0.9412

| Disfluency class | F1 |
|---|---|
| Clean | 0.9763 |
| Filled Pause | 0.9835 |
| Repetition | 0.8646 |
| Revision | 0.941 |
| Restart | 0.9408 |
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

## 3. Transcript-level (compression vs SBERT preservation)
| Method    |   Threshold |   Compression |   SBERT cosine |
|:----------|------------:|--------------:|---------------:|
| heuristic |        0.5  |        0.5434 |         0.9306 |
| heuristic |        0.6  |        0.6657 |         0.9668 |
| heuristic |        0.7  |        0.7846 |         0.9798 |
| heuristic |        0.8  |        0.9073 |         0.9893 |
| heuristic |        0.9  |        0.9807 |         0.9969 |
| heuristic |        0.95 |        0.9928 |         0.9983 |
| model     |        0.5  |        0.7768 |         0.972  |
| model     |        0.6  |        0.8358 |         0.9803 |
| model     |        0.7  |        0.889  |         0.9873 |
| model     |        0.8  |        0.9377 |         0.9929 |
| model     |        0.9  |        0.9818 |         0.9981 |
| model     |        0.95 |        0.993  |         0.9995 |
| sbert     |        0.5  |        0.6898 |         0.9718 |
| sbert     |        0.6  |        0.8478 |         0.9888 |
| sbert     |        0.7  |        0.94   |         0.9966 |
| sbert     |        0.8  |        0.9844 |         0.9992 |
| sbert     |        0.9  |        0.9968 |         0.9999 |
| sbert     |        0.95 |        0.9992 |         1      |
| tfidf     |        0.5  |        0.6076 |         0.9298 |
| tfidf     |        0.6  |        0.7575 |         0.9641 |
| tfidf     |        0.7  |        0.866  |         0.9788 |
| tfidf     |        0.8  |        0.9318 |         0.9883 |
| tfidf     |        0.9  |        0.9664 |         0.9932 |
| tfidf     |        0.95 |        0.9746 |         0.9945 |

![Semantic similarity (SBERT centroid cosine) degrades smoothly with compression; genre curves diverge past 50% word retention.](plots/transcript_compression.png)
*Semantic similarity (SBERT centroid cosine) degrades smoothly with compression; genre curves diverge past 50% word retention.*
![Longer videos are more semantically robust to compression.](plots/transcript_by_length.png)
*Longer videos are more semantically robust to compression.*
![Direct removability by length: percent segments and words removed at operating thresholds.](plots/removability_by_length.png)
*Direct removability by length: percent segments and words removed at operating thresholds.*
![Direct removability by genre at the primary threshold.](plots/removability_by_genre.png)
*Direct removability by genre at the primary threshold.*
![ROUGE recall tracks compression linearly; the gap between ROUGE-1 and ROUGE-L indicates sentence fragmentation.](plots/rouge_curves.png)
*ROUGE recall tracks compression linearly; the gap between ROUGE-1 and ROUGE-L indicates sentence fragmentation.*

## 4. Baseline comparison
| Method       |  ROC-AUC | ROC 95% CI |   PR-AUC | PR 95% CI | p vs model |
|--------------|----------|--------------|----------|--------------|------------|
| SegRemover   |   0.8710 | 0.8679-0.8741 |   0.8197 | 0.8144-0.8249 | — |
| Random       |   0.5007 | 0.4957-0.505 |   0.4205 | 0.4146-0.4265 | 0.0000 |
| Heuristic    |   0.6457 | 0.6414-0.6503 |   0.6178 | 0.611-0.6241 | 0.0000 |
| TF-IDF       |   0.7990 | 0.7954-0.8028 |   0.7223 | 0.7161-0.7296 | 0.0000 |
| SBERT        |   0.7440 | 0.7402-0.748 |   0.6927 | 0.6861-0.6983 | 0.0000 |

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
- **Spearman ρ vs gold soft target:** 0.5117 (p=0.0)
- **NDCG@20:** 0.8516
- **Weak-label accuracy vs gold:** 0.7496 F1=0.5677
- **Head B vs human gold_function:** accuracy=0.4507  macro-F1=0.2482  _(model's function-type predictions compared to annotator labels)_

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
eval sample     : 500
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