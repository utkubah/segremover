# Gold Evaluation

## Spearman ρ (model p_remove vs gold soft target)
- ρ = 0.5202  (p = 0.0000, n = 414)

## NDCG@20 (ranking quality)
- NDCG@20 = 0.8439

## Binary accuracy vs gold labels
| method       |   accuracy |     f1 |   n |
|:-------------|-----------:|-------:|----:|
| weak labels  |     0.7995 | 0.6719 | 414 |
| model @t=0.5 |     0.7415 | 0.5869 | 414 |
| model @t=0.7 |     0.7657 | 0.5268 | 414 |

## Head B vs human gold_function labels
- accuracy=0.4770  macro-F1=0.2406  (n=413)

| Function class | F1 vs human |
|---|---|
| new_information | 0.7236 |
| useful_repetition | 0.0645 |
| redundant_repetition | 0.0000 |
| clarification | 0.2600 |
| discourse_filler | 0.2419 |
| off_topic | 0.1538 |
