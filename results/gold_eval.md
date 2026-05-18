# Gold Evaluation

## Spearman ρ and NDCG@20 vs gold soft target (all methods)
| method     |      ρ |   p-value |   NDCG@20 |    n |
|:-----------|-------:|----------:|----------:|-----:|
| SegRemover | 0.5117 |    0      |    0.8516 | 1989 |
| tfidf      | 0.4223 |    0      |    0.7121 | 1989 |
| sbert      | 0.4684 |    0      |    0.9688 | 1989 |
| heuristic  | 0.3395 |    0      |    0.6203 | 1989 |
| random     | 0.0378 |    0.0921 |    0.2708 | 1989 |

## Binary accuracy vs gold labels
| method       |   accuracy |     f1 |    n |
|:-------------|-----------:|-------:|-----:|
| weak labels  |     0.7496 | 0.5677 | 1989 |
| model @t=0.5 |     0.7169 | 0.5192 | 1989 |
| model @t=0.7 |     0.7733 | 0.4927 | 1989 |
| tfidf        |     0.6018 | 0.4605 | 1989 |
| sbert        |     0.6461 | 0.5204 | 1989 |
| heuristic    |     0.5596 | 0.4057 | 1989 |
| random       |     0.5063 | 0.3311 | 1989 |

## Head B vs human gold_function labels
- accuracy=0.4507  macro-F1=0.2482  (n=1988)

| Function class | F1 vs human |
|---|---|
| new_information | 0.6881 |
| useful_repetition | 0.0500 |
| redundant_repetition | 0.0365 |
| clarification | 0.2604 |
| discourse_filler | 0.2495 |
| off_topic | 0.2049 |
