"""
src/baselines.py — Four baseline p_remove scorers for segremover.

Tier       Baseline    Signal
─────────────────────────────────────────────────────────────────
Sanity     random      Biased coin matching corpus positive rate
Heuristic  heuristic   Filler density + inverted TF-IDF rank (surface only)
Lexical    tfidf       1 − normalised TF-IDF sum within document
Neural     sbert       1 − cosine similarity to document centroid (all-MiniLM-L6-v2)

Output:
    data/processed/baselines.jsonl
    one row per segment: {video_id, genre, seg_idx, random, heuristic, tfidf, sbert}

AUC is evaluated against weak labels (p_remove > 0.5 as binary target),
matching the training-time dev metric in train.py.

Usage:
    python src/baselines.py
    python src/baselines.py --max-docs 10   # smoke test
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

SEGMENTS_PATH    = Path("data/processed/segments_topic.jsonl")
WEAK_LABELS_PATH = Path("data/processed/weak_labels.jsonl")
OUT_PATH         = Path("data/processed/baselines.jsonl")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Surface parameters (kept identical to weak_labels.py for consistency)
FILLER_WORDS = {
    "um", "uh", "uhh", "ah", "er", "hmm",
    "like", "basically", "literally", "actually", "okay", "right",
}
MULTIWORD_FILLERS = ["you know", "i mean", "sort of", "kind of"]
FILLER_THRESHOLD  = 0.03   # density that maps to score = 1.0


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(max_docs=None):
    """Return docs dict and weak-label lookup.

    docs  : {video_id: {genre, segments: [{seg_idx, text}]}}
    weak  : {(video_id, seg_idx): p_remove}
    """
    weak = {}
    with WEAK_LABELS_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            weak[(r["video_id"], r["seg_idx"])] = r["p_remove"]

    docs: dict[str, dict] = defaultdict(lambda: {"genre": None, "segments": []})
    with SEGMENTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            seg = json.loads(line)
            if (seg["video_id"], seg["seg_idx"]) not in weak:
                continue
            vid = seg["video_id"]
            docs[vid]["genre"] = seg["genre"]
            docs[vid]["segments"].append({"seg_idx": seg["seg_idx"], "text": seg["text"]})

    for d in docs.values():
        d["segments"].sort(key=lambda s: s["seg_idx"])

    video_ids = sorted(docs.keys())
    if max_docs:
        video_ids = video_ids[:max_docs]

    return {v: docs[v] for v in video_ids}, weak


# ── Baseline 1: random ────────────────────────────────────────────────────────

def baseline_random(docs, weak, seed=42):
    """Biased coin: P(remove) drawn from Uniform[0,1], threshold at pos_rate for AUC."""
    rng = random.Random(seed)
    all_p = list(weak.values())
    pos_rate = sum(1 for p in all_p if p > 0.5) / len(all_p)  # noqa: F841 (kept for reference)

    scores: dict[tuple, float] = {}
    for vid, doc in docs.items():
        for seg in doc["segments"]:
            scores[(vid, seg["seg_idx"])] = rng.random()
    return scores


# ── Baseline 2: heuristic ────────────────────────────────────────────────────

def _filler_score(text: str) -> float:
    """Filler density normalised to [0, 1] (saturates at FILLER_THRESHOLD)."""
    words = text.lower().split()
    if not words:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for w in words if w.strip(".,!?;") in FILLER_WORDS)
    hits += sum(text_lower.count(p) * len(p.split()) for p in MULTIWORD_FILLERS)
    density = hits / len(words)
    return min(density / FILLER_THRESHOLD, 1.0)


def baseline_heuristic(docs):
    """Average of filler score and inverted TF-IDF rank (no embeddings needed)."""
    scores: dict[tuple, float] = {}

    for vid, doc in docs.items():
        texts = [s["text"] for s in doc["segments"]]
        seg_idxs = [s["seg_idx"] for s in doc["segments"]]

        # TF-IDF rank within document (low TF-IDF sum → high p_remove)
        if len(texts) >= 2:
            try:
                tfidf_sums = (
                    TfidfVectorizer(stop_words="english")
                    .fit_transform(texts)
                    .toarray()
                    .sum(axis=1)
                )
                # invert: lowest TF-IDF → score 1.0
                inv_tfidf = 1.0 - minmax_scale(tfidf_sums)
            except ValueError:
                inv_tfidf = np.full(len(texts), 0.5)
        else:
            inv_tfidf = np.full(len(texts), 0.5)

        for seg, si in zip(doc["segments"], range(len(texts))):
            filler = _filler_score(seg["text"])
            score  = 0.5 * filler + 0.5 * float(inv_tfidf[si])
            scores[(vid, seg["seg_idx"])] = round(score, 4)

    return scores


# ── Baseline 3: tfidf ─────────────────────────────────────────────────────────

def baseline_tfidf(docs):
    """1 − normalised TF-IDF sum within document (pure lexical signal)."""
    scores: dict[tuple, float] = {}

    for vid, doc in docs.items():
        texts = [s["text"] for s in doc["segments"]]
        if len(texts) < 2:
            for seg in doc["segments"]:
                scores[(vid, seg["seg_idx"])] = 0.5
            continue
        try:
            tfidf_sums = (
                TfidfVectorizer(stop_words="english")
                .fit_transform(texts)
                .toarray()
                .sum(axis=1)
            )
            inv_scores = 1.0 - minmax_scale(tfidf_sums)
        except ValueError:
            inv_scores = np.full(len(texts), 0.5)

        for seg, score in zip(doc["segments"], inv_scores):
            scores[(vid, seg["seg_idx"])] = round(float(score), 4)

    return scores


# ── Baseline 4: sbert ─────────────────────────────────────────────────────────

def baseline_sbert(docs, device="cpu"):
    """1 − cosine similarity of each segment to its document centroid."""
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(EMBED_MODEL, device=device)
    scores: dict[tuple, float] = {}

    for vid, doc in tqdm(docs.items(), desc="sbert baseline"):
        texts = [s["text"] for s in doc["segments"]]
        embs  = embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        centroid = embs.mean(axis=0)
        # Normalise centroid for cosine via dot product
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm
        sims = embs @ centroid   # cosine similarity (embeddings already L2-normed)

        for seg, sim in zip(doc["segments"], sims):
            scores[(vid, seg["seg_idx"])] = round(float(1.0 - sim), 4)

    return scores


# ── AUC evaluation ────────────────────────────────────────────────────────────

def evaluate_auc(name, scores, weak, docs):
    """AUC-ROC of score vs binarised weak label (p_remove > 0.5)."""
    preds, targets = [], []
    for vid, doc in docs.items():
        for seg in doc["segments"]:
            key = (vid, seg["seg_idx"])
            if key in scores and key in weak:
                preds.append(scores[key])
                targets.append(1 if weak[key] > 0.5 else 0)
    if len(set(targets)) < 2:
        print(f"  {name:12s}  AUC=N/A (single class in data)")
        return
    auc = roc_auc_score(targets, preds)
    print(f"  {name:12s}  AUC={auc:.4f}  (n={len(preds):,})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run all baselines for segremover")
    parser.add_argument("--max-docs", type=int, default=None, dest="max_docs",
                        help="Limit to N documents (smoke test)")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading data ...")
    docs, weak = load_data(args.max_docs)
    n_segs = sum(len(d["segments"]) for d in docs.values())
    print(f"  {len(docs)} documents, {n_segs:,} segments")

    # ── Run baselines ──────────────────────────────────────────────────────────
    print("\nRunning baselines ...")

    print("  [1/4] random ...")
    rand_scores  = baseline_random(docs, weak, seed=args.seed)

    print("  [2/4] heuristic ...")
    heur_scores  = baseline_heuristic(docs)

    print("  [3/4] tfidf ...")
    tfidf_scores = baseline_tfidf(docs)

    print("  [4/4] sbert ...")
    sbert_scores = baseline_sbert(docs, device=device)

    # ── AUC report ────────────────────────────────────────────────────────────
    print("\nAUC-ROC vs weak labels (p_remove > 0.5):")
    for name, sc in [
        ("random",    rand_scores),
        ("heuristic", heur_scores),
        ("tfidf",     tfidf_scores),
        ("sbert",     sbert_scores),
    ]:
        evaluate_auc(name, sc, weak, docs)

    # ── Write output ──────────────────────────────────────────────────────────
    out_path = (OUT_PATH.parent / (OUT_PATH.stem + "_smoke" + OUT_PATH.suffix)
                if args.max_docs else OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for vid, doc in docs.items():
            for seg in doc["segments"]:
                key = (vid, seg["seg_idx"])
                f.write(json.dumps({
                    "video_id": vid,
                    "genre":    doc["genre"],
                    "seg_idx":  seg["seg_idx"],
                    "random":   rand_scores.get(key),
                    "heuristic": heur_scores.get(key),
                    "tfidf":    tfidf_scores.get(key),
                    "sbert":    sbert_scores.get(key),
                }, ensure_ascii=False) + "\n")

    print(f"\nWrote {n_segs:,} rows to {out_path}")


if __name__ == "__main__":
    main()
