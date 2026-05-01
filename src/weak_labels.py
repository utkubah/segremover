"""
Generate weak labels for all topic segments.

Runs 6 local labeling functions (LFs) over segments_topic.jsonl and
aggregates their votes into a soft p_remove score per segment using
Snorkel's LabelModel.

LFs implemented here (no LLM required):
    LF_filler          filler density > 0.05 → remove
    LF_repetition      max cosine to any earlier segment in doc > 0.85 → remove
    LF_short           < 5 content words → remove
    LF_low_tfidf       bottom-20% TF-IDF within document → remove
    LF_named_entity    ≥ 2 named entities (spaCy NER) → keep
    LF_question_or_def contains "?" or definition phrase → keep

LF_summary_align (LLM-based) is not implemented here.
Run src/summary_align.py first if you want to include it, then re-run.

Output:
    data/processed/weak_labels.jsonl
    one line per segment: {video_id, genre, seg_idx, p_remove, votes}

Usage:
    python src/weak_labels.py
    python src/weak_labels.py --max-docs 10   # smoke test
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from snorkel.labeling.model import LabelModel
from tqdm import tqdm

SEGMENTS_PATH = Path("data/processed/segments_topic.jsonl")
SUMMARY_ALIGN_PATH = Path("data/processed/summary_align_votes.jsonl")
OUT_PATH = Path("data/processed/weak_labels.jsonl")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Snorkel label constants
ABSTAIN = -1
KEEP = 0
REMOVE = 1

LF_NAMES = [
    "LF_filler",
    "LF_repetition",
    "LF_short",
    "LF_low_tfidf",
    "LF_named_entity",
    "LF_question_or_def",
]

# ── LF configuration ───────────────────────────────────────────────────────────

FILLER_WORDS = {
    "um", "uh", "uhh", "ah", "er", "hmm",
    "like", "basically", "literally", "actually", "okay", "right",
}
MULTIWORD_FILLERS = ["you know", "i mean", "sort of", "kind of"]

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "or", "and", "but", "not",
    "this", "that", "it", "its", "i", "you", "we", "they", "he", "she",
}

DEFINITION_PHRASES = ["is defined as", "what is", "refers to", "means that"]

FILLER_THRESHOLD = 0.03        # lowered: catch more filler-heavy segments
REPETITION_THRESHOLD = 0.70    # lowered: catch paraphrased repetition, not just verbatim
TFIDF_REMOVE_PERCENTILE = 20
MIN_CONTENT_WORDS = 5
MIN_NAMED_ENTITIES = 3         # raised: 2 NEs is too common, 3 is more selective


# ── Per-segment LFs ────────────────────────────────────────────────────────────

def lf_filler(text: str) -> int:
    words = text.lower().split()
    if not words:
        return ABSTAIN
    text_lower = text.lower()
    hits = sum(1 for w in words if w in FILLER_WORDS)
    hits += sum(text_lower.count(p) * len(p.split()) for p in MULTIWORD_FILLERS)
    return REMOVE if hits / len(words) > FILLER_THRESHOLD else ABSTAIN


def lf_short(text: str) -> int:
    content = [w for w in text.lower().split() if w not in STOPWORDS and w.isalpha()]
    return REMOVE if len(content) < MIN_CONTENT_WORDS else ABSTAIN


def lf_question_or_def(text: str) -> int:
    text_lower = text.lower()
    if "?" in text:
        return KEEP
    if any(p in text_lower for p in DEFINITION_PHRASES):
        return KEEP
    return ABSTAIN


def lf_named_entity(text: str, nlp) -> int:
    return KEEP if len(nlp(text).ents) >= MIN_NAMED_ENTITIES else ABSTAIN


# ── Per-document LFs (need all segments together) ─────────────────────────────

def lf_repetition_votes(embeddings: np.ndarray) -> list[int]:
    """
    For each segment i, check cosine similarity against every earlier
    segment j < i in the same document. Embeddings must be L2-normalised
    (cosine = dot product when normalised).
    """
    votes = [ABSTAIN] * len(embeddings)
    for i in range(1, len(embeddings)):
        sims = embeddings[:i] @ embeddings[i]
        if float(sims.max()) > REPETITION_THRESHOLD:
            votes[i] = REMOVE
    return votes


def lf_tfidf_votes(texts: list[str]) -> list[int]:
    """Segments in the bottom TFIDF_REMOVE_PERCENTILE % by TF-IDF sum → remove."""
    if len(texts) < 5:
        return [ABSTAIN] * len(texts)
    try:
        scores = TfidfVectorizer(stop_words="english").fit_transform(texts).toarray().sum(axis=1)
    except ValueError:
        return [ABSTAIN] * len(texts)
    threshold = np.percentile(scores, TFIDF_REMOVE_PERCENTILE)
    return [REMOVE if s <= threshold else ABSTAIN for s in scores]


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate(L: np.ndarray) -> np.ndarray:
    """
    Fit Snorkel's LabelModel on the vote matrix and return p_remove per segment.
    Falls back to simple majority vote if LabelModel fails.
    """
    try:
        lm = LabelModel(cardinality=2, verbose=False)
        lm.fit(L_train=L, n_epochs=300, seed=42, log_freq=100)
        return lm.predict_proba(L)[:, REMOVE]
    except Exception as e:
        print(f"  LabelModel failed ({e}) — falling back to majority vote")
        p = np.zeros(len(L))
        for i, row in enumerate(L):
            active = row[row != ABSTAIN]
            p[i] = float((active == REMOVE).mean()) if len(active) else 0.5
        return p


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate weak labels for topic segments")
    parser.add_argument("--max-docs", type=int, help="Limit to N documents (smoke test)")
    args = parser.parse_args()

    # Load segments grouped by video_id
    print(f"Loading segments from {SEGMENTS_PATH} ...")
    docs: dict[str, list[dict]] = defaultdict(list)
    with SEGMENTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            seg = json.loads(line)
            docs[seg["video_id"]].append(seg)

    for vid in docs:
        docs[vid].sort(key=lambda s: s["seg_idx"])

    video_ids = list(docs.keys())
    if args.max_docs:
        video_ids = video_ids[:args.max_docs]

    total_segs = sum(len(docs[v]) for v in video_ids)
    print(f"Processing {len(video_ids)} documents ({total_segs:,} segments) ...")

    print("Loading spaCy (en_core_web_sm) ...")
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "lemmatizer", "attribute_ruler"])

    print(f"Loading sentence-embedding model ({EMBED_MODEL}) ...")
    embedder = SentenceTransformer(EMBED_MODEL)

    # Load summary-alignment votes if available (generated by summary_align.py)
    summary_align: dict[tuple, int] = {}
    if SUMMARY_ALIGN_PATH.exists():
        with SUMMARY_ALIGN_PATH.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                summary_align[(row["video_id"], row["seg_idx"])] = row["lf_summary_align"]
        print(f"Loaded LF_summary_align votes for {len(summary_align):,} segments")
    else:
        print(f"LF_summary_align not found — run src/summary_align.py to add it")

    use_summary_align = bool(summary_align)
    lf_names = LF_NAMES + ["LF_summary_align"] if use_summary_align else LF_NAMES

    all_segs: list[dict] = []
    all_votes: list[list[int]] = []

    for vid in tqdm(video_ids, desc="labeling"):
        segs = docs[vid]
        texts = [s["text"] for s in segs]

        # Encode all segments for this document at once
        embeds = embedder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )

        rep_votes = lf_repetition_votes(embeds)
        tfidf_votes = lf_tfidf_votes(texts)

        for i, seg in enumerate(segs):
            text = seg["text"]
            votes = [
                lf_filler(text),
                rep_votes[i],
                lf_short(text),
                tfidf_votes[i],
                lf_named_entity(text, nlp),
                lf_question_or_def(text),
            ]
            if use_summary_align:
                votes.append(summary_align.get((seg["video_id"], seg["seg_idx"]), ABSTAIN))
            all_segs.append(seg)
            all_votes.append(votes)

    L = np.array(all_votes, dtype=int)

    print("\nLF coverage report:")
    for j, name in enumerate(lf_names):
        col = L[:, j]
        n_remove = int((col == REMOVE).sum())
        n_keep = int((col == KEEP).sum())
        n_abstain = int((col == ABSTAIN).sum())
        pct = 100 * (1 - n_abstain / len(col))
        print(f"  {name:25s}  remove={n_remove:5d}  keep={n_keep:5d}  abstain={n_abstain:5d}  coverage={pct:.1f}%")

    print("\nAggregating with Snorkel LabelModel ...")
    p_remove = aggregate(L)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for seg, votes, p in zip(all_segs, all_votes, p_remove):
            f.write(json.dumps({
                "video_id": seg["video_id"],
                "genre": seg["genre"],
                "seg_idx": seg["seg_idx"],
                "p_remove": round(float(p), 4),
                "votes": dict(zip(lf_names, [int(v) for v in votes])),
            }, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_segs):,} weak labels to {OUT_PATH}")
    print(f"Mean p_remove = {p_remove.mean():.3f}  (std = {p_remove.std():.3f})")
    print(f"  p > 0.70  (probably remove): {(p_remove > 0.70).sum():,}")
    print(f"  0.30–0.70 (uncertain):        {((p_remove >= 0.30) & (p_remove <= 0.70)).sum():,}")
    print(f"  p < 0.30  (probably keep):    {(p_remove < 0.30).sum():,}")


if __name__ == "__main__":
    main()
