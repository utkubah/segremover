"""
Segment raw transcripts into the units the model will classify.

Two views are produced:
  1. Sentence-level segments — one sentence per row. Cheap, deterministic,
     used as a baseline / ablation in modeling.
  2. Topic-level segments — semantic boundary detection using sentence
     embeddings. Adjacent-sentence cosine similarity drops indicate topic
     shifts; we cut at local minima.

Usage:
    python src/segment.py                          # process all genres
    python src/segment.py --genre lectures         # one genre only
    python src/segment.py --max-docs 50            # smoke-test on 50 docs

Output:
    data/processed/segments_sent.jsonl    one JSON line per sentence segment
    data/processed/segments_topic.jsonl   one JSON line per topic segment

Each output line has the schema:
    {
        "video_id":   str,
        "genre":      str,
        "seg_idx":    int,             # 0-indexed within the document
        "start":      float,           # seconds, from earliest caption entry
        "end":        float,           # seconds
        "text":       str,             # joined sentence(s)
        "n_sentences": int,            # always 1 for sentence view
    }
"""

import argparse
import json
from pathlib import Path

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
SENT_PATH = OUT_DIR / "segments_sent.jsonl"
TOPIC_PATH = OUT_DIR / "segments_topic.jsonl"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Topic segmentation hyperparameters
MIN_SENTS_PER_TOPIC = 3       # don't produce micro-segments
MAX_SENTS_PER_TOPIC = 25      # cap very long topic spans
SMOOTH_WINDOW = 3             # smooth similarity curve over this many sentences


# ── Loading ────────────────────────────────────────────────────────────────────

def load_docs(genres: list[str] | None, max_docs: int | None) -> list[dict]:
    """Load transcript JSONs filtered by optional genre and count."""
    all_paths = []
    if genres:
        for g in genres:
            all_paths.extend(sorted((RAW_DIR / g).rglob("*.json")))
    else:
        all_paths = sorted(RAW_DIR.rglob("*.json"))

    if max_docs is not None:
        all_paths = all_paths[:max_docs]

    docs = []
    for path in all_paths:
        try:
            docs.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"  skip {path}: {e}")
    return docs


# ── Sentence splitting + timestamp alignment ───────────────────────────────────

def split_sentences(transcript: list[dict], nlp) -> list[dict]:
    """
    Concatenate caption entries into a continuous text, then split into sentences.
    Each sentence keeps an approximate (start, end) by mapping character offsets
    back to caption-entry timestamps.

    Returns list of {text, start, end} dicts.
    """
    if not transcript:
        return []

    # Build a flat text and a parallel char-offset → time map
    char_to_time = []  # list of (char_offset, start_sec)
    pieces = []
    cursor = 0
    for entry in transcript:
        text = entry["text"].replace("\n", " ").strip()
        if not text:
            continue
        char_to_time.append((cursor, entry["start"]))
        pieces.append(text)
        cursor += len(text) + 1  # +1 for the space we'll join with

    full = " ".join(pieces)
    if not full:
        return []

    char_offsets = np.array([c for c, _ in char_to_time])
    times = np.array([t for _, t in char_to_time])

    def time_at(char_idx: int) -> float:
        idx = np.searchsorted(char_offsets, char_idx, side="right") - 1
        idx = max(0, min(idx, len(times) - 1))
        return float(times[idx])

    sents = []
    for sent in nlp(full).sents:
        s_text = sent.text.strip()
        if not s_text:
            continue
        sents.append({
            "text": s_text,
            "start": time_at(sent.start_char),
            "end": time_at(sent.end_char),
        })
    return sents


# ── Topic segmentation via embedding boundary detection ────────────────────────

def topic_segment(sentences: list[dict], embedder: SentenceTransformer) -> list[list[int]]:
    """
    Given sentence dicts, return a list of segments where each segment is a
    list of sentence indices. Boundaries are placed at local minima of
    smoothed adjacent-sentence cosine similarity.
    """
    if len(sentences) <= MIN_SENTS_PER_TOPIC:
        return [list(range(len(sentences)))]

    texts = [s["text"] for s in sentences]
    embeds = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)

    # Cosine sim between consecutive sentences (vectors are normalized)
    sims = (embeds[:-1] * embeds[1:]).sum(axis=1)

    # Smooth with a moving average so we don't fire on every tiny dip
    if SMOOTH_WINDOW > 1 and len(sims) >= SMOOTH_WINDOW:
        kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
        smooth = np.convolve(sims, kernel, mode="same")
    else:
        smooth = sims

    # Find local minima — these are candidate topic boundaries (cut between i and i+1)
    boundaries = []
    for i in range(1, len(smooth) - 1):
        if smooth[i] < smooth[i - 1] and smooth[i] < smooth[i + 1]:
            boundaries.append(i + 1)  # cut before sentence i+1

    # Enforce min/max segment size
    boundaries = enforce_size_constraints(boundaries, len(sentences))

    # Build segments from boundaries
    segs = []
    prev = 0
    for b in boundaries:
        segs.append(list(range(prev, b)))
        prev = b
    segs.append(list(range(prev, len(sentences))))
    return segs


def enforce_size_constraints(boundaries: list[int], n: int) -> list[int]:
    """Drop boundaries that would create too-small segments; insert ones for too-large."""
    # Drop boundaries that are too close to each other or to the edges
    kept = []
    last = 0
    for b in boundaries:
        if b - last >= MIN_SENTS_PER_TOPIC and (n - b) >= MIN_SENTS_PER_TOPIC:
            kept.append(b)
            last = b

    # If any segment is too long, force a cut roughly in its middle
    final = []
    prev = 0
    for b in kept + [n]:
        size = b - prev
        if size > MAX_SENTS_PER_TOPIC:
            # split roughly evenly
            n_splits = size // MAX_SENTS_PER_TOPIC
            step = size // (n_splits + 1)
            for k in range(1, n_splits + 1):
                final.append(prev + k * step)
        if b != n:
            final.append(b)
        prev = b

    return sorted(set(final))


# ── Writing ────────────────────────────────────────────────────────────────────

def write_segments(
    doc: dict,
    sentences: list[dict],
    topic_segs: list[list[int]],
    sent_fp,
    topic_fp,
):
    vid = doc["video_id"]
    genre = doc["genre"]

    # Sentence-level
    for i, s in enumerate(sentences):
        sent_fp.write(json.dumps({
            "video_id": vid,
            "genre": genre,
            "seg_idx": i,
            "start": s["start"],
            "end": s["end"],
            "text": s["text"],
            "n_sentences": 1,
        }, ensure_ascii=False) + "\n")

    # Topic-level
    for i, idxs in enumerate(topic_segs):
        if not idxs:
            continue
        topic_fp.write(json.dumps({
            "video_id": vid,
            "genre": genre,
            "seg_idx": i,
            "start": sentences[idxs[0]]["start"],
            "end": sentences[idxs[-1]]["end"],
            "text": " ".join(sentences[j]["text"] for j in idxs),
            "n_sentences": len(idxs),
        }, ensure_ascii=False) + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Segment transcripts (sentence + topic views)")
    parser.add_argument("--genre", action="append", help="Limit to genre (repeatable)")
    parser.add_argument("--max-docs", type=int, help="Process at most N documents (smoke test)")
    args = parser.parse_args()

    docs = load_docs(args.genre, args.max_docs)
    if not docs:
        print(f"No documents found under {RAW_DIR.resolve()}. Run scrape.py first.")
        return
    print(f"Loaded {len(docs)} documents.")

    print("Loading spaCy (en_core_web_sm) ...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer", "attribute_ruler"])
    nlp.add_pipe("sentencizer") if "sentencizer" not in nlp.pipe_names else None

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading sentence-embedding model ({EMBED_MODEL}) ...")
    embedder = SentenceTransformer(EMBED_MODEL, device=device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_sent = 0
    n_topic = 0

    with SENT_PATH.open("w", encoding="utf-8") as sf, TOPIC_PATH.open("w", encoding="utf-8") as tf:
        for doc in tqdm(docs, desc="segmenting"):
            sentences = split_sentences(doc["transcript"], nlp)
            if len(sentences) < MIN_SENTS_PER_TOPIC:
                continue
            topic_segs = topic_segment(sentences, embedder)
            write_segments(doc, sentences, topic_segs, sf, tf)
            n_sent += len(sentences)
            n_topic += len(topic_segs)

    print(f"\nWrote {n_sent:,} sentence segments to {SENT_PATH}")
    print(f"Wrote {n_topic:,} topic  segments to {TOPIC_PATH}")
    print(f"Avg sentences per doc: {n_sent / len(docs):.1f}")
    print(f"Avg topics    per doc: {n_topic / len(docs):.1f}")


if __name__ == "__main__":
    main()
