"""
LF_summary_align — summary-alignment labeling function.

For each video:
  1. Reconstruct the full transcript text from topic segments.
  2. Chunk it into ~600-word pieces and summarize each chunk with
     facebook/bart-large-cnn (runs locally, no API key required).
  3. Align each segment to the combined summary using SBERT cosine similarity.
  4. Bottom REMOVE_PERCENTILE % → remove, top KEEP_PERCENTILE % → keep,
     middle → abstain.

Output:
    data/processed/summary_align_votes.jsonl
    one line per segment: {video_id, seg_idx, lf_summary_align, align_score}

Once this file exists, weak_labels.py automatically includes it as a 7th LF.

Usage:
    python src/summary_align.py
    python src/summary_align.py --max-docs 5   # smoke test

Note: First run downloads facebook/bart-large-cnn (~1.6 GB). Subsequent runs
use the cached model. Runs on CPU; expect ~10-20 s per video.

When you have an LLM API key later, swap generate_summary() to call the API
instead — the rest of the pipeline stays unchanged.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

SEGMENTS_PATH = Path("data/processed/segments_topic.jsonl")
OUT_PATH = Path("data/processed/summary_align_votes.jsonl")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARIZER_MODEL = "facebook/bart-large-cnn"

ABSTAIN = -1
KEEP = 0
REMOVE = 1

CHUNK_WORDS = 600       # words per chunk fed to BART
SUMMARY_MAX_TOKENS = 150
SUMMARY_MIN_TOKENS = 40
REMOVE_PERCENTILE = 30  # bottom 30% alignment → remove
KEEP_PERCENTILE = 70    # top 30% alignment → keep


# ── Summarization ──────────────────────────────────────────────────────────────

def build_full_text(segs: list[dict]) -> str:
    """Reconstruct full document text from sorted segments."""
    return " ".join(s["text"] for s in sorted(segs, key=lambda s: s["seg_idx"]))


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i: i + chunk_words])
        for i in range(0, len(words), chunk_words)
        if words[i: i + chunk_words]  # skip empty tail
    ]


def generate_summary(text: str, tokenizer, model) -> str:
    """
    Summarize a document by chunking and summarizing each piece with BART.
    Swap this function for an API call when you have a key — the rest is unchanged.
    """
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        # BART input limit is ~1024 tokens; trim chunk to be safe
        trimmed = " ".join(chunk.split()[:700])
        inputs = tokenizer(trimmed, return_tensors="pt", max_length=1024, truncation=True).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=SUMMARY_MAX_TOKENS,
                min_length=SUMMARY_MIN_TOKENS,
                num_beams=4,
                early_stopping=True,
            )
        summaries.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    return " ".join(summaries)


# ── Alignment ──────────────────────────────────────────────────────────────────

def align_segments_to_summary(
    seg_embeds: np.ndarray,
    summary: str,
    embedder: SentenceTransformer,
) -> np.ndarray:
    """
    For each segment, compute max cosine similarity to any summary sentence.
    Returns a 1-D array of alignment scores ∈ [0, 1].
    """
    # Split summary into sentences (simple split on "." is fine here)
    summary_sents = [s.strip() for s in summary.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    if not summary_sents:
        return np.full(len(seg_embeds), 0.5)

    sum_embeds = embedder.encode(
        summary_sents, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    # cosine similarity: seg_embeds (n,d) @ sum_embeds.T (d,m) → (n,m)
    scores = seg_embeds @ sum_embeds.T
    return scores.max(axis=1)  # best match per segment


def votes_from_scores(scores: np.ndarray) -> list[int]:
    """Convert alignment scores to LF votes using percentile thresholds."""
    low = np.percentile(scores, REMOVE_PERCENTILE)
    high = np.percentile(scores, KEEP_PERCENTILE)
    votes = []
    for s in scores:
        if s <= low:
            votes.append(REMOVE)
        elif s >= high:
            votes.append(KEEP)
        else:
            votes.append(ABSTAIN)
    return votes


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LF_summary_align votes")
    parser.add_argument("--max-docs", type=int, help="Limit to N documents (smoke test)")
    args = parser.parse_args()

    print(f"Loading segments from {SEGMENTS_PATH} ...")
    docs: dict[str, list[dict]] = defaultdict(list)
    with SEGMENTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            seg = json.loads(line)
            docs[seg["video_id"]].append(seg)

    video_ids = list(docs.keys())
    if args.max_docs:
        video_ids = video_ids[:args.max_docs]
    # ── Resume: skip already-processed video_ids ───────────────────────────────
    done: set[str] = set()
    if OUT_PATH.exists():
        with OUT_PATH.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["video_id"])
                except Exception:
                    pass
        if done:
            print(f"Resuming: {len(done)} videos already done, skipping them.")
            video_ids = [v for v in video_ids if v not in done]

    print(f"Processing {len(video_ids)} documents ...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading summarizer ({SUMMARIZER_MODEL}) — first run downloads ~1.6 GB ...")
    tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL).to(device)
    model.eval()

    print(f"Loading sentence-embedding model ({EMBED_MODEL}) ...")
    embedder = SentenceTransformer(EMBED_MODEL, device=device)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_remove = total_keep = total_abstain = 0

    with OUT_PATH.open("a", encoding="utf-8") as out_f:
        for vid in tqdm(video_ids, desc="summarizing"):
            segs = sorted(docs[vid], key=lambda s: s["seg_idx"])
            texts = [s["text"] for s in segs]
            full_text = " ".join(texts)

            summary = generate_summary(full_text, tokenizer, model)

            seg_embeds = embedder.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )
            scores = align_segments_to_summary(seg_embeds, summary, embedder)
            votes = votes_from_scores(scores)

            for seg, vote, score in zip(segs, votes, scores):
                out_f.write(json.dumps({
                    "video_id": seg["video_id"],
                    "seg_idx": seg["seg_idx"],
                    "lf_summary_align": int(vote),
                    "align_score": round(float(score), 4),
                }) + "\n")

            total_remove += votes.count(REMOVE)
            total_keep += votes.count(KEEP)
            total_abstain += votes.count(ABSTAIN)

    total = total_remove + total_keep + total_abstain
    print(f"\nWrote {total:,} new votes to {OUT_PATH}")
    if total == 0:
        print("  (nothing to do — all documents already processed)")
    else:
        print(f"  remove={total_remove:,}  keep={total_keep:,}  abstain={total_abstain:,}")
        print(f"  coverage = {100*(total-total_abstain)/total:.1f}%")
    print("\nNow re-run: python src/weak_labels.py")


if __name__ == "__main__":
    main()
