"""
Generate weak labels for Head C (Disfluency, 5-class).

Disfluency classes:
    0  clean         — well-formed speech
    1  filled_pause  — um, uh, hmm, er, ah
    2  repetition    — immediate word/phrase repetition
    3  revision      — self-correction or abandoned reformulation
    4  restart       — sentence abandoned mid-way and restarted

7 rule-based LFs + 1 optional open-source LLM LF.
Disfluency is largely surface-defined so rules are very strong here.

Output:
    data/processed/disfluency_labels.jsonl
    one line per segment: {video_id, genre, seg_idx, label, label_name, probs, votes}

Usage:
    python src/disfluency_labels.py
    python src/disfluency_labels.py --llm-model microsoft/Phi-3.5-mini-instruct
    python src/disfluency_labels.py --max-docs 10   # smoke test
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from snorkel.labeling.model import LabelModel
from tqdm import tqdm

SEGMENTS_PATH = Path("data/processed/segments_topic.jsonl")
OUT_PATH = Path("data/processed/disfluency_labels.jsonl")

# ── Label constants ────────────────────────────────────────────────────────────
ABSTAIN = -1
CLEAN = 0
FILLED_PAUSE = 1
REPETITION = 2
REVISION = 3
RESTART = 4
N_CLASSES = 5

DISFL_NAMES = ["clean", "filled_pause", "repetition", "revision", "restart"]

# ── Surface markers ────────────────────────────────────────────────────────────
FILLED_PAUSE_WORDS = {
    "um", "uh", "uhh", "ah", "er", "hmm", "hm", "ugh", "erm",
}

REVISION_PHRASES = [
    "i mean", "or rather", "actually no", "no wait",
    "let me rephrase", "what i meant", "let me try again",
    "or actually", "well actually", "rather,", "scratch that",
]

# Immediate word repetition: "the the", "I went I went to"
WORD_REPEAT_RE = re.compile(r'\b(\w+)\s+\1\b', re.IGNORECASE)
PHRASE_REPEAT_RE = re.compile(r'\b(\w+\s+\w+)\s+\1\b', re.IGNORECASE)

# Abandoned-sentence restart: word(s) then dash/ellipsis then capital
RESTART_DASH_RE = re.compile(r'\w+\s*[-–—]\s+[A-Z]')
RESTART_ELLIPSIS_RE = re.compile(r'\w+\s*\.\.\.\s+[A-Z]')

FILLED_PAUSE_THRESHOLD = 0.02    # ≥2% of words → filled_pause
FILLED_PAUSE_DENSE = 0.04        # ≥4% → strongly filled_pause
CLEAN_MAX_FILLER = 0.005         # <0.5% filler density qualifies as clean
CLEAN_MIN_WORDS = 8              # only label clean on reasonably long segments


# ── Rule-based LFs ─────────────────────────────────────────────────────────────

def lf_filled_pause_any(text: str) -> int:
    """Any filled pause word → filled_pause."""
    words = text.lower().split()
    if not words:
        return ABSTAIN
    if any(w.strip(".,!?;") in FILLED_PAUSE_WORDS for w in words):
        return FILLED_PAUSE
    return ABSTAIN


def lf_filled_pause_dense(text: str) -> int:
    """High density of filled pauses (≥4%) → strongly filled_pause."""
    words = text.lower().split()
    if not words:
        return ABSTAIN
    density = sum(1 for w in words if w.strip(".,!?;") in FILLED_PAUSE_WORDS) / len(words)
    return FILLED_PAUSE if density >= FILLED_PAUSE_DENSE else ABSTAIN


def lf_word_repetition(text: str) -> int:
    """Immediate word or bigram repetition → repetition disfluency."""
    if WORD_REPEAT_RE.search(text) or PHRASE_REPEAT_RE.search(text):
        return REPETITION
    return ABSTAIN


def lf_revision_phrase(text: str) -> int:
    """Self-correction markers → revision."""
    tl = text.lower()
    return REVISION if any(p in tl for p in REVISION_PHRASES) else ABSTAIN


def lf_restart_marker(text: str) -> int:
    """Abandoned-sentence patterns → restart."""
    if RESTART_DASH_RE.search(text):
        return RESTART
    if RESTART_ELLIPSIS_RE.search(text):
        return RESTART
    return ABSTAIN


def lf_clean_formal(text: str) -> int:
    """Well-formed sentence with no disfluency markers → clean."""
    words = text.lower().split()
    if len(words) < CLEAN_MIN_WORDS:
        return ABSTAIN
    # Must have no filled pauses
    if any(w.strip(".,!?;") in FILLED_PAUSE_WORDS for w in words):
        return ABSTAIN
    # Must have no revision phrases
    tl = text.lower()
    if any(p in tl for p in REVISION_PHRASES):
        return ABSTAIN
    # Must have no repetition
    if WORD_REPEAT_RE.search(text) or PHRASE_REPEAT_RE.search(text):
        return ABSTAIN
    # Must have no restart markers
    if RESTART_DASH_RE.search(text) or RESTART_ELLIPSIS_RE.search(text):
        return ABSTAIN
    # Starts with capital, ends with sentence-final punctuation
    stripped = text.strip()
    if re.match(r'^[A-Z]', stripped) and re.search(r'[.!?]$', stripped):
        return CLEAN
    return ABSTAIN


def lf_clean_low_filler(text: str) -> int:
    """Very low filler density + sufficient length → clean."""
    words = text.lower().split()
    if len(words) < CLEAN_MIN_WORDS:
        return ABSTAIN
    density = sum(1 for w in words if w.strip(".,!?;") in FILLED_PAUSE_WORDS) / len(words)
    return CLEAN if density <= CLEAN_MAX_FILLER else ABSTAIN


LF_RULE_NAMES = [
    "LF_filled_pause_any",
    "LF_filled_pause_dense",
    "LF_word_repetition",
    "LF_revision_phrase",
    "LF_restart_marker",
    "LF_clean_formal",
    "LF_clean_low_filler",
]


# ── LLM LF ────────────────────────────────────────────────────────────────────

LLM_DISFL_PROMPT = """\
Classify each transcript segment for speech disfluency.
Return ONLY a JSON array of integers, one per segment (in order).

Classes:
0 = clean         (well-formed speech, no disfluency)
1 = filled_pause  (contains um, uh, hmm, er, ah, etc.)
2 = repetition    (immediate word/phrase repetition: "the the", "I went I went")
3 = revision      (self-correction: "I mean", "or rather", abandoning a phrase mid-way)
4 = restart       (sentence abandoned and fully restarted from scratch)
-1 = uncertain

Segments:
{segments}

Return only the JSON array, e.g.: [0, 1, 0, 3, -1]"""


def lf_llm_disfluency_batch(texts: list[str], tokenizer, model, device: str) -> list[int]:
    """Classify one batch with the LLM. Returns one vote per segment."""
    import torch
    prompt = LLM_DISFL_PROMPT.format(
        segments="\n".join(f'{i+1}. "{t[:300]}"' for i, t in enumerate(texts))
    )
    try:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
    except Exception:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    try:
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            votes = json.loads(match.group())
            if len(votes) == len(texts):
                return [int(v) if int(v) in range(-1, N_CLASSES) else ABSTAIN for v in votes]
    except Exception:
        pass
    return [ABSTAIN] * len(texts)


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate(L: np.ndarray) -> np.ndarray:
    """LabelModel → (n × 5) probability matrix. Falls back to vote-count."""
    try:
        lm = LabelModel(cardinality=N_CLASSES, verbose=False)
        lm.fit(L_train=L, n_epochs=300, seed=42, log_freq=100)
        return lm.predict_proba(L)
    except Exception as e:
        print(f"  LabelModel failed ({e}) — falling back to vote count")
        probs = np.full((len(L), N_CLASSES), 1.0 / N_CLASSES)
        for i, row in enumerate(L):
            active = row[row != ABSTAIN]
            if len(active):
                for c in range(N_CLASSES):
                    probs[i, c] = float((active == c).sum()) / len(active)
        return probs


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate disfluency weak labels (Head C)")
    parser.add_argument("--max-docs", type=int)
    parser.add_argument(
        "--llm-model", type=str, default=None,
        help="HuggingFace causal LM for LF_llm_disfluency "
             "(e.g. microsoft/Phi-3.5-mini-instruct)",
    )
    parser.add_argument("--llm-batch-size", type=int, default=8)
    args = parser.parse_args()

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

    # Optional LLM
    llm_fn = None
    lf_names = LF_RULE_NAMES.copy()
    if args.llm_model:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading LLM ({args.llm_model}) ...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            args.llm_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(device)
        mdl.eval()
        batch_size = args.llm_batch_size

        def llm_fn(texts: list[str]) -> list[int]:
            results: list[int] = []
            for s in range(0, len(texts), batch_size):
                results.extend(lf_llm_disfluency_batch(texts[s:s + batch_size], tok, mdl, device))
            return results

        lf_names.append("LF_llm_disfluency")
        print(f"LLM loaded on {device}")

    all_segs: list[dict] = []
    all_votes: list[list[int]] = []

    for vid in tqdm(video_ids, desc="disfluency labeling"):
        segs = docs[vid]
        texts = [s["text"] for s in segs]
        llm_votes = llm_fn(texts) if llm_fn else [ABSTAIN] * len(texts)

        for i, (seg, text) in enumerate(zip(segs, texts)):
            votes = [
                lf_filled_pause_any(text),
                lf_filled_pause_dense(text),
                lf_word_repetition(text),
                lf_revision_phrase(text),
                lf_restart_marker(text),
                lf_clean_formal(text),
                lf_clean_low_filler(text),
            ]
            if llm_fn:
                votes.append(llm_votes[i])
            all_segs.append(seg)
            all_votes.append(votes)

    L = np.array(all_votes, dtype=int)

    print("\nLF coverage report:")
    for j, name in enumerate(lf_names):
        col = L[:, j]
        n_active = int((col != ABSTAIN).sum())
        pct = 100 * n_active / len(col)
        counts = "  ".join(f"{DISFL_NAMES[c]}={int((col==c).sum())}" for c in range(N_CLASSES))
        print(f"  {name:30s}  coverage={pct:5.1f}%  {counts}")

    print(f"\nAggregating with Snorkel LabelModel (cardinality={N_CLASSES}) ...")
    probs = aggregate(L)
    labels = probs.argmax(axis=1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for seg, votes, prob_vec, label in zip(all_segs, all_votes, probs, labels):
            f.write(json.dumps({
                "video_id": seg["video_id"],
                "genre": seg["genre"],
                "seg_idx": seg["seg_idx"],
                "label": int(label),
                "label_name": DISFL_NAMES[int(label)],
                "probs": [round(float(p), 4) for p in prob_vec],
                "votes": dict(zip(lf_names, [int(v) for v in votes])),
            }, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_segs):,} disfluency labels to {OUT_PATH}")
    print("Label distribution:")
    for c, name in enumerate(DISFL_NAMES):
        n = int((labels == c).sum())
        print(f"  {name:15s}  {n:6d}  ({100*n/len(labels):.1f}%)")


if __name__ == "__main__":
    main()
