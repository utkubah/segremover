"""
Generate weak labels for Head B (Function, 6-class).

Function classes:
    0  new_information      — introduces a fact not yet stated
    1  useful_repetition    — repeats for emphasis or pedagogy
    2  redundant_repetition — restates with no added value
    3  clarification        — elaborates or explains prior content
    4  discourse_filler     — hedges, transitions, meta-speech
    5  off_topic            — tangent or banter

13 rule-based LFs + 1 optional open-source LLM LF.

Output:
    data/processed/function_labels.jsonl
    one line per segment: {video_id, genre, seg_idx, label, label_name, probs, votes}

Usage:
    python src/function_labels.py
    python src/function_labels.py --llm-model microsoft/Phi-3.5-mini-instruct
    python src/function_labels.py --max-docs 10   # smoke test
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from snorkel.labeling.model import LabelModel
from tqdm import tqdm

SEGMENTS_PATH = Path("data/processed/segments_topic.jsonl")
OUT_PATH = Path("data/processed/function_labels.jsonl")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Label constants ────────────────────────────────────────────────────────────
ABSTAIN = -1
NEW_INFO = 0
USEFUL_REP = 1
REDUNDANT_REP = 2
CLARIFICATION = 3
DISCOURSE_FILLER = 4
OFF_TOPIC = 5
N_CLASSES = 6

FUNCTION_NAMES = [
    "new_information", "useful_repetition", "redundant_repetition",
    "clarification", "discourse_filler", "off_topic",
]

# ── LF configuration ──────────────────────────────────────────────────────────
FILLER_WORDS = {
    "um", "uh", "uhh", "ah", "er", "hmm", "like", "basically",
    "literally", "actually", "okay", "right",
}
MULTIWORD_FILLERS = ["you know", "i mean", "sort of", "kind of"]

DEFINITION_PHRASES = [
    "is defined as", "refers to", "means that", "is called", "known as",
    "is a type of", "is an example of",
]
ELABORATION_PHRASES = [
    "for example", "for instance", "such as", "specifically",
    "in particular", "namely", "e.g.", "i.e.",
]
CLARIFICATION_PHRASES = [
    "in other words", "that is to say", "which means",
    "to put it another way", "more precisely", "to clarify",
    "what i mean is", "let me rephrase",
]
RECAP_PHRASES = [
    "in summary", "to summarize", "in conclusion", "to recap",
    "as i mentioned", "as we saw", "as discussed", "to review",
    "in short", "so far we've", "we've seen that", "as i said",
    "going back to", "remember that", "as you recall",
]
TRANSITION_FILLERS = [
    "okay so", "alright so", "so anyway", "moving on",
    "let's move on", "next up", "let's talk about", "so let me",
    "okay let's", "right so", "now let's", "alright let's",
    "so yeah", "anyway", "so moving on",
]
QUESTION_INTROS = [
    "what is", "what are", "why does", "why is", "how do",
    "how does", "can you", "do you know", "have you ever",
    "what happens", "why would", "how can", "what would",
]

FILLER_THRESHOLD = 0.03
TFIDF_HIGH_PERCENTILE = 70
REPETITION_THRESHOLD = 0.70
CENTROID_LOW_PERCENTILE = 20
MIN_NUMERICAL_HITS = 2
NE_MIN = 2
LLM_BATCH_SIZE = 8


# ── Rule-based LFs ─────────────────────────────────────────────────────────────

def lf_high_tfidf(i: int, tfidf_scores: np.ndarray) -> int:
    """Top 30% TF-IDF → new information."""
    threshold = np.percentile(tfidf_scores, TFIDF_HIGH_PERCENTILE)
    return NEW_INFO if tfidf_scores[i] >= threshold else ABSTAIN


def lf_filler_density(text: str) -> int:
    """High filler word density → discourse filler."""
    words = text.lower().split()
    if not words:
        return ABSTAIN
    text_lower = text.lower()
    hits = sum(1 for w in words if w in FILLER_WORDS)
    hits += sum(text_lower.count(p) * len(p.split()) for p in MULTIWORD_FILLERS)
    return DISCOURSE_FILLER if hits / len(words) > FILLER_THRESHOLD else ABSTAIN


def lf_definition_phrase(text: str) -> int:
    """Definition phrases → clarification."""
    tl = text.lower()
    return CLARIFICATION if any(p in tl for p in DEFINITION_PHRASES) else ABSTAIN


def lf_elaboration_phrase(text: str) -> int:
    """'For example / such as' → clarification."""
    tl = text.lower()
    return CLARIFICATION if any(p in tl for p in ELABORATION_PHRASES) else ABSTAIN


def lf_clarification_phrase(text: str) -> int:
    """'In other words' style phrases → clarification."""
    tl = text.lower()
    return CLARIFICATION if any(p in tl for p in CLARIFICATION_PHRASES) else ABSTAIN


def lf_recap_phrase(text: str) -> int:
    """Summary/recap phrases → useful repetition."""
    tl = text.lower()
    return USEFUL_REP if any(p in tl for p in RECAP_PHRASES) else ABSTAIN


def lf_transition_filler(text: str) -> int:
    """Transition filler phrases → discourse filler."""
    tl = text.lower()
    return DISCOURSE_FILLER if any(p in tl for p in TRANSITION_FILLERS) else ABSTAIN


def lf_question_new_info(text: str) -> int:
    """Rhetorical questions introducing topics → new information."""
    tl = text.lower()
    if "?" in text and any(tl.lstrip().startswith(q) or f" {q}" in tl for q in QUESTION_INTROS):
        return NEW_INFO
    return ABSTAIN


def lf_named_entity_dense(text: str, nlp) -> int:
    """Dense named entities suggest factual, new content."""
    return NEW_INFO if len(nlp(text).ents) >= NE_MIN else ABSTAIN


def lf_numerical_content(text: str) -> int:
    """Numbers, percentages, statistics → new information."""
    hits = len(re.findall(
        r'\b\d+\.?\d*\s*%'
        r'|\b\d+\.?\d*\s*(million|billion|thousand|hundred)'
        r'|\b\d{4}\b'
        r'|\b\d+\.\d+\b',
        text
    ))
    return NEW_INFO if hits >= MIN_NUMERICAL_HITS else ABSTAIN


def lf_repetition_cosine(i: int, embeddings: np.ndarray) -> int:
    """High cosine similarity to any earlier segment → redundant repetition."""
    if i == 0:
        return ABSTAIN
    sims = embeddings[:i] @ embeddings[i]
    return REDUNDANT_REP if float(sims.max()) > REPETITION_THRESHOLD else ABSTAIN


def lf_off_topic_centroid(i: int, embeddings: np.ndarray, centroid: np.ndarray) -> int:
    """Low cosine to document centroid → off topic."""
    sims = embeddings @ centroid
    threshold = np.percentile(sims, CENTROID_LOW_PERCENTILE)
    return OFF_TOPIC if sims[i] <= threshold else ABSTAIN


def lf_position_discourse(i: int, n: int) -> int:
    """First/last 2 segments tend to be intro/outro filler."""
    return DISCOURSE_FILLER if (i < 2 or i >= n - 2) else ABSTAIN


LF_RULE_NAMES = [
    "LF_high_tfidf",
    "LF_filler_density",
    "LF_definition_phrase",
    "LF_elaboration_phrase",
    "LF_clarification_phrase",
    "LF_recap_phrase",
    "LF_transition_filler",
    "LF_question_new_info",
    "LF_named_entity_dense",
    "LF_numerical_content",
    "LF_repetition_cosine",
    "LF_off_topic_centroid",
    "LF_position_discourse",
]


# ── LLM LF ────────────────────────────────────────────────────────────────────

LLM_FUNCTION_PROMPT = """\
Classify each transcript segment into one function class.
Return ONLY a JSON array of integers, one per segment (in order).

Classes:
0 = new_information      (introduces a fact not stated before)
1 = useful_repetition    (repeats for emphasis or pedagogy)
2 = redundant_repetition (restates with no added value)
3 = clarification        (elaborates or explains with examples/analogies)
4 = discourse_filler     (hedges, transitions, meta-speech, um/uh)
5 = off_topic            (tangent or banter unrelated to main topic)
-1 = uncertain

Segments:
{segments}

Return only the JSON array, e.g.: [0, 4, 3, -1, 2]"""


def lf_llm_function_batch(texts: list[str], tokenizer, model, device: str) -> list[int]:
    """Classify one batch of segments with the LLM. Returns one vote per segment."""
    import torch
    prompt = LLM_FUNCTION_PROMPT.format(
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


# ── Per-document processing ────────────────────────────────────────────────────

def doc_votes(segs: list[dict], embedder, nlp, llm_fn=None) -> list[list[int]]:
    texts = [s["text"] for s in segs]
    n = len(texts)

    embeds = embedder.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    centroid = embeds.mean(axis=0)
    norm = np.linalg.norm(centroid)
    centroid = centroid / norm if norm > 0 else centroid

    try:
        tfidf_scores = TfidfVectorizer(stop_words="english").fit_transform(texts).toarray().sum(axis=1)
    except ValueError:
        tfidf_scores = np.zeros(n)

    llm_votes = [ABSTAIN] * n
    if llm_fn is not None:
        for start in range(0, n, LLM_BATCH_SIZE):
            batch_votes = llm_fn(texts[start:start + LLM_BATCH_SIZE])
            llm_votes[start:start + len(batch_votes)] = batch_votes

    all_votes = []
    for i, text in enumerate(texts):
        votes = [
            lf_high_tfidf(i, tfidf_scores),
            lf_filler_density(text),
            lf_definition_phrase(text),
            lf_elaboration_phrase(text),
            lf_clarification_phrase(text),
            lf_recap_phrase(text),
            lf_transition_filler(text),
            lf_question_new_info(text),
            lf_named_entity_dense(text, nlp),
            lf_numerical_content(text),
            lf_repetition_cosine(i, embeds),
            lf_off_topic_centroid(i, embeds, centroid),
            lf_position_discourse(i, n),
        ]
        if llm_fn is not None:
            votes.append(llm_votes[i])
        all_votes.append(votes)
    return all_votes


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate(L: np.ndarray) -> np.ndarray:
    """LabelModel → (n × 6) probability matrix. Falls back to vote-count."""
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
    parser = argparse.ArgumentParser(description="Generate function weak labels (Head B)")
    parser.add_argument("--max-docs", type=int)
    parser.add_argument(
        "--llm-model", type=str, default=None,
        help="HuggingFace causal LM for LF_llm_function "
             "(e.g. microsoft/Phi-3.5-mini-instruct, mistralai/Mistral-7B-Instruct-v0.3)",
    )
    parser.add_argument("--llm-batch-size", type=int, default=LLM_BATCH_SIZE)
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

    print("Loading spaCy (en_core_web_sm) ...")
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "lemmatizer", "attribute_ruler"])

    print(f"Loading sentence embedder ({EMBED_MODEL}) ...")
    embedder = SentenceTransformer(EMBED_MODEL)

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
        llm_fn = lambda texts: lf_llm_function_batch(texts, tok, mdl, device)  # noqa: E731
        lf_names.append("LF_llm_function")
        print(f"LLM loaded on {device}")

    all_segs: list[dict] = []
    all_votes: list[list[int]] = []

    for vid in tqdm(video_ids, desc="function labeling"):
        segs = docs[vid]
        votes = doc_votes(segs, embedder, nlp, llm_fn)
        all_segs.extend(segs)
        all_votes.extend(votes)

    L = np.array(all_votes, dtype=int)

    print("\nLF coverage report:")
    for j, name in enumerate(lf_names):
        col = L[:, j]
        n_active = int((col != ABSTAIN).sum())
        pct = 100 * n_active / len(col)
        counts = "  ".join(f"{FUNCTION_NAMES[c]}={int((col==c).sum())}" for c in range(N_CLASSES))
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
                "label_name": FUNCTION_NAMES[int(label)],
                "probs": [round(float(p), 4) for p in prob_vec],
                "votes": dict(zip(lf_names, [int(v) for v in votes])),
            }, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_segs):,} function labels to {OUT_PATH}")
    print("Label distribution:")
    for c, name in enumerate(FUNCTION_NAMES):
        n = int((labels == c).sum())
        print(f"  {name:25s}  {n:6d}  ({100*n/len(labels):.1f}%)")


if __name__ == "__main__":
    main()
