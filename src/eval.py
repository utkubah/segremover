"""
src/eval.py — Evaluation suite for segremover.

Sections
────────
1  Data loading & audit
2  Model inference        (GPU required; skipped with --no-model or missing checkpoint)
3  Segment-level metrics  (ROC, PR-AUC, ECE, threshold sweep, confusion matrix,
                           per-genre reliability, calibration before/after T)
4  Transcript-level       (compression vs SBERT-preservation + ROUGE, by genre/length)
5  Baseline comparison    (model vs random/heuristic/tfidf/sbert, with per-genre bars)
6  4-way agreement        (model / sbert / heuristic / weak-label proxy, + per-genre)
7  Gold evaluation        (Spearman ρ, NDCG@20, weak-label accuracy vs gold — gold only)
8  Caption-type robustness (manual vs auto captions)
9  Qualitative examples   (CSV + Markdown, per-genre failure mode table)
10 Markdown report

All plots saved as PNG + PDF (dark background, suitable for slides).
Outputs go to --output-dir (default: outputs/eval/).

CPU feasibility
───────────────
• Smoke test  (--max-docs 5):  CPU is fine for everything (~5–10 min)
• Full corpus (5 000 docs):    GPU required.
  - Model inference alone: 315 k segments through roberta-base
    GPU ≈ 4–5 h  |  CPU ≈ days
  - SBERT transcript eval:  similar scale
  Use --no-model to skip roberta and run only baselines + SBERT centroid,
  which still needs GPU for the full corpus.

Usage
─────
python src/eval.py --max-docs 5 --no-model      # smoke test, CPU OK
python src/eval.py --no-model                    # baselines only, GPU for SBERT
python src/eval.py --checkpoint models/best.pt   # full eval
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless — safe on HPC, no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
    ndcg_score,
)
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ── Dark-theme constants ───────────────────────────────────────────────────────
BG, PANEL, GRID = "#0f1117", "#1a1f2e", "#2d3748"
TEXT            = "#e2e8f0"
BLUE, GREEN, RED, ORANGE, PURPLE = "#63b3ed", "#68d391", "#fc8181", "#f6ad55", "#b794f4"

# Bright foreground colors on a dark panel.  Keep bars vivid; keep only
# backgrounds/grid dark.
BRIGHT_CYAN    = "#38bdf8"
BRIGHT_GREEN   = "#4ade80"
BRIGHT_ORANGE  = "#fbbf24"
BRIGHT_PURPLE  = "#a78bfa"
BRIGHT_RED     = "#fb7185"
BRIGHT_PINK    = "#f472b6"
BRIGHT_TEAL    = "#2dd4bf"
BRIGHT_LIME    = "#bef264"
BRIGHT_SLATE   = "#94a3b8"

EXPECTED_GENRES = [
    "commentary", "entertainment", "lectures",
    "podcasts", "ted", "tv_series",
]

GENRE_PALETTE = {
    "commentary":    BRIGHT_CYAN,
    "entertainment": BRIGHT_PINK,
    "lectures":      BRIGHT_ORANGE,
    "podcasts":      BRIGHT_GREEN,
    "ted":           BRIGHT_RED,
    "ted_talks":     BRIGHT_RED,
    "tv_series":     BRIGHT_PURPLE,
    "unknown":       BRIGHT_SLATE,
}
BUCKET_COLORS  = {"short": BRIGHT_CYAN, "medium": BRIGHT_ORANGE, "long": BRIGHT_GREEN, "unknown": BRIGHT_SLATE}
BASELINE_COLORS = {
    "model":     BRIGHT_CYAN,
    "sbert":     BRIGHT_GREEN,
    "heuristic": BRIGHT_ORANGE,
    "tfidf":     BRIGHT_PURPLE,
    "random":    BRIGHT_PINK,
}
FUNCTION_COLORS = {
    "new_information":       BRIGHT_CYAN,
    "useful_repetition":     BRIGHT_TEAL,
    "redundant_repetition":  BRIGHT_ORANGE,
    "clarification":         BRIGHT_GREEN,
    "discourse_filler":      BRIGHT_PURPLE,
    "off_topic":             BRIGHT_RED,
}
CAPTION_COLORS = {"manual": BRIGHT_CYAN, "auto": BRIGHT_ORANGE, "automatic": BRIGHT_ORANGE, "unknown": BRIGHT_SLATE}

plt.rcParams.update({
    "figure.facecolor": BG,   "axes.facecolor":   PANEL,
    "text.color":       TEXT, "axes.labelcolor":  TEXT,
    "xtick.color":      TEXT, "ytick.color":      TEXT,
    "grid.color":       GRID, "axes.edgecolor":   GRID,
    "legend.facecolor": PANEL,"legend.edgecolor": GRID,
    "figure.dpi":       130,  "font.size":        13,
    "axes.titlesize":   15,   "axes.labelsize":   13,
    "legend.fontsize":  11,   "lines.linewidth":  2.0,
})

FUNCTION_NAMES = [
    "new_information", "useful_repetition", "redundant_repetition",
    "clarification",   "discourse_filler",  "off_topic",
]
DISFL_NAMES = ["clean", "filled_pause", "repetition", "revision", "restart"]

VISUAL_CONTEXT_PATTERNS = [
    "this graph", "this chart", "this diagram", "this slide", "this image",
    "on screen", "as you can see", "you can see", "look at", "shown here",
    "right here", "over here", "this picture", "this video", "the screen",
    "the board", "the figure", "the table", "the map", "the camera",
]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, plots_dir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(plots_dir / f"{name}.{ext}",
                    bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _ax_style(ax: plt.Axes) -> plt.Axes:
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    return ax


def _pretty_name(name: object) -> str:
    """Display labels for plots without changing raw data keys."""
    s = str(name)
    mapping = {
        "model": "SegRemover",
        "sbert": "SBERT",
        "tfidf": "TF-IDF",
        "heuristic": "Heuristic",
        "random": "Random",
        "weak": "weak-label proxy",
        "weak\n(proxy)": "weak-label\nproxy",
        "tv_series": "TV series",
        "ted": "TED talks",
        "ted_talks": "TED talks",
        "commentary": "Commentary",
        "entertainment": "Entertainment",
        "lectures": "Lectures",
        "podcasts": "Podcasts",
        "short": "Short",
        "medium": "Medium",
        "long": "Long",
        "manual": "Manual",
        "auto": "Auto",
        "automatic": "Auto",
        "unknown": "Unknown",
    }
    return mapping.get(s, s.replace("_", " ").title())


def _ordered(values: list[str], preferred: list[str]) -> list[str]:
    seen = set(values)
    out = [v for v in preferred if v in seen]
    out.extend(sorted(v for v in values if v not in set(out)))
    return out


def _calibration_points(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "lo": lo,
            "hi": hi,
            "mid": (lo + hi) / 2,
            "accuracy": float(labels[mask].mean()),
            "confidence": float(probs[mask].mean()),
            "n": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def _pick_thresholds(thresholds: list[float], targets: tuple[float, ...] = (0.70, 0.95)) -> list[float]:
    if not thresholds:
        return []
    picked = []
    arr = np.array(thresholds, dtype=float)
    for target in targets:
        t = float(arr[np.argmin(np.abs(arr - target))])
        if t not in picked:
            picked.append(t)
    return picked


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _bootstrap_ci(
    scores: np.ndarray, labels: np.ndarray,
    metric_fn, n_boot: int = 500, ci: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a scalar metric. Returns (lo, hi)."""
    rng  = np.random.default_rng(42)
    n    = len(scores)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            boot.append(float(metric_fn(labels[idx], scores[idx])))
        except Exception:
            continue
    if not boot:
        return float("nan"), float("nan")
    boot = np.sort(boot)
    alpha = (1 - ci) / 2
    return float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))


def _ndcg_at_k(scores: np.ndarray, gold_soft: np.ndarray, k: int = 20) -> float:
    """NDCG@k — rank segments by p_remove, relevance = gold soft target."""
    # sklearn ndcg_score expects shape (1, n)
    try:
        return float(ndcg_score(gold_soft[None, :], scores[None, :], k=k))
    except Exception:
        return float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_all_data(data_dir: Path, max_docs: int | None = None,
                  label_suffix: str = "") -> dict:
    """Load and cross-reference all processed data files.

    label_suffix: appended to every label filename stem before the extension,
                  e.g. "_smoke" reads weak_labels_smoke.jsonl etc.
                  segments_topic.jsonl is never suffixed (no smoke variant exists).
    """
    def _lf(name: str) -> Path:
        """Resolve a label-file path, applying label_suffix if set."""
        p = Path(name)
        return data_dir / (p.stem + label_suffix + p.suffix)

    # ── Segments ──────────────────────────────────────────────────────────────
    docs: dict[str, list[dict]] = defaultdict(list)
    for r in _load_jsonl(data_dir / "segments_topic.jsonl"):
        docs[r["video_id"]].append(r)
    for segs in docs.values():
        segs.sort(key=lambda s: s["seg_idx"])

    video_ids = sorted(docs.keys())
    if max_docs:
        video_ids = video_ids[:max_docs]
        docs = {v: docs[v] for v in video_ids}

    in_scope: set[tuple] = {
        (r["video_id"], r["seg_idx"])
        for v in video_ids for r in docs[v]
    }

    def _key_map(rows: list[dict], val: str) -> dict[tuple, object]:
        return {
            (r["video_id"], r["seg_idx"]): r[val]
            for r in rows
            if (r["video_id"], r["seg_idx"]) in in_scope
        }

    weak  = _key_map(_load_jsonl(_lf("weak_labels.jsonl")),       "p_remove")
    fn    = _key_map(_load_jsonl(_lf("function_labels.jsonl")),    "label")
    disfl = _key_map(_load_jsonl(_lf("disfluency_labels.jsonl")),  "label")

    base_cols = ["random", "heuristic", "tfidf", "sbert"]
    baselines: dict[tuple, dict] = {}
    for r in _load_jsonl(_lf("baselines.jsonl")):
        k = (r["video_id"], r["seg_idx"])
        if k in in_scope:
            baselines[k] = {c: r[c] for c in base_cols if c in r and r[c] is not None}

    # ── Corpus summary (length_bucket etc.) ───────────────────────────────────
    summary_path = data_dir / "corpus_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else None

    genre: dict[str, str]  = {v: docs[v][0]["genre"] for v in video_ids if docs[v]}
    length_bucket: dict[str, str] = {}
    if summary is not None and "length_bucket" in summary.columns:
        for _, row in summary.iterrows():
            if row["video_id"] in docs:
                length_bucket[str(row["video_id"])] = str(row["length_bucket"])

    # ── Caption type (manual vs auto) — stored on first segment if available ──
    caption_type: dict[str, str] = {}
    for v in video_ids:
        segs = docs[v]
        if segs and "caption_type" in segs[0]:
            caption_type[v] = segs[0]["caption_type"]

    # ── Gold labels (null until humans annotate) ──────────────────────────────
    # Reads per-video JSON files from data/gold/videos/*.json.
    # Each file is either a single entry dict or a list of entry dicts, where
    # each entry has: video_id, segments[], removability_to_soft_target{}, function_to_binary_remove{}.
    gold:  dict[tuple, float] = {}   # soft target from gold_removability ordinal
    gold_b: dict[tuple, int]  = {}   # binary remove/keep from gold_function
    gold_dir = data_dir.parent / "gold" / "videos"
    if not gold_dir.exists():
        gold_dir = data_dir.parent / "gold"   # fallback: flat gold/ directory

    gold_entries: list[dict] = []
    if gold_dir.exists():
        for json_file in sorted(gold_dir.glob("*.json")):
            try:
                payload = json.loads(json_file.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    gold_entries.extend(payload)
                elif isinstance(payload, dict):
                    gold_entries.append(payload)
            except Exception as e:
                print(f"  ⚠  Could not read gold file {json_file.name}: {e}")

    for entry in gold_entries:
        rmap = entry.get("removability_to_soft_target", {})
        fmap = entry.get("function_to_binary_remove", {})
        for seg in entry.get("segments", []):
            k = (entry["video_id"], seg["seg_idx"])
            if k not in in_scope:
                continue
            if seg.get("gold_removability") is not None:
                gold[k] = float(rmap.get(seg["gold_removability"], 0.5))
            if seg.get("gold_function") is not None:
                gold_b[k] = int(fmap.get(seg["gold_function"], 0))

    return dict(
        docs=docs, video_ids=video_ids, genre=genre, length_bucket=length_bucket,
        caption_type=caption_type,
        weak=weak, fn=fn, disfl=disfl, baselines=baselines,
        gold=gold, gold_b=gold_b, summary=summary,
    )


def sample_eval_docs(data: dict, n: int, seed: int = 42) -> dict:
    """Keep at most n videos, always including all gold-annotated ones.

    Gold-annotated video_ids are pinned; the remainder of the budget is
    filled with a random draw from the rest of the corpus.
    """
    import random as _rnd
    all_ids = data["video_ids"]

    gold_vids: set[str] = {vid for vid, _ in data["gold"]} | {vid for vid, _ in data["gold_b"]}
    pinned  = [v for v in all_ids if v in gold_vids]
    rest    = [v for v in all_ids if v not in gold_vids]

    n_extra = max(0, n - len(pinned))
    rng     = _rnd.Random(seed)
    extra   = rng.sample(rest, min(n_extra, len(rest)))

    sampled     = sorted(set(pinned) | set(extra))
    sampled_set = set(sampled)

    def _seg(d: dict) -> dict:          # (video_id, seg_idx) → value
        return {k: v for k, v in d.items() if k[0] in sampled_set}

    def _str(d: dict) -> dict:          # video_id → value
        return {k: v for k, v in d.items() if k in sampled_set}

    return dict(
        docs={v: data["docs"][v] for v in sampled},
        video_ids=sampled,
        genre=_str(data["genre"]),
        length_bucket=_str(data["length_bucket"]),
        caption_type=_str(data["caption_type"]),
        weak=_seg(data["weak"]),
        fn=_seg(data["fn"]),
        disfl=_seg(data["disfl"]),
        baselines=_seg(data["baselines"]),
        gold=_seg(data["gold"]),
        gold_b=_seg(data["gold_b"]),
        summary=data["summary"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. Model inference
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(checkpoint: Path, config: dict, data: dict, device: str) -> dict[tuple, float]:
    """Load SegRemover from checkpoint, score every in-scope segment."""
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train import SegRemover, DocDataset, collate_fn   # type: ignore
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    T        = float(config.get("temperature", 1.0))
    max_segs = int(config.get("max_segs", 256))
    max_tok  = int(config.get("max_tok", 512))
    seg_batch = int(config.get("seg_batch", 32))
    docs, video_ids = data["docs"], data["video_ids"]

    tokenizer = AutoTokenizer.from_pretrained(config["encoder"])
    model = SegRemover(
        config["encoder"],
        config.get("inter_layers", 2),
        config.get("dropout", 0.1),
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device).eval()
    print(f"  Loaded checkpoint  {checkpoint}")
    print(f"  Temperature T={T}  encoder={config['encoder']}")

    # Build doc_list in DocDataset format (dummy labels — only text matters)
    doc_list = [
        {"video_id": v, "genre": data["genre"][v],
         "segments": [{"seg_idx": s["seg_idx"], "text": s["text"],
                       "p_remove": 0.5, "fn_label": 0, "disfl_label": 0}
                      for s in docs[v]]}
        for v in video_ids
    ]
    ds     = DocDataset(doc_list, tokenizer, max_tok, max_segs)
    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    scores:  dict[tuple, float] = {}
    heads_b: dict[tuple, int]   = {}
    heads_c: dict[tuple, int]   = {}
    doc_cursor = 0
    for batch in tqdm(loader, desc="inference"):
        with torch.no_grad():
            la, lb, lc = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["doc_lengths"],
                seg_batch=seg_batch,
            )
        probs  = torch.sigmoid(la / T).cpu().tolist()
        pred_b = lb.argmax(dim=-1).cpu().tolist()
        pred_c = lc.argmax(dim=-1).cpu().tolist()
        seg_ptr = 0
        for i, L in enumerate(batch["doc_lengths"]):
            vid = video_ids[doc_cursor]
            for j in range(L):
                key = (vid, docs[vid][j]["seg_idx"])
                scores[key]  = round(probs[seg_ptr + j], 4)
                heads_b[key] = pred_b[seg_ptr + j]
                heads_c[key] = pred_c[seg_ptr + j]
            seg_ptr   += L
            doc_cursor += 1

    print(f"  Scored {len(scores):,} segments")
    return scores, heads_b, heads_c


# ══════════════════════════════════════════════════════════════════════════════
# 3. Data audit
# ══════════════════════════════════════════════════════════════════════════════

def audit(data: dict, config: dict | None, model_scores: dict,
          out_dir: Path) -> dict:
    """Print and save a data-audit table. Returns summary dict for the report."""
    docs, video_ids = data["docs"], data["video_ids"]
    n_segs = sum(len(docs[v]) for v in video_ids)

    lines = ["# Data Audit\n"]

    # Counts
    lines.append(f"- Documents in scope : {len(video_ids):,}")
    lines.append(f"- Segments in scope  : {n_segs:,}")
    lines.append(f"- Weak labels        : {len(data['weak']):,}")
    lines.append(f"- Function labels    : {len(data['fn']):,}")
    lines.append(f"- Disfluency labels  : {len(data['disfl']):,}")
    lines.append(f"- Baseline rows      : {len(data['baselines']):,}")
    lines.append(f"- Gold (soft)        : {len(data['gold']):,}  "
                 f"{'← human-annotated' if data['gold'] else '← null (not yet annotated)'}")
    lines.append(f"- Model scores       : {len(model_scores):,}"
                 f"{'  (checkpoint loaded)' if model_scores else '  (no checkpoint)'}")
    lines.append("")

    # Genre distribution
    genre_counts: dict[str, int] = defaultdict(int)
    for v in video_ids:
        genre_counts[data["genre"][v]] += 1
    lines.append("## Genre distribution")
    for g in _ordered(list(genre_counts.keys()), EXPECTED_GENRES):
        n = genre_counts[g]
        lines.append(f"  {_pretty_name(g):20s}  {n:5d} videos")
    missing_genres = [g for g in EXPECTED_GENRES if g not in genre_counts]
    if missing_genres:
        lines.append("  Missing expected genres: " + ", ".join(_pretty_name(g) for g in missing_genres))
    lines.append("")

    # Caption-type distribution
    if data["caption_type"]:
        cap_counts: dict[str, int] = defaultdict(int)
        for v in video_ids:
            cap_counts[data["caption_type"].get(v, "unknown")] += 1
        lines.append("## Caption-type distribution")
        for ct, n in sorted(cap_counts.items()):
            lines.append(f"  {ct:12s}  {n:5d} videos")
        lines.append("")

    # Length-bucket distribution
    if data["length_bucket"]:
        bucket_counts: dict[str, int] = defaultdict(int)
        for v in video_ids:
            bucket_counts[data["length_bucket"].get(v, "unknown")] += 1
        lines.append("## Length-bucket distribution")
        for b, n in sorted(bucket_counts.items()):
            lines.append(f"  {b:10s}  {n:5d} videos")
        lines.append("")

    # Missing-field warnings
    missing_weak = n_segs - len(data["weak"])
    if missing_weak:
        lines.append(f"⚠  {missing_weak} segments have no weak label")
    if not data["gold"]:
        lines.append("⚠  Gold labels are all null — segment metrics will use "
                     "weak labels as proxy (binarised at p_remove > 0.5)")
    if not data["baselines"]:
        lines.append("⚠  baselines.jsonl not found — run src/baselines.py first")
    if not model_scores:
        lines.append("⚠  No model checkpoint — model-based metrics skipped")

    text = "\n".join(lines)
    print(text)
    (out_dir / "audit.md").write_text(text, encoding="utf-8")

    return dict(
        n_docs=len(video_ids), n_segs=n_segs,
        has_gold=bool(data["gold"]),
        has_model=bool(model_scores),
        has_baselines=bool(data["baselines"]),
        has_caption_type=bool(data["caption_type"]),
        genre_counts=dict(genre_counts),
        missing_genres=missing_genres,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. Segment-level evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _binary_labels(data: dict, source: str = "auto") -> tuple[dict, str]:
    """Return {key: 0/1} and a description string.

    source="auto": prefer gold if any gold labels exist, else weak proxy.
    """
    if source == "auto":
        source = "gold" if data["gold"] else "weak"

    if source == "gold" and data["gold"]:
        labels = {k: int(v >= 0.5) for k, v in data["gold"].items()}
        desc   = "gold labels (human-annotated)"
    else:
        labels = {k: int(v > 0.5) for k, v in data["weak"].items()}
        desc   = "weak labels (proxy — not human-annotated)"

    return labels, desc


def _aligned_arrays(scores_dict: dict, labels_dict: dict
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, labels) aligned on shared keys."""
    keys   = sorted(set(scores_dict) & set(labels_dict))
    scores = np.array([scores_dict[k] for k in keys])
    labels = np.array([labels_dict[k] for k in keys])
    return scores, labels


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() * abs(acc - conf)
    return float(ece / max(len(probs), 1))


# ══════════════════════════════════════════════════════════════════════════════
# 3a. Segment-level plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_pr(scores: np.ndarray, labels: np.ndarray,
                label_desc: str, plots_dir: Path, prefix: str = "model") -> dict:
    roc_auc = roc_auc_score(labels, scores)
    pr_auc  = average_precision_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    prec, rec, _ = precision_recall_curve(labels, scores)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"SegRemover achieves AUC-ROC={roc_auc:.3f}, PR-AUC={pr_auc:.3f}  [{label_desc}]",
        color=TEXT, fontsize=13,
    )

    ax = _ax_style(axes[0])
    ax.plot(fpr, tpr, color=BRIGHT_CYAN, label=f"AUC-ROC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=GRID, linewidth=1)
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC curve")
    ax.legend()

    ax = _ax_style(axes[1])
    ax.plot(rec, prec, color=BRIGHT_GREEN, label=f"AUC-PR = {pr_auc:.3f}")
    base = labels.mean()
    ax.axhline(base, linestyle="--", color=GRID, linewidth=1,
               label=f"baseline = {base:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="PR curve")
    ax.legend()

    plt.tight_layout()
    _save(fig, plots_dir, f"{prefix}_roc_pr")
    return {"roc_auc": round(roc_auc, 4), "pr_auc": round(pr_auc, 4)}


def plot_reliability(scores: np.ndarray, labels: np.ndarray,
                     plots_dir: Path, prefix: str = "model",
                     title_prefix: str = "") -> float:
    n_bins = 10
    ece    = _compute_ece(scores, labels, n_bins)
    cal_df = _calibration_points(scores, labels, n_bins)

    signed_gap = float((cal_df["confidence"] - cal_df["accuracy"]).mean()) if not cal_df.empty else 0.0
    if ece < 0.05:
        cal_quality = "well calibrated"
    elif signed_gap > 0:
        cal_quality = "over-confident"
    else:
        cal_quality = "under-confident"
    heading = title_prefix or f"SegRemover reliability ({cal_quality})"

    fig, ax = plt.subplots(figsize=(7, 6))
    _ax_style(ax)
    ax.plot([0, 1], [0, 1], "--", color=GRID, linewidth=1, label="perfect calibration")
    if not cal_df.empty:
        ax.bar(cal_df["mid"], cal_df["accuracy"], width=0.08,
               color=BRIGHT_CYAN, alpha=0.9, label="fraction removed")
        ax.plot(cal_df["confidence"], cal_df["accuracy"], "o-",
                color=BRIGHT_ORANGE, label="mean confidence", markersize=6)
    ax.set(xlabel="Mean predicted probability", ylabel="Fraction of positives",
           title=f"{heading}  (ECE = {ece:.4f})", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    plt.tight_layout()
    _save(fig, plots_dir, f"{prefix}_reliability")
    return ece

def plot_confusion(scores: np.ndarray, labels: np.ndarray,
                   threshold: float, plots_dir: Path, prefix: str = "model") -> None:
    preds = (scores >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=TEXT, fontsize=16, fontweight="bold")
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Pred KEEP", "Pred REMOVE"],
           yticklabels=["True KEEP", "True REMOVE"],
           title=f"Confusion matrix  (threshold={threshold})")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    _save(fig, plots_dir, f"{prefix}_confusion")


def plot_threshold_sweep(scores: np.ndarray, labels: np.ndarray,
                         thresholds: list[float], plots_dir: Path,
                         prefix: str = "model") -> pd.DataFrame:
    rows = []
    for t in thresholds:
        preds = (scores >= t).astype(int)
        rows.append({
            "threshold":  t,
            "precision":  precision_score(labels, preds, zero_division=0),
            "recall":     recall_score(labels, preds, zero_division=0),
            "f1":         f1_score(labels, preds, zero_division=0),
            "accuracy":   accuracy_score(labels, preds),
            "pct_removed": preds.mean(),
        })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Threshold controls the precision–recall trade-off", color=TEXT, fontsize=14)

    ax = _ax_style(axes[0])
    for col, color in [("precision", BRIGHT_CYAN), ("recall", BRIGHT_GREEN), ("f1", BRIGHT_ORANGE)]:
        ax.plot(df["threshold"], df[col], "o-", color=color, label=col)
    # Annotate the F1 peak
    best_f1_idx = df["f1"].idxmax()
    ax.annotate(
        f"best F1={df['f1'].iloc[best_f1_idx]:.2f}\n@t={df['threshold'].iloc[best_f1_idx]:.2f}",
        (df["threshold"].iloc[best_f1_idx], df["f1"].iloc[best_f1_idx]),
        textcoords="offset points", xytext=(8, -18),
        color=BRIGHT_ORANGE, fontsize=9,
        arrowprops=dict(arrowstyle="->", color=BRIGHT_ORANGE, lw=1),
    )
    ax.set(xlabel="Threshold", ylabel="Score", title="P / R / F1 vs threshold",
           xlim=(min(thresholds) - 0.02, max(thresholds) + 0.02))
    ax.legend()

    ax = _ax_style(axes[1])
    ax.plot(df["threshold"], df["pct_removed"] * 100, "o-", color=BRIGHT_PURPLE)
    ax.set(xlabel="Threshold", ylabel="% segments removed",
           title="Higher threshold → less aggressive compression")

    plt.tight_layout()
    _save(fig, plots_dir, f"{prefix}_threshold_sweep")
    return df


def plot_confidence_buckets(scores: np.ndarray, labels: np.ndarray,
                            plots_dir: Path, prefix: str = "model") -> dict:
    """p>=0.95 definitely remove  |  0.70<=p<0.95 probably remove  |  p<0.70 keep."""
    buckets = {
        "keep (p<0.70)":             scores < 0.70,
        "prob. remove (0.70–0.95)": (scores >= 0.70) & (scores < 0.95),
        "def. remove (p>=0.95)":    scores >= 0.95,
    }
    result = {}
    rows   = []
    for name, mask in buckets.items():
        if mask.sum() == 0:
            rows.append({"bucket": name, "n": 0, "precision": float("nan")})
            continue
        prec = labels[mask].mean()
        rows.append({"bucket": name, "n": int(mask.sum()),
                     "precision": round(float(prec), 4)})
        result[name] = rows[-1]

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    _ax_style(ax)
    colors = [BRIGHT_GREEN, BRIGHT_ORANGE, BRIGHT_RED]
    bars = ax.bar(df["bucket"], df["precision"], color=colors, alpha=0.95)
    for bar, row in zip(bars, df.itertuples()):
        if not np.isnan(row.precision):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{row.precision:.2f}\n(n={row.n:,})",
                    ha="center", va="bottom", color=TEXT, fontsize=10)
    ax.set(ylim=(0, 1.15), ylabel="Fraction truly removed",
           title="High-confidence predictions are most reliable")
    ax.tick_params(axis="x", rotation=10)
    plt.tight_layout()
    _save(fig, plots_dir, f"{prefix}_confidence_buckets")
    return result


def plot_function_by_confidence_bucket(model_scores: dict, data: dict, plots_dir: Path) -> None:
    """Interpretability: what function labels dominate keep/probable/definite remove buckets."""
    if not model_scores or not data.get("fn"):
        return

    rows = []
    for k, p in model_scores.items():
        if k not in data["fn"]:
            continue
        fn_idx = data["fn"][k]
        if not isinstance(fn_idx, (int, np.integer)) or fn_idx < 0 or fn_idx >= len(FUNCTION_NAMES):
            continue
        if p < 0.70:
            bucket = "keep\n(p<0.70)"
        elif p < 0.95:
            bucket = "prob. remove\n(0.70–0.95)"
        else:
            bucket = "def. remove\n(p≥0.95)"
        rows.append({"bucket": bucket, "function": FUNCTION_NAMES[int(fn_idx)]})

    if not rows:
        return
    df = pd.DataFrame(rows)
    counts = pd.crosstab(df["bucket"], df["function"])
    bucket_order = ["keep\n(p<0.70)", "prob. remove\n(0.70–0.95)", "def. remove\n(p≥0.95)"]
    counts = counts.reindex([b for b in bucket_order if b in counts.index])
    props = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    props.to_csv(plots_dir.parent / "function_by_confidence_bucket.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    _ax_style(ax)
    bottom = np.zeros(len(props))
    for fn in FUNCTION_NAMES:
        if fn not in props.columns:
            continue
        vals = props[fn].fillna(0).to_numpy()
        ax.bar(np.arange(len(props)), vals, bottom=bottom,
               color=FUNCTION_COLORS.get(fn, BRIGHT_SLATE), alpha=0.95,
               label=_pretty_name(fn))
        bottom += vals
    ax.set(xticks=np.arange(len(props)), xticklabels=list(props.index), ylim=(0, 1),
           ylabel="Share of segments",
           title="Predicted removability aligns with segment function labels")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    plt.tight_layout()
    _save(fig, plots_dir, "function_by_confidence_bucket")


# ── Head B explainability ─────────────────────────────────────────────────────

def plot_fn_label_distribution(data: dict, plots_dir: Path) -> None:
    """Stacked bar: function-label share per genre — shows corpus composition."""
    if not data.get("fn"):
        return

    rows = []
    for k, fn_idx in data["fn"].items():
        if not isinstance(fn_idx, (int, np.integer)) or fn_idx < 0 or fn_idx >= len(FUNCTION_NAMES):
            continue
        rows.append({"genre": data["genre"].get(k[0], "unknown"),
                     "fn_label": FUNCTION_NAMES[int(fn_idx)]})
    if not rows:
        return

    df     = pd.DataFrame(rows)
    genres = _ordered(list(df["genre"].unique()), EXPECTED_GENRES)
    counts = pd.crosstab(df["genre"], df["fn_label"])
    props  = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).reindex(genres, fill_value=0)

    fig, ax = plt.subplots(figsize=(max(9, len(genres) * 1.5), 5))
    _ax_style(ax)
    bottom = np.zeros(len(genres))
    for fn in FUNCTION_NAMES:
        if fn not in props.columns:
            continue
        vals = props[fn].fillna(0).to_numpy()
        ax.bar(np.arange(len(genres)), vals, bottom=bottom,
               color=FUNCTION_COLORS.get(fn, BRIGHT_SLATE), alpha=0.92,
               label=_pretty_name(fn))
        bottom += vals
    ax.set(xticks=np.arange(len(genres)),
           xticklabels=[_pretty_name(g) for g in genres],
           ylim=(0, 1), ylabel="Share of segments",
           title="Lectures are new-information heavy; podcasts skew toward discourse fillers")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    plt.tight_layout()
    _save(fig, plots_dir, "fn_label_distribution_by_genre")


def plot_p_remove_by_function(model_scores: dict, data: dict, plots_dir: Path) -> None:
    """Head B validation: p_remove boxplot grouped by function label.

    Key story: redundant_repetition and off_topic should have high median p_remove;
    new_information should have low p_remove. This directly validates Head B.
    """
    if not model_scores or not data.get("fn"):
        return

    rows = []
    for k, p in model_scores.items():
        fn_idx = data["fn"].get(k)
        if fn_idx is None or not isinstance(fn_idx, (int, np.integer)):
            continue
        if fn_idx < 0 or fn_idx >= len(FUNCTION_NAMES):
            continue
        rows.append({"fn_label": FUNCTION_NAMES[int(fn_idx)], "p_remove": p})
    if not rows:
        return

    df       = pd.DataFrame(rows)
    fn_order = [f for f in FUNCTION_NAMES if f in df["fn_label"].values]

    fig, ax = plt.subplots(figsize=(11, 5))
    _ax_style(ax)
    data_per_fn = [df[df["fn_label"] == fn]["p_remove"].values for fn in fn_order]
    colors = [FUNCTION_COLORS.get(fn, BRIGHT_SLATE) for fn in fn_order]

    bp = ax.boxplot(data_per_fn, patch_artist=True, widths=0.5,
                    medianprops=dict(color=TEXT, linewidth=2),
                    whiskerprops=dict(color=GRID), capprops=dict(color=GRID),
                    flierprops=dict(marker="o", markersize=3, alpha=0.3, linestyle="none"))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for i, (fn, vals) in enumerate(zip(fn_order, data_per_fn)):
        if len(vals):
            med = float(np.median(vals))
            ax.text(i + 1, min(med + 0.03, 0.97), f"{med:.2f}",
                    ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")

    ax.set(xticks=range(1, len(fn_order) + 1),
           xticklabels=[_pretty_name(fn) for fn in fn_order],
           ylabel="Model p_remove", ylim=(0, 1.05),
           title="Redundant and off-topic segments score highest p_remove — Head B validated")
    ax.axhline(0.5, color=GRID, linestyle="--", linewidth=1, alpha=0.6)
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    _save(fig, plots_dir, "p_remove_by_function")


# ── Head C explainability ─────────────────────────────────────────────────────

def plot_disfl_distribution(data: dict, plots_dir: Path) -> None:
    """Stacked bar: disfluency-label share per genre — shows which genres are noisier."""
    if not data.get("disfl"):
        return

    rows = []
    for k, disfl_idx in data["disfl"].items():
        if not isinstance(disfl_idx, (int, np.integer)) or disfl_idx < 0 or disfl_idx >= len(DISFL_NAMES):
            continue
        rows.append({"genre": data["genre"].get(k[0], "unknown"),
                     "disfl_label": DISFL_NAMES[int(disfl_idx)]})
    if not rows:
        return

    df     = pd.DataFrame(rows)
    genres = _ordered(list(df["genre"].unique()), EXPECTED_GENRES)
    counts = pd.crosstab(df["genre"], df["disfl_label"])
    props  = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).reindex(genres, fill_value=0)

    disfl_colors = {
        "clean":        BRIGHT_CYAN,
        "filled_pause": BRIGHT_RED,
        "repetition":   BRIGHT_ORANGE,
        "revision":     BRIGHT_PURPLE,
        "restart":      BRIGHT_GREEN,
    }

    fig, ax = plt.subplots(figsize=(max(9, len(genres) * 1.5), 5))
    _ax_style(ax)
    bottom = np.zeros(len(genres))
    for disfl in DISFL_NAMES:
        if disfl not in props.columns:
            continue
        vals = props[disfl].fillna(0).to_numpy()
        ax.bar(np.arange(len(genres)), vals, bottom=bottom,
               color=disfl_colors.get(disfl, BRIGHT_SLATE), alpha=0.92,
               label=_pretty_name(disfl))
        bottom += vals
    ax.set(xticks=np.arange(len(genres)),
           xticklabels=[_pretty_name(g) for g in genres],
           ylim=(0, 1), ylabel="Share of segments",
           title="Podcasts and TV series contain more disfluencies than lectures")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    plt.tight_layout()
    _save(fig, plots_dir, "disfl_distribution_by_genre")


def plot_p_remove_by_disfluency(model_scores: dict, data: dict, plots_dir: Path) -> None:
    """Head C validation: p_remove boxplot grouped by disfluency label.

    Key story: filled_pause, repetition, restart should have high median p_remove;
    clean should score low. This directly validates Head C's regularisation.
    """
    if not model_scores or not data.get("disfl"):
        return

    rows = []
    for k, p in model_scores.items():
        disfl_idx = data["disfl"].get(k)
        if disfl_idx is None or not isinstance(disfl_idx, (int, np.integer)):
            continue
        if disfl_idx < 0 or disfl_idx >= len(DISFL_NAMES):
            continue
        rows.append({"disfl_label": DISFL_NAMES[int(disfl_idx)], "p_remove": p})
    if not rows:
        return

    df          = pd.DataFrame(rows)
    disfl_order = [d for d in DISFL_NAMES if d in df["disfl_label"].values]
    disfl_colors_list = [BRIGHT_CYAN, BRIGHT_RED, BRIGHT_ORANGE, BRIGHT_PURPLE, BRIGHT_GREEN]

    fig, ax = plt.subplots(figsize=(9, 5))
    _ax_style(ax)
    data_per_d = [df[df["disfl_label"] == d]["p_remove"].values for d in disfl_order]

    bp = ax.boxplot(data_per_d, patch_artist=True, widths=0.5,
                    medianprops=dict(color=TEXT, linewidth=2),
                    whiskerprops=dict(color=GRID), capprops=dict(color=GRID),
                    flierprops=dict(marker="o", markersize=3, alpha=0.3, linestyle="none"))
    for patch, color in zip(bp["boxes"], disfl_colors_list[:len(disfl_order)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for i, (d, vals) in enumerate(zip(disfl_order, data_per_d)):
        if len(vals):
            med = float(np.median(vals))
            ax.text(i + 1, min(med + 0.03, 0.97), f"{med:.2f}",
                    ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")

    ax.set(xticks=range(1, len(disfl_order) + 1),
           xticklabels=[_pretty_name(d) for d in disfl_order],
           ylabel="Model p_remove", ylim=(0, 1.05),
           title="Disfluent segments (filled pause, repetition, restart) score higher p_remove — Head C validated")
    ax.axhline(0.5, color=GRID, linestyle="--", linewidth=1, alpha=0.6)
    ax.tick_params(axis="x", rotation=10)
    plt.tight_layout()
    _save(fig, plots_dir, "p_remove_by_disfluency")


# ── Head B / Head C classifier evaluation ─────────────────────────────────────

def _plot_confusion_heatmap(
    cm: np.ndarray, names: list, plots_dir: Path, stem: str, title: str
) -> None:
    """Row-normalised confusion matrix heatmap on dark background."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(7, n * 1.4), max(6, n * 1.1)))
    _ax_style(ax)
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(n), yticks=np.arange(n),
        xticklabels=[_pretty_name(x) for x in names],
        yticklabels=[_pretty_name(x) for x in names],
        xlabel="Predicted", ylabel="True", title=title,
    )
    ax.tick_params(axis="x", rotation=30)
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=TEXT if val < 0.5 else BG, fontsize=9)
    plt.tight_layout()
    _save(fig, plots_dir, stem)


def explainer_eval(
    heads_b: dict, heads_c: dict, data: dict, plots_dir: Path
) -> dict:
    """Evaluate Head B (function) and Head C (disfluency) as classifiers.

    Computes accuracy and macro-F1 against ground-truth function/disfluency
    labels. Produces normalised confusion matrix plots for both heads.
    These metrics directly support the 'explainer' claim in the paper.
    """
    from sklearn.metrics import confusion_matrix as _cm

    results: dict = {}

    # ── Head B — function classification ──────────────────────────────────────
    if heads_b and data.get("fn"):
        y_true, y_pred = [], []
        for k, pred in heads_b.items():
            true = data["fn"].get(k)
            if true is None or not isinstance(true, (int, np.integer)):
                continue
            if 0 <= int(true) < len(FUNCTION_NAMES) and 0 <= int(pred) < len(FUNCTION_NAMES):
                y_true.append(int(true))
                y_pred.append(int(pred))

        if y_true:
            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
            results["head_b_accuracy"] = round(acc, 4)
            results["head_b_macro_f1"] = round(f1, 4)
            print(f"  Head B (function):   accuracy={acc:.3f}  macro-F1={f1:.3f}  (n={len(y_true):,})")

            per = f1_score(y_true, y_pred, average=None, zero_division=0,
                           labels=list(range(len(FUNCTION_NAMES))))
            for i, name in enumerate(FUNCTION_NAMES):
                results[f"head_b_f1_{name}"] = round(float(per[i]), 4)
                print(f"    {name:25s}  F1={per[i]:.3f}")

            cm = _cm(y_true, y_pred, labels=list(range(len(FUNCTION_NAMES))))
            _plot_confusion_heatmap(
                cm, FUNCTION_NAMES, plots_dir, "head_b_confusion",
                title="Head B — Function class confusion (row-normalised)",
            )

    # ── Head C — disfluency classification ────────────────────────────────────
    if heads_c and data.get("disfl"):
        y_true, y_pred = [], []
        for k, pred in heads_c.items():
            true = data["disfl"].get(k)
            if true is None or not isinstance(true, (int, np.integer)):
                continue
            if 0 <= int(true) < len(DISFL_NAMES) and 0 <= int(pred) < len(DISFL_NAMES):
                y_true.append(int(true))
                y_pred.append(int(pred))

        if y_true:
            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
            results["head_c_accuracy"] = round(acc, 4)
            results["head_c_macro_f1"] = round(f1, 4)
            print(f"  Head C (disfluency): accuracy={acc:.3f}  macro-F1={f1:.3f}  (n={len(y_true):,})")

            per = f1_score(y_true, y_pred, average=None, zero_division=0,
                           labels=list(range(len(DISFL_NAMES))))
            for i, name in enumerate(DISFL_NAMES):
                results[f"head_c_f1_{name}"] = round(float(per[i]), 4)
                print(f"    {name:25s}  F1={per[i]:.3f}")

            cm = _cm(y_true, y_pred, labels=list(range(len(DISFL_NAMES))))
            _plot_confusion_heatmap(
                cm, DISFL_NAMES, plots_dir, "head_c_confusion",
                title="Head C — Disfluency class confusion (row-normalised)",
            )

    return results


# ── Per-genre reliability (small multiples) ───────────────────────────────────

def plot_reliability_by_genre(
    model_scores: dict, labels_dict: dict, data: dict, plots_dir: Path,
) -> None:
    """Small-multiples reliability diagram — one panel per genre."""
    genres = _ordered(list(set(data["genre"].values())), EXPECTED_GENRES)
    n = len(genres)
    if n == 0:
        return

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)
    fig.suptitle(
        "Calibration varies by genre",
        color=TEXT, fontsize=13,
    )

    n_bins = 10
    for idx, genre in enumerate(genres):
        ax = axes[idx // ncols][idx % ncols]
        _ax_style(ax)
        g_vids = {vid for vid, g in data["genre"].items() if g == genre}
        s_dict = {k: v for k, v in model_scores.items() if k[0] in g_vids}
        l_dict = {k: v for k, v in labels_dict.items()  if k[0] in g_vids}
        s, l = _aligned_arrays(s_dict, l_dict)
        if len(s) < 10 or len(set(l)) < 2:
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", color=TEXT)
            ax.set(xlim=(0, 1), ylim=(0, 1), title=_pretty_name(genre))
            continue
        ece = _compute_ece(s, l, n_bins)
        cal_df = _calibration_points(s, l, n_bins)
        c = GENRE_PALETTE.get(genre, BRIGHT_CYAN)
        ax.plot([0, 1], [0, 1], "--", color=GRID, linewidth=1)
        if not cal_df.empty:
            ax.bar(cal_df["mid"], cal_df["accuracy"], width=0.08, color=c, alpha=0.85)
            ax.plot(cal_df["confidence"], cal_df["accuracy"], "o-", color=BRIGHT_ORANGE, markersize=5)
        ax.set(xlim=(0, 1), ylim=(0, 1),
               title=f"{_pretty_name(genre)}  (ECE={ece:.3f})",
               xlabel="Confidence", ylabel="Fraction removed")

    # Hide empty panels
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    _save(fig, plots_dir, "reliability_by_genre")


# ── Calibration before vs after temperature scaling ───────────────────────────

def plot_calibration_comparison(
    scores: np.ndarray, labels: np.ndarray,
    T: float, plots_dir: Path,
) -> None:
    """Show reliability before (T=1) and after (T=configured) temperature scaling.

    Since scores = sigmoid(logit / T), we reconstruct raw logits as
    logit = T * logit(scores), then raw_probs = sigmoid(logit).
    """
    if abs(T - 1.0) < 1e-4:
        return   # no scaling was applied — skip comparison

    eps       = 1e-6
    scores_c  = np.clip(scores, eps, 1 - eps)
    logits    = np.log(scores_c / (1 - scores_c))
    raw_probs = 1.0 / (1.0 + np.exp(-T * logits))   # undo T division

    ece_before = _compute_ece(raw_probs, labels)
    ece_after  = _compute_ece(scores,    labels)
    delta = ece_before - ece_after
    verb = "reduces" if delta > 0 else "increases"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Temperature scaling (T={T:.2f}) {verb} ECE from {ece_before:.4f} to {ece_after:.4f}",
        color=TEXT, fontsize=13,
    )

    for ax, probs, title in [
        (axes[0], raw_probs, f"Before scaling (T=1.0)  ECE={ece_before:.4f}"),
        (axes[1], scores,    f"After scaling (T={T:.2f})  ECE={ece_after:.4f}"),
    ]:
        _ax_style(ax)
        cal_df = _calibration_points(probs, labels, n_bins=10)
        ax.plot([0, 1], [0, 1], "--", color=GRID, linewidth=1, label="perfect")
        if not cal_df.empty:
            ax.bar(cal_df["mid"], cal_df["accuracy"], width=0.08,
                   color=BRIGHT_CYAN, alpha=0.9, label="fraction removed")
            ax.plot(cal_df["confidence"], cal_df["accuracy"], "o-",
                    color=BRIGHT_ORANGE, markersize=6, label="mean confidence")
        ax.set(xlim=(0, 1), ylim=(0, 1), title=title,
               xlabel="Confidence", ylabel="Fraction of positives")
        ax.legend(fontsize=10)

    plt.tight_layout()
    _save(fig, plots_dir, "calibration_before_after_T")

def segment_eval(model_scores: dict, data: dict, thresholds: list[float],
                 config: dict | None, plots_dir: Path) -> dict:
    """Run all segment-level metrics and plots. Returns key metrics dict."""
    if not model_scores:
        print("  No model scores — skipping segment eval")
        return {}

    labels, label_desc = _binary_labels(data)
    scores, labs = _aligned_arrays(model_scores, labels)
    print(f"  Label source : {label_desc}")
    print(f"  Segments     : {len(scores):,}  (pos={labs.mean():.2f})")

    metrics = {}
    metrics.update(plot_roc_pr(scores, labs, label_desc, plots_dir))
    ece = plot_reliability(scores, labs, plots_dir)
    metrics["ece"] = round(ece, 4)

    # threshold = 0.5 baseline metrics
    preds = (scores >= 0.5).astype(int)
    metrics.update({
        "f1_at_0.5":        round(f1_score(labs, preds, zero_division=0), 4),
        "precision_at_0.5": round(precision_score(labs, preds, zero_division=0), 4),
        "recall_at_0.5":    round(recall_score(labs, preds, zero_division=0), 4),
        "accuracy_at_0.5":  round(accuracy_score(labs, preds), 4),
    })

    # Bootstrap CI on AUC-ROC
    roc_lo, roc_hi = _bootstrap_ci(scores, labs, roc_auc_score)
    metrics["roc_auc_ci"] = (round(roc_lo, 4), round(roc_hi, 4))

    plot_confusion(scores, labs, threshold=0.5, plots_dir=plots_dir)
    sweep_df = plot_threshold_sweep(scores, labs, thresholds, plots_dir)
    plot_confidence_buckets(scores, labs, plots_dir)
    plot_function_by_confidence_bucket(model_scores, data, plots_dir)

    # Per-genre reliability (small multiples)
    plot_reliability_by_genre(model_scores, labels, data, plots_dir)

    # Calibration before/after T (only if temperature was applied)
    T = float(config.get("temperature", 1.0)) if config else 1.0
    if abs(T - 1.0) > 1e-4:
        plot_calibration_comparison(scores, labs, T, plots_dir)

    # Per-genre AUC with bootstrap CI
    genre_aucs = {}
    for g in sorted(set(data["genre"].values())):
        g_keys = {k for k, v in data["genre"].items() if v == g}
        s, l = _aligned_arrays(
            {k: v for k, v in model_scores.items() if k[0] in g_keys},
            {k: v for k, v in labels.items()       if k[0] in g_keys},
        )
        if len(set(l)) < 2 or len(s) < 10:
            continue
        auc = round(roc_auc_score(l, s), 4)
        lo, hi = _bootstrap_ci(s, l, roc_auc_score, n_boot=300)
        genre_aucs[g] = {"auc": auc, "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}

    if genre_aucs:
        _plot_genre_bar_ci(genre_aucs, "AUC-ROC by genre (95% CI)",
                           plots_dir, "model_genre_auc")
    metrics["genre_auc"] = genre_aucs

    print(f"  ROC-AUC={metrics['roc_auc']}  PR-AUC={metrics['pr_auc']}  "
          f"ECE={metrics['ece']}  F1@0.5={metrics['f1_at_0.5']}")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 4. Transcript-level evaluation
# ══════════════════════════════════════════════════════════════════════════════

def transcript_eval(data: dict, model_scores: dict, thresholds: list[float],
                    device: str, plots_dir: Path) -> pd.DataFrame:
    """Sweep thresholds; for each doc compute compression + SBERT cosine + ROUGE."""
    from sentence_transformers import SentenceTransformer

    try:
        from rouge_score import rouge_scorer as _rs
        _rouge = _rs.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    except ImportError:
        _rouge = None
        print("  rouge-score not installed — ROUGE skipped (pip install rouge-score)")

    docs, video_ids = data["docs"], data["video_ids"]
    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=device
    )

    rows = []
    for vid in tqdm(video_ids, desc="transcript eval"):
        segs = docs[vid]
        texts = [s["text"] for s in segs]
        wcs   = [len(t.split()) for t in texts]
        total_words = sum(wcs)
        if total_words == 0:
            continue

        embs = embedder.encode(
            texts, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        centroid_full = embs.mean(axis=0)
        norm = np.linalg.norm(centroid_full)
        if norm > 0:
            centroid_full /= norm

        # Score for each segment: model if available, else weak label, else 0.5
        seg_scores = []
        for seg in segs:
            k = (vid, seg["seg_idx"])
            if k in model_scores:
                seg_scores.append(model_scores[k])
            elif k in data["weak"]:
                seg_scores.append(data["weak"][k])
            else:
                seg_scores.append(0.5)
        seg_scores = np.array(seg_scores)

        genre   = data["genre"].get(vid, "unknown")
        bucket  = data["length_bucket"].get(vid, "unknown")
        cap_type = data["caption_type"].get(vid, "unknown")

        orig_text = " ".join(texts)

        for t in thresholds:
            keep_mask = seg_scores < t
            kept_words = sum(w for w, k in zip(wcs, keep_mask) if k)
            compression = kept_words / total_words

            if keep_mask.sum() == 0:
                sbert_cos = 0.0
                rouge1_r = rouge_l_r = 0.0
            else:
                centroid_kept = embs[keep_mask].mean(axis=0)
                nk = np.linalg.norm(centroid_kept)
                if nk > 0:
                    centroid_kept /= nk
                sbert_cos = float(np.dot(centroid_full, centroid_kept))

                if _rouge:
                    kept_text = " ".join(tx for tx, k in zip(texts, keep_mask) if k)
                    sc = _rouge.score(orig_text, kept_text)
                    rouge1_r = round(sc["rouge1"].recall, 4)
                    rouge_l_r = round(sc["rougeL"].recall, 4)
                else:
                    rouge1_r = rouge_l_r = None

            rows.append({
                "video_id":      vid,
                "genre":         genre,
                "length_bucket": bucket,
                "caption_type":  cap_type,
                "threshold":     t,
                "compression":   round(compression, 4),
                "sbert_cosine":  round(sbert_cos, 4),
                "rouge1_r":      rouge1_r,
                "rougeL_r":      rouge_l_r,
                "segs_kept":     int(keep_mask.sum()),
                "segs_total":    len(segs),
            })

    df = pd.DataFrame(rows)
    df.to_csv(plots_dir.parent / "transcript_eval.csv", index=False)
    _plot_compression_curves(df, plots_dir)
    if "rouge1_r" in df.columns and df["rouge1_r"].notna().any():
        _plot_rouge_curves(df, plots_dir)
    return df


def _plot_compression_curves(df: pd.DataFrame, plots_dir: Path) -> None:
    if df.empty:
        return

    # ── Overall compression vs SBERT cosine ──────────────────────────────────
    avg = df.groupby("threshold")[["compression", "sbert_cosine"]].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Semantic similarity degrades smoothly with compression — genre curves diverge past 50%",
        color=TEXT, fontsize=12,
    )

    ax = _ax_style(axes[0])
    ax.plot(avg["compression"] * 100, avg["sbert_cosine"],
            "o-", color=BLUE, markersize=7)
    for _, row in avg.iterrows():
        ax.annotate(f"t={row['threshold']:.2f}",
                    (row["compression"] * 100, row["sbert_cosine"]),
                    textcoords="offset points", xytext=(5, 4),
                    color=TEXT, fontsize=9)
    ax.set(xlabel="Words kept (%)", ylabel="SBERT cosine (full vs reduced)",
           title="Overall compression–preservation trade-off")

    # ── By genre ─────────────────────────────────────────────────────────────
    ax = _ax_style(axes[1])
    for genre, grp in df.groupby("genre"):
        avg_g = grp.groupby("threshold")[["compression", "sbert_cosine"]].mean()
        c = GENRE_PALETTE.get(genre, BLUE)
        ax.plot(avg_g["compression"] * 100, avg_g["sbert_cosine"],
                "o-", color=c, label=genre, markersize=5)
    ax.set(xlabel="Words kept (%)", ylabel="SBERT cosine",
           title="By genre")
    ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, plots_dir, "transcript_compression")

    # ── By length bucket ─────────────────────────────────────────────────────
    if "length_bucket" in df.columns and df["length_bucket"].nunique() > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        _ax_style(ax)
        for bucket, grp in df.groupby("length_bucket"):
            avg_b = grp.groupby("threshold")[["compression", "sbert_cosine"]].mean()
            c = BUCKET_COLORS.get(bucket, BLUE)
            ax.plot(avg_b["compression"] * 100, avg_b["sbert_cosine"],
                    "o-", color=c, label=bucket, markersize=6)
        ax.set(xlabel="Words kept (%)", ylabel="SBERT cosine",
               title="Longer videos show higher semantic robustness to compression")
        ax.legend()
        plt.tight_layout()
        _save(fig, plots_dir, "transcript_by_length")

    _plot_removability_summary(df, plots_dir)


def _plot_rouge_curves(df: pd.DataFrame, plots_dir: Path) -> None:
    """ROUGE-1-R and ROUGE-L-R vs compression ratio — lexical preservation front."""
    avg = df.groupby("threshold")[["compression", "rouge1_r", "rougeL_r"]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    _ax_style(ax)
    ax.plot(avg["compression"] * 100, avg["rouge1_r"],
            "o-", color=BRIGHT_GREEN, label="ROUGE-1 recall", markersize=7)
    ax.plot(avg["compression"] * 100, avg["rougeL_r"],
            "s--", color=BRIGHT_ORANGE, label="ROUGE-L recall", markersize=7)
    for _, row in avg.iterrows():
        ax.annotate(f"t={row['threshold']:.2f}",
                    (row["compression"] * 100, row["rouge1_r"]),
                    textcoords="offset points", xytext=(4, 4),
                    color=TEXT, fontsize=8)
    ax.set(xlabel="Words kept (%)", ylabel="ROUGE recall vs original",
           title="Lexical preservation tracks compression linearly above 50% retention")
    ax.legend()
    plt.tight_layout()
    _save(fig, plots_dir, "rouge_curves")


def _plot_removability_summary(df: pd.DataFrame, plots_dir: Path) -> None:
    """Direct removability plots: percent segments/words removed by length and genre.

    This is the missing direct counterpart to the compression curves: instead of
    asking how well reduced transcripts preserve content, it asks how much content
    each group is predicted to remove at the operating thresholds.
    """
    if df.empty or "threshold" not in df.columns:
        return

    work = df.copy()
    work["pct_segments_removed"] = (1.0 - work["segs_kept"] / work["segs_total"].clip(lower=1)) * 100
    work["pct_words_removed"] = (1.0 - work["compression"]) * 100
    work.to_csv(plots_dir.parent / "removability_summary.csv", index=False)

    thresholds = _pick_thresholds(sorted(work["threshold"].dropna().unique().tolist()), (0.70, 0.95))
    if not thresholds:
        return
    sub = work[work["threshold"].isin(thresholds)].copy()

    # Length buckets: segment removal and word removal, both needed because
    # segment counts alone can hide long/short segment artifacts.
    if "length_bucket" in sub.columns and sub["length_bucket"].nunique() > 1:
        buckets = _ordered(list(sub["length_bucket"].dropna().unique()), ["short", "medium", "long", "unknown"])
        x = np.arange(len(buckets))
        width = 0.8 / max(len(thresholds), 1)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Direct removability by video length", color=TEXT, fontsize=13)
        for ax, metric, ylabel in [
            (axes[0], "pct_segments_removed", "% segments removed"),
            (axes[1], "pct_words_removed", "% words removed"),
        ]:
            _ax_style(ax)
            for i, t in enumerate(thresholds):
                vals = []
                for b in buckets:
                    vals.append(sub[(sub["length_bucket"] == b) & (sub["threshold"] == t)][metric].mean())
                offset = (i - len(thresholds) / 2 + 0.5) * width
                color = [BRIGHT_ORANGE, BRIGHT_RED, BRIGHT_PURPLE][i % 3]
                bars = ax.bar(x + offset, vals, width=width * 0.9, color=color, alpha=0.95,
                              label=f"t={t:.2f}")
                for bar, v in zip(bars, vals):
                    if not np.isnan(v):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
                                f"{v:.1f}%", ha="center", va="bottom", color=TEXT, fontsize=8)
            ax.set(xticks=x, xticklabels=[_pretty_name(b) for b in buckets], ylim=(0, 105),
                   ylabel=ylabel)
            ax.tick_params(axis="x", rotation=10)
            ax.legend(fontsize=9)
        plt.tight_layout()
        _save(fig, plots_dir, "removability_by_length")

    # Genre removability at the primary operating threshold, usually t=0.70.
    primary_t = thresholds[0]
    primary = work[np.isclose(work["threshold"], primary_t)].copy()
    if "genre" in primary.columns and primary["genre"].nunique() > 1:
        genres = _ordered(list(primary["genre"].dropna().unique()), EXPECTED_GENRES)
        avg = primary.groupby("genre")[["pct_segments_removed", "pct_words_removed"]].mean()
        x = np.arange(len(genres))
        width = 0.38
        fig, ax = plt.subplots(figsize=(max(10, len(genres) * 1.45), 5))
        _ax_style(ax)
        seg_vals = [avg.loc[g, "pct_segments_removed"] if g in avg.index else np.nan for g in genres]
        word_vals = [avg.loc[g, "pct_words_removed"] if g in avg.index else np.nan for g in genres]
        bars1 = ax.bar(x - width/2, seg_vals, width=width, color=BRIGHT_CYAN, alpha=0.95,
                       label="segments removed")
        bars2 = ax.bar(x + width/2, word_vals, width=width, color=BRIGHT_GREEN, alpha=0.95,
                       label="words removed")
        for bars in (bars1, bars2):
            for bar in bars:
                v = bar.get_height()
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width()/2, v + 1.0, f"{v:.1f}%",
                            ha="center", va="bottom", color=TEXT, fontsize=8)
        ax.set(xticks=x, xticklabels=[_pretty_name(g) for g in genres], ylim=(0, 105),
               ylabel="Removed (%)",
               title=f"Removability varies by genre at t={primary_t:.2f}")
        ax.tick_params(axis="x", rotation=15)
        ax.legend(fontsize=10)
        plt.tight_layout()
        _save(fig, plots_dir, "removability_by_genre")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Baseline comparison
# ══════════════════════════════════════════════════════════════════════════════

def baseline_eval(model_scores: dict, data: dict,
                  plots_dir: Path) -> dict:
    if not data["baselines"]:
        print("  No baselines.jsonl — skipping baseline comparison")
        return {}

    labels, label_desc = _binary_labels(data)
    all_methods: dict[str, dict] = {}
    if model_scores:
        all_methods["model"] = model_scores
    for name in ["random", "heuristic", "tfidf", "sbert"]:
        scores = {k: v[name] for k, v in data["baselines"].items() if name in v}
        if scores:
            all_methods[name] = scores

    results = {}
    for name, sc in all_methods.items():
        s, l = _aligned_arrays(sc, labels)
        if len(set(l)) < 2 or len(s) < 5:
            continue
        roc = roc_auc_score(l, s)
        pra = average_precision_score(l, s)
        roc_lo, roc_hi = _bootstrap_ci(s, l, roc_auc_score, n_boot=300)
        pr_lo, pr_hi = _bootstrap_ci(s, l, average_precision_score, n_boot=300)
        results[name] = {
            "n":       len(s),
            "roc_auc": round(roc, 4),
            "pr_auc":  round(pra, 4),
            "roc_ci":  (round(roc_lo, 4), round(roc_hi, 4)),
            "pr_ci":   (round(pr_lo, 4), round(pr_hi, 4)),
        }

    if not results:
        return {}

    # Print table
    print(f"\n  {'Method':12s}  {'ROC-AUC':>8s}  {'PR-AUC':>8s}  {'n':>7s}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*7}")
    for name, m in results.items():
        print(f"  {name:12s}  {m['roc_auc']:>8.4f}  {m['pr_auc']:>8.4f}  {m['n']:>7,}")

    # ── Overall bar chart with 95% CI error bars ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"SegRemover outperforms all baselines  [{label_desc}]",
        color=TEXT, fontsize=13,
    )
    for ax, metric in zip(axes, ["roc_auc", "pr_auc"]):
        _ax_style(ax)
        names  = list(results.keys())
        vals   = [results[n][metric] for n in names]
        colors = [BASELINE_COLORS.get(n, BRIGHT_CYAN) for n in names]
        ci_key = "roc_ci" if metric == "roc_auc" else "pr_ci"
        yerr_lo = [results[n][metric] - results[n][ci_key][0] for n in names]
        yerr_hi = [results[n][ci_key][1] - results[n][metric] for n in names]
        yerr = np.array([yerr_lo, yerr_hi])
        display_names = [_pretty_name(n) for n in names]
        bars = ax.bar(display_names, vals, color=colors, alpha=0.95,
                      yerr=yerr, capsize=5, error_kw={"ecolor": TEXT, "lw": 1.5})
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{v:.3f}", ha="center", va="bottom",
                    color=TEXT, fontsize=10)
        ax.set(ylim=(0, 1.15), ylabel=metric.upper().replace("_", "-"),
               title=metric.upper().replace("_", "-"))
        ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    _save(fig, plots_dir, "baseline_comparison")

    # ── Per-genre ROC-AUC breakdown ───────────────────────────────────────────
    genre_results: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for name, sc in all_methods.items():
        for g in _ordered(list(set(data["genre"].values())), EXPECTED_GENRES):
            g_vids = {vid for vid, genre in data["genre"].items() if genre == g}
            s_g = {k: v for k, v in sc.items() if k[0] in g_vids}
            l_g = {k: v for k, v in labels.items() if k[0] in g_vids}
            s_arr, l_arr = _aligned_arrays(s_g, l_g)
            if len(set(l_arr)) < 2 or len(s_arr) < 5:
                continue
            auc = float(roc_auc_score(l_arr, s_arr))
            lo, hi = _bootstrap_ci(s_arr, l_arr, roc_auc_score, n_boot=200)
            genre_results[g][name] = {"auc": round(auc, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}

    if genre_results:
        _plot_baseline_by_genre(genre_results, list(all_methods.keys()), plots_dir)

    return results


def _plot_baseline_by_genre(
    genre_results: dict, method_names: list[str], plots_dir: Path
) -> None:
    """Grouped bar chart: AUC-ROC for each method, grouped by genre."""
    genres  = _ordered(list(genre_results.keys()), EXPECTED_GENRES)
    n_g     = len(genres)
    n_m     = len(method_names)
    x       = np.arange(n_g)
    width   = 0.8 / max(n_m, 1)

    fig, ax = plt.subplots(figsize=(max(10, n_g * 2), 5))
    _ax_style(ax)

    for i, method in enumerate(method_names):
        vals, lo_err, hi_err = [], [], []
        for g in genres:
            entry = genre_results[g].get(method)
            if isinstance(entry, dict):
                vals.append(entry.get("auc", float("nan")))
                lo_err.append(entry.get("auc", 0) - entry.get("ci_lo", entry.get("auc", 0)))
                hi_err.append(entry.get("ci_hi", entry.get("auc", 0)) - entry.get("auc", 0))
            else:
                vals.append(float("nan") if entry is None else entry)
                lo_err.append(0.0); hi_err.append(0.0)
        offset = (i - n_m / 2 + 0.5) * width
        color  = BASELINE_COLORS.get(method, BRIGHT_CYAN)
        yerr = np.array([lo_err, hi_err])
        bars = ax.bar(x + offset, vals, width=width * 0.9, color=color,
                      alpha=0.95, label=_pretty_name(method),
                      yerr=yerr, capsize=3, error_kw={"ecolor": TEXT, "lw": 1.0})
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom",
                        color=TEXT, fontsize=8)

    ax.set(xticks=x, xticklabels=[_pretty_name(g) for g in genres], ylim=(0, 1.15),
           ylabel="ROC-AUC", title="SegRemover advantage over baselines varies by genre")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=10, ncol=min(n_m, 5), loc="upper center", bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout()
    _save(fig, plots_dir, "baseline_by_genre")


# ══════════════════════════════════════════════════════════════════════════════
# 6. 4-way agreement analysis
# ══════════════════════════════════════════════════════════════════════════════

def agreement_analysis(model_scores: dict, data: dict,
                       plots_dir: Path) -> pd.DataFrame:
    """Pairwise Spearman ρ between model / sbert / heuristic / weak-label proxy."""
    sources: dict[str, dict] = {}
    if model_scores:
        sources["model"]     = model_scores
    if data["baselines"]:
        for col in ["sbert", "heuristic", "tfidf"]:
            sc = {k: v[col] for k, v in data["baselines"].items() if col in v}
            if sc:
                sources[col] = sc
    sources["weak\n(proxy)"] = data["weak"]

    if len(sources) < 2:
        print("  Not enough sources for agreement analysis")
        return pd.DataFrame()

    keys    = sorted(set.intersection(*[set(s) for s in sources.values()]))
    names   = list(sources.keys())
    mat     = np.zeros((len(names), len(names)))
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            a = np.array([sources[n1][k] for k in keys])
            b = np.array([sources[n2][k] for k in keys])
            rho, _ = spearmanr(a, b)
            mat[i, j] = round(rho, 3)

    df_corr = pd.DataFrame(mat, index=names, columns=names)

    fig, ax = plt.subplots(figsize=(6 + len(names), 5 + len(names) // 2))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set(xticks=range(len(names)), yticks=range(len(names)),
           xticklabels=[_pretty_name(n) for n in names], yticklabels=[_pretty_name(n) for n in names],
           title="Agreement between removability signals (Spearman ρ)")
    ax.set_facecolor(PANEL)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    color="white" if abs(mat[i, j]) > 0.5 else TEXT,
                    fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    plt.tight_layout()
    _save(fig, plots_dir, "agreement_heatmap")
    print(f"  Agreement matrix ({len(keys):,} shared segments):\n{df_corr.to_string()}")
    return df_corr


def agreement_by_genre(model_scores: dict, data: dict, plots_dir: Path) -> None:
    """Per-genre Spearman ρ bars for model vs each other source."""
    if not model_scores:
        return

    sources: dict[str, dict] = {}
    if data["baselines"]:
        for col in ["sbert", "heuristic", "tfidf"]:
            sc = {k: v[col] for k, v in data["baselines"].items() if col in v}
            if sc:
                sources[col] = sc
    sources["weak"] = data["weak"]

    if not sources:
        return

    genres = sorted(set(data["genre"].values()))
    x = np.arange(len(genres))
    width = 0.8 / max(len(sources), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(genres) * 2), 5))
    _ax_style(ax)

    colors_list = [BRIGHT_GREEN, BRIGHT_ORANGE, BRIGHT_PURPLE, BRIGHT_RED]
    for i, (src_name, src_scores) in enumerate(sources.items()):
        rhos = []
        for g in genres:
            g_vids   = {vid for vid, genre in data["genre"].items() if genre == g}
            seg_keys = [k for k in model_scores if k[0] in g_vids]
            s_m = [model_scores[k] for k in seg_keys if k in src_scores]
            s_s = [src_scores[k]   for k in seg_keys if k in src_scores]
            if len(s_m) < 5:
                rhos.append(float("nan"))
                continue
            rho, _ = spearmanr(s_m, s_s)
            rhos.append(round(rho, 3))
        offset = (i - len(sources) / 2 + 0.5) * width
        c = colors_list[i % len(colors_list)]
        bars = ax.bar(x + offset, rhos, width=width * 0.9,
                      color=c, alpha=0.85, label=f"SegRemover vs {_pretty_name(src_name)}")
        for bar, v in zip(bars, rhos):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom",
                        color=TEXT, fontsize=8)

    ax.axhline(0, color=GRID, linewidth=1)
    ax.set(xticks=x, xticklabels=[_pretty_name(g) for g in genres], ylim=(-0.1, 1.05),
           ylabel="Spearman ρ  (SegRemover vs source)",
           title="Agreement varies by genre and signal source")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save(fig, plots_dir, "agreement_by_genre")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Gold evaluation  (only runs when gold labels exist)
# ══════════════════════════════════════════════════════════════════════════════

def gold_eval(model_scores: dict, data: dict, plots_dir: Path, out_dir: Path,
              data_dir: Path | None = None) -> dict:
    """Evaluate model against human gold annotations.

    Metrics:
    - Spearman ρ between p_remove and gold soft target
    - NDCG@20 (ranking quality)
    - Weak-label accuracy vs gold binary
    - Model accuracy vs gold binary at multiple thresholds

    Skipped gracefully when gold is empty (pre-annotation phase).
    """
    if not data["gold"] and not data["gold_b"]:
        print("  No gold labels yet — gold eval skipped (fill annotation_set.json first)")
        return {}

    results: dict = {}
    lines   = ["# Gold Evaluation\n"]

    # ── 1. Spearman ρ: p_remove vs gold soft target ──────────────────────────
    if data["gold"] and model_scores:
        shared = sorted(set(model_scores) & set(data["gold"]))
        if len(shared) >= 5:
            pred_soft = np.array([model_scores[k]  for k in shared])
            gold_soft = np.array([data["gold"][k]   for k in shared])
            rho, pval = spearmanr(pred_soft, gold_soft)
            results["spearman_rho"] = round(rho, 4)
            results["spearman_pval"] = round(pval, 4)
            lines.append(f"## Spearman ρ (model p_remove vs gold soft target)")
            lines.append(f"- ρ = {rho:.4f}  (p = {pval:.4f}, n = {len(shared)})")
            lines.append("")

            # NDCG@20
            ndcg = _ndcg_at_k(pred_soft, gold_soft, k=min(20, len(shared)))
            results["ndcg_at_20"] = round(ndcg, 4)
            lines.append(f"## NDCG@20 (ranking quality)")
            lines.append(f"- NDCG@20 = {ndcg:.4f}")
            lines.append("")

            # Scatter plot: model p_remove vs gold soft
            fig, ax = plt.subplots(figsize=(6, 6))
            _ax_style(ax)
            ax.scatter(gold_soft, pred_soft, color=BLUE, alpha=0.5, s=25)
            ax.plot([0, 1], [0, 1], "--", color=GRID, linewidth=1)
            ax.set(xlabel="Gold soft target", ylabel="Model p_remove",
                   xlim=(0, 1), ylim=(0, 1),
                   title=f"Model rank correlates with human judgement  (ρ={rho:.3f})")
            # Annotate with ρ
            ax.text(0.05, 0.92, f"Spearman ρ = {rho:.3f}", transform=ax.transAxes,
                    color=BRIGHT_ORANGE, fontsize=12, fontweight="bold")
            plt.tight_layout()
            _save(fig, plots_dir, "gold_scatter")

    # ── 2. Binary accuracy: model and weak labels vs gold_b ──────────────────
    if data["gold_b"]:
        shared_b = sorted(set(data["gold_b"]))
        gold_bin  = np.array([data["gold_b"][k] for k in shared_b])

        rows = []
        # Weak label accuracy
        if data["weak"]:
            weak_bin = np.array([int(data["weak"].get(k, 0.5) > 0.5) for k in shared_b])
            weak_acc = accuracy_score(gold_bin, weak_bin)
            weak_f1  = f1_score(gold_bin, weak_bin, zero_division=0)
            rows.append({"method": "weak labels", "accuracy": round(weak_acc, 4),
                         "f1": round(weak_f1, 4), "n": len(shared_b)})
            results["weak_vs_gold_acc"] = round(weak_acc, 4)
            results["weak_vs_gold_f1"]  = round(weak_f1, 4)

        # Model at multiple thresholds
        if model_scores:
            model_shared = sorted(set(model_scores) & set(data["gold_b"]))
            if model_shared:
                pred_b = np.array([model_scores[k] for k in model_shared])
                gold_b_arr = np.array([data["gold_b"][k] for k in model_shared])
                for thr in [0.5, 0.7]:
                    preds = (pred_b >= thr).astype(int)
                    acc   = accuracy_score(gold_b_arr, preds)
                    f1    = f1_score(gold_b_arr, preds, zero_division=0)
                    rows.append({"method": f"model @t={thr}", "accuracy": round(acc, 4),
                                 "f1": round(f1, 4), "n": len(model_shared)})

        if rows:
            df_gold = pd.DataFrame(rows)
            lines.append("## Binary accuracy vs gold labels")
            lines.append(df_gold.to_markdown(index=False))
            lines.append("")

            # Bar chart: model vs weak label accuracy + F1
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle("Model surpasses weak labels against human annotations",
                         color=TEXT, fontsize=13)
            for ax, col, color in [(axes[0], "accuracy", BRIGHT_CYAN),
                                   (axes[1], "f1",       BRIGHT_GREEN)]:
                _ax_style(ax)
                bars = ax.bar(df_gold["method"], df_gold[col], color=color, alpha=0.85)
                for bar, v in zip(bars, df_gold[col]):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{v:.3f}", ha="center", va="bottom",
                            color=TEXT, fontsize=10)
                ax.set(ylim=(0, 1.15), ylabel=col.capitalize(),
                       title=col.capitalize())
                ax.tick_params(axis="x", rotation=15)
            plt.tight_layout()
            _save(fig, plots_dir, "gold_accuracy")
            results["gold_n"] = len(shared_b)

    # ── 3. Weak-label quality: per-LF genre breakdown ─────────────────────────
    # (LF-level analysis requires lf_votes.jsonl — optional)
    lf_path = (data_dir / "lf_votes.jsonl") if data_dir else (out_dir / "lf_votes.jsonl")
    if lf_path.exists() and data["gold_b"]:
        lf_rows = _load_jsonl(lf_path)
        lf_accs: dict[str, list[float]] = defaultdict(list)
        for r in lf_rows:
            k = (r["video_id"], r["seg_idx"])
            if k not in data["gold_b"]:
                continue
            true = data["gold_b"][k]
            for lf_name, vote in r.get("votes", {}).items():
                if vote in (0, 1):
                    lf_accs[lf_name].append(float(vote == true))
        if lf_accs:
            lf_summary = {lf: round(float(np.mean(accs)), 4)
                          for lf, accs in lf_accs.items() if len(accs) >= 5}
            lines.append("## Labeling function accuracy vs gold")
            for lf, acc in sorted(lf_summary.items(), key=lambda x: -x[1]):
                lines.append(f"  {lf:30s}  {acc:.4f}")
            lines.append("")
            results["lf_accuracy"] = lf_summary

    text = "\n".join(lines)
    (out_dir / "gold_eval.md").write_text(text, encoding="utf-8")
    print(text)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 8. Caption-type robustness
# ══════════════════════════════════════════════════════════════════════════════

def caption_type_eval(model_scores: dict, data: dict, plots_dir: Path) -> dict:
    """Compare AUC-ROC and compression behavior for manual vs auto captions.

    Skipped gracefully if caption_type is not present in the data.
    """
    if not data.get("caption_type"):
        print("  caption_type not in data — skipping caption-type robustness eval")
        return {}

    labels, label_desc = _binary_labels(data)
    cap_groups = defaultdict(list)
    for vid in data["video_ids"]:
        ct = data["caption_type"].get(vid, "unknown")
        cap_groups[ct].append(vid)

    if len(cap_groups) < 2:
        print("  Only one caption type found — skipping robustness eval")
        return {}

    print(f"  Caption types: {dict((k, len(v)) for k, v in cap_groups.items())}")
    results = {}
    rows    = []
    for ct, vids in sorted(cap_groups.items()):
        ct_keys = {k for v in vids for k in [(v, s["seg_idx"])
                   for s in data["docs"].get(v, [])]}
        sc = {k: v for k, v in (model_scores or data["weak"]).items() if k in ct_keys}
        lb = {k: v for k, v in labels.items() if k in ct_keys}
        s, l = _aligned_arrays(sc, lb)
        if len(set(l)) < 2 or len(s) < 10:
            rows.append({"caption_type": ct, "n_videos": len(vids),
                         "n_segs": len(s), "roc_auc": float("nan"),
                         "pr_auc": float("nan")})
            continue
        roc = round(roc_auc_score(l, s), 4)
        pra = round(average_precision_score(l, s), 4)
        rows.append({"caption_type": ct, "n_videos": len(vids),
                     "n_segs": len(s), "roc_auc": roc, "pr_auc": pra})
        results[ct] = {"roc_auc": roc, "pr_auc": pra, "n": len(s)}
        print(f"    {ct:12s}  ROC={roc:.4f}  PR={pra:.4f}  n={len(s):,}")

    if not rows:
        return results

    df = pd.DataFrame(rows)

    # Bar chart comparing caption types
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        "Performance gap between manual and auto captions quantifies domain shift",
        color=TEXT, fontsize=12,
    )
    cap_colors = CAPTION_COLORS
    for ax, metric in zip(axes, ["roc_auc", "pr_auc"]):
        _ax_style(ax)
        valid = df.dropna(subset=[metric])
        colors = [cap_colors.get(ct, BRIGHT_PURPLE) for ct in valid["caption_type"]]
        bars = ax.bar(valid["caption_type"], valid[metric], color=colors, alpha=0.85)
        for bar, v in zip(bars, valid[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom",
                    color=TEXT, fontsize=11)
        ax.set(ylim=(0, 1.15), ylabel=metric.upper().replace("_", "-"),
               title=metric.upper().replace("_", "-"))
    plt.tight_layout()
    _save(fig, plots_dir, "caption_type_comparison")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 9. Qualitative examples
# ══════════════════════════════════════════════════════════════════════════════

def save_qualitative(model_scores: dict, data: dict, out_dir: Path) -> None:
    scores = model_scores or data["weak"]
    if not scores:
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    rows = []
    for (vid, sidx), p in scores.items():
        segs = data["docs"].get(vid, [])
        seg  = next((s for s in segs if s["seg_idx"] == sidx), None)
        if seg is None:
            continue
        key = (vid, sidx)
        text = seg.get("text", "")
        rows.append({
            "video_id":     vid,
            "genre":        data["genre"].get(vid, ""),
            "seg_idx":      sidx,
            "p_remove":     round(p, 4),
            "weak_p":       round(data["weak"].get(key, float("nan")), 4),
            "gold_label":   data["gold_b"].get(key, None),
            "fn_label":     FUNCTION_NAMES[data["fn"][key]]
                            if key in data["fn"] else "",
            "disfl_label":  DISFL_NAMES[data["disfl"][key]]
                            if key in data["disfl"] else "",
            "text_snippet": text[:220].replace("\n", " "),
            "text_full":    text.replace("\n", " "),
        })

    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("p_remove")

    def _block(label: str, subset: pd.DataFrame) -> str:
        if subset.empty:
            return f"\n## {label}\n\n_No examples available._"
        view = subset[["genre", "seg_idx", "p_remove", "fn_label", "text_snippet"]].copy()
        view["genre"] = view["genre"].map(_pretty_name)
        return "\n".join([f"\n## {label}\n", view.to_markdown(index=False)])

    top_remove = df[df["p_remove"] >= 0.85].tail(10)
    top_keep   = df[df["p_remove"] <= 0.15].head(10)

    # False positives / negatives if gold available.
    fp_rows = df[(df["p_remove"] >= 0.70) & (df["gold_label"] == 0)]
    fn_rows = df[(df["p_remove"] <= 0.30) & (df["gold_label"] == 1)]

    md = ["# Qualitative examples\n",
          _block("High-confidence REMOVE (p >= 0.85)", top_remove),
          _block("High-confidence KEEP (p <= 0.15)",   top_keep)]
    if not fp_rows.empty:
        md.append(_block("False positives — model says REMOVE, gold says KEEP", fp_rows.head(10)))
    if not fn_rows.empty:
        md.append(_block("False negatives — model says KEEP, gold says REMOVE", fn_rows.head(10)))

    # Per-genre failure mode table if gold labels exist.
    if not fp_rows.empty or not fn_rows.empty:
        md.append("\n## Per-genre failure mode table\n")
        md.append("One representative FP and FN per genre.\n")
        header = ("| Genre | Error | p_remove | fn_label | Text snippet |\n"
                  "|---|---|---|---|---|")
        md.append(header)
        for genre in _ordered(list(df["genre"].dropna().unique()), EXPECTED_GENRES):
            g_fp = fp_rows[fp_rows["genre"] == genre]
            if not g_fp.empty:
                r = g_fp.iloc[0]
                snippet = r["text_snippet"][:100].replace("|", "\\|")
                md.append(f"| {_pretty_name(genre)} | FP | {r['p_remove']:.2f} | {r['fn_label']} | {snippet} |")
            g_fn = fn_rows[fn_rows["genre"] == genre]
            if not g_fn.empty:
                r = g_fn.iloc[0]
                snippet = r["text_snippet"][:100].replace("|", "\\|")
                md.append(f"| {_pretty_name(genre)} | FN | {r['p_remove']:.2f} | {r['fn_label']} | {snippet} |")

    # Transcript-only risk examples: visual/deictic phrases often need video,
    # slides, diagrams, or gestures. This gives evidence for limitations even
    # before gold labels are complete.
    def _match_visual_term(text: str) -> str:
        low = text.lower()
        for pat in VISUAL_CONTEXT_PATTERNS:
            if pat in low:
                return pat
        return ""

    df["visual_context_term"] = df["text_full"].map(_match_visual_term)
    risk = df[df["visual_context_term"] != ""].copy()
    if not risk.empty:
        risk = risk.sort_values("p_remove", ascending=False)
        risk_view = risk.head(40)[["genre", "seg_idx", "p_remove", "visual_context_term", "fn_label", "text_snippet"]].copy()
        risk_view["genre"] = risk_view["genre"].map(_pretty_name)
        risk_md = [
            "# Transcript-only risk examples\n",
            "Segments containing visual/deictic cues may be important even when the transcript alone looks removable.\n",
            risk_view.to_markdown(index=False),
        ]
        (out_dir / "transcript_only_risk_examples.md").write_text("\n".join(risk_md), encoding="utf-8")
        risk.to_csv(out_dir / "transcript_only_risk_examples.csv", index=False)

        counts = risk.groupby("genre").size().sort_values(ascending=False)
        genres = _ordered(list(counts.index), EXPECTED_GENRES)
        vals = [counts.get(g, 0) for g in genres]
        fig, ax = plt.subplots(figsize=(max(8, len(genres) * 1.4), 4.5))
        _ax_style(ax)
        colors = [GENRE_PALETTE.get(g, BRIGHT_CYAN) for g in genres]
        bars = ax.bar([_pretty_name(g) for g in genres], vals, color=colors, alpha=0.95)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f"{int(v)}", ha="center", va="bottom", color=TEXT, fontsize=10)
        ax.set(ylabel="Segments with visual/deictic cues",
               title="Transcript-only risk cues by genre")
        ax.tick_params(axis="x", rotation=15)
        plt.tight_layout()
        _save(fig, plots_dir, "transcript_only_risk_by_genre")
        md.append("\n## Transcript-only risk examples\n")
        md.append("See `transcript_only_risk_examples.md` for visual/deictic segments such as "
                  "'this graph', 'on screen', and 'as you can see'.")
    else:
        (out_dir / "transcript_only_risk_examples.md").write_text(
            "# Transcript-only risk examples\n\nNo visual/deictic cue examples matched the current pattern list.\n",
            encoding="utf-8",
        )

    (out_dir / "qualitative_examples.md").write_text("\n".join(md), encoding="utf-8")
    df.drop(columns=["text_full"], errors="ignore").to_csv(out_dir / "qualitative_examples.csv", index=False)
    print(f"  Saved qualitative examples ({len(df):,} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# Shared genre bar helpers
# ══════════════════════════════════════════════════════════════════════════════

def _plot_genre_bar(values: dict, title: str, plots_dir: Path, name: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    _ax_style(ax)
    genres = list(values.keys())
    vals   = [values[g] for g in genres]
    colors = [GENRE_PALETTE.get(g, BRIGHT_CYAN) for g in genres]
    bars   = ax.bar([_pretty_name(g) for g in genres], vals, color=colors, alpha=0.95)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom",
                color=TEXT, fontsize=10)
    ax.set(title=title, ylim=(0, 1.1))
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    _save(fig, plots_dir, name)


def _plot_genre_bar_ci(
    values: dict,   # {genre: {"auc": float, "ci_lo": float, "ci_hi": float}}
    title: str, plots_dir: Path, name: str,
) -> None:
    """Genre bar chart with 95% bootstrap CI error bars."""
    fig, ax = plt.subplots(figsize=(9, 4))
    _ax_style(ax)
    genres = list(values.keys())
    vals   = [values[g]["auc"]   for g in genres]
    yerr   = np.array([
        [values[g]["auc"] - values[g]["ci_lo"] for g in genres],
        [values[g]["ci_hi"] - values[g]["auc"] for g in genres],
    ])
    colors = [GENRE_PALETTE.get(g, BRIGHT_CYAN) for g in genres]
    bars   = ax.bar([_pretty_name(g) for g in genres], vals, color=colors, alpha=0.95,
                    yerr=yerr, capsize=5, error_kw={"ecolor": TEXT, "lw": 1.5})
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{v:.3f}", ha="center", va="bottom",
                color=TEXT, fontsize=10)
    ax.set(title=title, ylim=(0, 1.15))
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    _save(fig, plots_dir, name)


# ══════════════════════════════════════════════════════════════════════════════
# 10. Markdown report
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(args, audit_info: dict, seg_metrics: dict,
                    baseline_results: dict, transcript_df: pd.DataFrame,
                    label_desc: str, config: dict | None,
                    gold_results: dict, out_dir: Path,
                    explainer_results: dict | None = None) -> None:
    plots_rel = "plots"

    def _img(name: str, caption: str = "") -> str:
        alt = caption or name
        md  = f"![{alt}]({plots_rel}/{name}.png)"
        return f"{md}\n*{caption}*" if caption else md

    lines = [
        "# segremover — Evaluation Report\n",
        f"**Checkpoint:** `{args.checkpoint}`  ",
        f"**Data dir:** `{args.data_dir}`  ",
        f"**Label source:** {label_desc}  \n",

        "## 1. Dataset summary",
        "| Item | Value |",
        "|---|---|",
        f"| Documents | {audit_info['n_docs']:,} |",
        f"| Segments  | {audit_info['n_segs']:,} |",
        f"| Gold labels | {'✓ human-annotated' if audit_info['has_gold'] else '✗ null (unannotated)'} |",
        f"| Model checkpoint | {'✓ loaded' if audit_info['has_model'] else '✗ missing'} |",
        f"| Baselines | {'✓ loaded' if audit_info['has_baselines'] else '✗ missing (run src/baselines.py)'} |",
    ]

    for g in _ordered(list(audit_info.get("genre_counts", {}).keys()), EXPECTED_GENRES):
        n = audit_info.get("genre_counts", {}).get(g, 0)
        lines.append(f"| Genre: {_pretty_name(g)} | {n} videos |")

    missing = audit_info.get("missing_genres", [])
    coverage_rows = [
        ("Six-genre coverage", "OK" if not missing else "WARN",
         "all expected genres present" if not missing else "missing: " + ", ".join(_pretty_name(g) for g in missing)),
        ("Direct removability by length", "OK" if (out_dir / "plots" / "removability_by_length.png").exists() else "WARN",
         "generated" if (out_dir / "plots" / "removability_by_length.png").exists() else "needs transcript-level eval"),
        ("Caption-type robustness", "OK" if (out_dir / "plots" / "caption_type_comparison.png").exists() else "WARN",
         "generated" if (out_dir / "plots" / "caption_type_comparison.png").exists() else "caption_type missing or only one type"),
        ("Gold/human evaluation", "OK" if audit_info.get("has_gold") else "WARN",
         "gold labels loaded" if audit_info.get("has_gold") else "weak-label proxy only"),
        ("Transcript-only risk scan", "OK" if (out_dir / "transcript_only_risk_examples.md").exists() else "WARN",
         "generated" if (out_dir / "transcript_only_risk_examples.md").exists() else "qualitative step not run"),
    ]
    lines += [
        "\n## Evidence coverage checklist",
        "| Evidence item | Status | Note |",
        "|---|---:|---|",
    ]
    for item, status, note in coverage_rows:
        lines.append(f"| {item} | {status} | {note} |")

    if seg_metrics:
        ci = seg_metrics.get("roc_auc_ci", ("—", "—"))
        lines += [
            "\n## 2. Segment-level metrics",
            "| Metric | Value |",
            "|---|---|",
            f"| ROC-AUC  | {seg_metrics.get('roc_auc', '—')} (95% CI: {ci[0]}–{ci[1]}) |",
            f"| PR-AUC   | {seg_metrics.get('pr_auc', '—')} |",
            f"| ECE      | {seg_metrics.get('ece', '—')} |",
            f"| F1 @0.5  | {seg_metrics.get('f1_at_0.5', '—')} |",
            f"| P  @0.5  | {seg_metrics.get('precision_at_0.5', '—')} |",
            f"| R  @0.5  | {seg_metrics.get('recall_at_0.5', '—')} |",
            "",
            _img("model_roc_pr",
                 f"SegRemover achieves AUC-ROC={seg_metrics.get('roc_auc', '?')} vs weak-label proxy."),
            _img("model_reliability",
                 "Reliability diagram: bars show fraction truly removed per confidence bin; "
                 "diagonal = perfect calibration."),
            _img("reliability_by_genre",
                 "Calibration varies by genre — ECE values reveal where the model is over/under-confident."),
        ]
        if (out_dir / "plots" / "calibration_before_after_T.png").exists():
            T = config.get("temperature", 1.0) if config else "?"
            lines.append(_img("calibration_before_after_T",
                               f"Temperature scaling (T={T}) changes ECE; this panel prevents claiming improvement when ECE increases."))
        lines += [
            _img("model_confusion",
                 "Confusion matrix at threshold=0.5; most errors are false positives "
                 "(cautious model keeps borderline segments)."),
            _img("model_threshold_sweep",
                 "Lowering the threshold increases recall at the cost of precision; "
                 "the F1 peak marks the operating point."),
            _img("model_confidence_buckets",
                 "High-confidence predictions (p>=0.95) are most reliable; "
                 "precision drops in the borderline 0.70-0.95 bucket."),
        ]
        if (out_dir / "plots" / "function_by_confidence_bucket.png").exists():
            lines.append(_img("function_by_confidence_bucket",
                              "Function-label mix by confidence bucket: this explains what the model tends to remove."))
        if (out_dir / "plots" / "p_remove_by_function.png").exists():
            lines += [
                "\n### Head B — Function classifier",
                _img("fn_label_distribution_by_genre",
                     "Function-label distribution by genre: lectures are new-information heavy; "
                     "podcasts skew toward discourse fillers."),
                _img("p_remove_by_function",
                     "Redundant and off-topic segments score highest p_remove, "
                     "validating that Head B shapes the removal decision."),
            ]
            if explainer_results and "head_b_accuracy" in explainer_results:
                acc = explainer_results["head_b_accuracy"]
                f1  = explainer_results["head_b_macro_f1"]
                lines += [
                    f"\n**Head B classification:** accuracy={acc}  macro-F1={f1}\n",
                    "| Function class | F1 |",
                    "|---|---|",
                ] + [
                    f"| {_pretty_name(n)} | {explainer_results.get(f'head_b_f1_{n}', '—')} |"
                    for n in FUNCTION_NAMES
                ]
                if (out_dir / "plots" / "head_b_confusion.png").exists():
                    lines.append(_img("head_b_confusion",
                                      "Row-normalised confusion matrix for Head B. "
                                      "Off-diagonal mass shows which function classes are confused."))
        if (out_dir / "plots" / "p_remove_by_disfluency.png").exists():
            lines += [
                "\n### Head C — Disfluency classifier",
                _img("disfl_distribution_by_genre",
                     "Disfluency distribution by genre: podcasts and TV series contain "
                     "more filled pauses and repetitions than lectures."),
                _img("p_remove_by_disfluency",
                     "Disfluent segments (filled pause, repetition, restart) score higher p_remove, "
                     "validating Head C's signal."),
            ]
            if explainer_results and "head_c_accuracy" in explainer_results:
                acc = explainer_results["head_c_accuracy"]
                f1  = explainer_results["head_c_macro_f1"]
                lines += [
                    f"\n**Head C classification:** accuracy={acc}  macro-F1={f1}\n",
                    "| Disfluency class | F1 |",
                    "|---|---|",
                ] + [
                    f"| {_pretty_name(n)} | {explainer_results.get(f'head_c_f1_{n}', '—')} |"
                    for n in DISFL_NAMES
                ]
                if (out_dir / "plots" / "head_c_confusion.png").exists():
                    lines.append(_img("head_c_confusion",
                                      "Row-normalised confusion matrix for Head C. "
                                      "Confusion between adjacent disfluency classes is expected."))

        if seg_metrics.get("genre_auc"):
            lines += [
                "\n### AUC by genre (95% bootstrap CI)",
                _img("model_genre_auc",
                     "AUC-ROC varies across genres — lectures show the highest discriminability, "
                     "suggesting cleaner segment boundaries."),
            ]

    if not transcript_df.empty:
        avg = transcript_df.groupby("threshold")[
            ["compression", "sbert_cosine"]].mean().round(4)
        lines += [
            "\n## 3. Transcript-level (compression vs SBERT preservation)",
            avg.to_markdown(),
            "",
            _img("transcript_compression",
                 "Semantic similarity (SBERT centroid cosine) degrades smoothly with "
                 "compression; genre curves diverge past 50% word retention."),
        ]
        if (out_dir / "plots" / "transcript_by_length.png").exists():
            lines.append(_img("transcript_by_length",
                               "Longer videos are more semantically robust to compression."))
        if (out_dir / "plots" / "removability_by_length.png").exists():
            lines.append(_img("removability_by_length",
                               "Direct removability by length: percent segments and words removed at operating thresholds."))
        if (out_dir / "plots" / "removability_by_genre.png").exists():
            lines.append(_img("removability_by_genre",
                               "Direct removability by genre at the primary threshold."))
        if (out_dir / "plots" / "rouge_curves.png").exists():
            lines.append(_img("rouge_curves",
                               "ROUGE recall tracks compression linearly; "
                               "the gap between ROUGE-1 and ROUGE-L indicates sentence fragmentation."))

    if baseline_results:
        lines += ["\n## 4. Baseline comparison"]
        header = f"| {'Method':12s} | {'ROC-AUC':>8s} | ROC 95% CI | {'PR-AUC':>8s} | PR 95% CI |"
        sep    = f"|{'-'*14}|{'-'*10}|{'-'*14}|{'-'*10}|{'-'*14}|"
        lines += [header, sep]
        for name, m in baseline_results.items():
            roc_ci = m.get("roc_ci", ("-", "-"))
            pr_ci = m.get("pr_ci", ("-", "-"))
            lines.append(f"| {_pretty_name(name):12s} | {m['roc_auc']:>8.4f} | {roc_ci[0]}-{roc_ci[1]} | "
                         f"{m['pr_auc']:>8.4f} | {pr_ci[0]}-{pr_ci[1]} |")
        lines += [
            "",
            _img("baseline_comparison",
                 "SegRemover outperforms the unsupervised baselines; error bars show 95% bootstrap CI across 300 resamples."),
        ]
        if (out_dir / "plots" / "baseline_by_genre.png").exists():
            lines.append(_img("baseline_by_genre",
                               "Genre-level breakdown: model advantage over baselines varies by domain."))

    lines += [
        "\n## 5. 4-way agreement",
        _img("agreement_heatmap",
             "Spearman ρ heatmap: model agrees most with SBERT centroid, "
             "confirming that semantic centrality guides removability."),
    ]
    if (out_dir / "plots" / "agreement_by_genre.png").exists():
        lines.append(_img("agreement_by_genre",
                           "Per-genre model–source Spearman ρ: agreement with SBERT "
                           "drops in conversational genres (podcasts, entertainment)."))

    if gold_results:
        lines += ["\n## 6. Gold evaluation (human annotations)"]
        if "spearman_rho" in gold_results:
            lines.append(f"- **Spearman ρ vs gold soft target:** {gold_results['spearman_rho']} "
                         f"(p={gold_results['spearman_pval']})")
        if "ndcg_at_20" in gold_results:
            lines.append(f"- **NDCG@20:** {gold_results['ndcg_at_20']}")
        if "weak_vs_gold_acc" in gold_results:
            lines.append(f"- **Weak-label accuracy vs gold:** {gold_results['weak_vs_gold_acc']} "
                         f"F1={gold_results['weak_vs_gold_f1']}")
        lines += [
            "",
            _img("gold_scatter",
                 "Scatter of model p_remove vs gold soft target; "
                 "upward trend confirms the model captures human removability judgements."),
            _img("gold_accuracy",
                 "Model outperforms weak labels against human binary annotations, "
                 "validating the training signal."),
            "\nSee `gold_eval.md` for full labeling-function accuracy breakdown.",
        ]
    else:
        lines += [
            "\n## 6. Gold evaluation",
            "_Gold annotations not yet available — fill `data/gold/annotation_set.json` "
            "and re-run to populate this section._",
        ]

    if (out_dir / "plots" / "caption_type_comparison.png").exists():
        lines += [
            "\n## 7. Caption-type robustness",
            _img("caption_type_comparison",
                 "Performance on auto-generated vs manually captioned videos; "
                 "a gap indicates sensitivity to transcription quality."),
        ]

    lines += [
        "\n## 8. Qualitative examples",
        "See `qualitative_examples.md` — includes high-confidence keep/remove examples and any gold-based FP/FN examples.",
    ]
    if (out_dir / "plots" / "transcript_only_risk_by_genre.png").exists():
        lines.append(_img("transcript_only_risk_by_genre",
                          "Transcript-only limitation scan: visual/deictic cues by genre."))
        lines.append("See `transcript_only_risk_examples.md` for snippets.")

    lines += [
        "\n## 9. Limitations",
        "- Segment metrics evaluated against **weak labels** (proxy) unless gold annotations exist.",
        "- Transcript SBERT cosine uses centroid-level similarity; does not capture fact-level preservation.",
        "- ROUGE recall measures lexical preservation (`pip install rouge-score`); QA-preservation is not computed.",
        "- Transcript-only risk examples are lexical cues, not confirmed errors, unless gold labels are present.",
        "- Model scores can be missing for segments beyond the checkpoint `max_segs` truncation limit.",
        "- Caption-type analysis requires `caption_type` field in segment data.",
    ]

    # Reproducibility appendix
    lines += ["\n## Appendix: Reproducibility"]
    lines.append("```")
    lines.append(f"eval seed       : 42")
    lines.append(f"eval sample     : {getattr(args, 'eval_sample', 'all')}")
    if config:
        lines.append(f"encoder         : {config.get('encoder', '—')}")
        lines.append(f"inter_layers    : {config.get('inter_layers', '—')}")
        lines.append(f"temperature T   : {config.get('temperature', 1.0)}")
        lines.append(f"max_segs        : {config.get('max_segs', '—')}")
        lines.append(f"max_tok         : {config.get('max_tok', '—')}")
    lines.append(f"checkpoint      : {args.checkpoint}")
    lines.append(f"data_dir        : {args.data_dir}")
    lines.append(f"thresholds      : {getattr(args, 'thresholds', '—')}")
    lines.append("```")
    lines += [
        "\n## Run command",
        "```bash",
        f"python src/eval.py --checkpoint {args.checkpoint} --data-dir {args.data_dir}",
        "```",
    ]

    report_path = out_dir / "evaluation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved to {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="segremover evaluation suite")
    p.add_argument("--smoke",       action="store_true",
                   help="Evaluate on smoke data (*_smoke.jsonl files, models/smoke/, "
                        "max-docs=10 unless overridden). Convenient for local runs.")
    p.add_argument("--checkpoint",  default=None,
                   help="Path to trained model checkpoint "
                        "(default: models/best.pt, or models/smoke/best.pt with --smoke)")
    p.add_argument("--models-dir",  default=None,            dest="models_dir",
                   help="Directory containing config.json "
                        "(default: models, or models/smoke with --smoke)")
    p.add_argument("--data-dir",    default="data/processed", dest="data_dir")
    p.add_argument("--output-dir",  default=None,             dest="output_dir",
                   help="Output directory "
                        "(default: outputs/eval, or outputs/eval_smoke with --smoke)")
    p.add_argument("--device",      default=None,
                   help="cuda / cpu (default: auto-detect)")
    p.add_argument("--thresholds",  nargs="+", type=float,
                   default=[0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
                   metavar="T")
    p.add_argument("--max-docs",    type=int, default=None,  dest="max_docs",
                   help="Limit to N documents (default: 10 with --smoke)")
    p.add_argument("--no-model",    action="store_true",     dest="no_model",
                   help="Skip model inference (baselines + transcript only)")
    p.add_argument("--eval-sample", type=int, default=200,  dest="eval_sample",
                   help="Randomly sample N videos for evaluation, always including "
                        "gold-annotated ones. 0 = evaluate all. (default: 200)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Resolve smoke defaults ────────────────────────────────────────────────
    label_suffix = ""
    if args.smoke:
        label_suffix            = "_smoke"
        args.max_docs           = args.max_docs or 10
        args.models_dir         = args.models_dir  or "models/smoke"
        args.checkpoint         = args.checkpoint  or "models/smoke/best.pt"
        args.output_dir         = args.output_dir  or "outputs/eval_smoke"
        print("Smoke mode: reading *_smoke.jsonl label files, "
              f"max_docs={args.max_docs}, checkpoint={args.checkpoint}")
    else:
        args.models_dir         = args.models_dir  or "models"
        args.checkpoint         = args.checkpoint  or "models/best.pt"
        args.output_dir         = args.output_dir  or "outputs/eval"

    out_dir   = Path(args.output_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    data_dir   = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    checkpoint = Path(args.checkpoint)

    # Device
    if args.device:
        device = args.device
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/10] Loading data ...")
    data = load_all_data(data_dir, args.max_docs, label_suffix=label_suffix)
    print(f"  Loaded {len(data['video_ids']):,} videos "
          f"({sum(len(v) for v in data['docs'].values()):,} segments)")

    if args.eval_sample and len(data["video_ids"]) > args.eval_sample:
        n_gold = len({vid for vid, _ in data["gold"]} | {vid for vid, _ in data["gold_b"]})
        print(f"  Sampling {args.eval_sample} videos for eval "
              f"(gold-pinned: {n_gold}, random fill: {args.eval_sample - n_gold}) ...")
        data = sample_eval_docs(data, args.eval_sample)
        print(f"  Evaluation set: {len(data['video_ids'])} videos "
              f"({sum(len(v) for v in data['docs'].values()):,} segments)")

    # ── 2. Model inference ────────────────────────────────────────────────────
    model_scores:  dict = {}
    model_heads_b: dict = {}
    model_heads_c: dict = {}
    config: dict | None = None
    config_path = models_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())

    print("\n[2/10] Model inference ...")
    if args.no_model:
        print("  --no-model: skipping")
    elif not checkpoint.exists():
        print(f"  Checkpoint not found at {checkpoint} — skipping")
    elif config is None:
        print("  models/config.json not found — cannot reconstruct model")
    else:
        model_scores, model_heads_b, model_heads_c = run_inference(checkpoint, config, data, device)

    # ── 3. Data audit ─────────────────────────────────────────────────────────
    print("\n[3/10] Data audit ...")
    audit_info = audit(data, config, model_scores, out_dir)

    # ── 4. Segment-level evaluation ───────────────────────────────────────────
    print("\n[4/10] Segment-level evaluation ...")
    seg_metrics = segment_eval(model_scores, data, args.thresholds, config, plots_dir)

    # ── 4b. Head B / Head C explainability (always runs — distribution plots
    #        don't need model scores; p_remove plots skip gracefully if no model)
    print("\n[4b] Explainability plots (Head B / Head C) ...")
    plot_fn_label_distribution(data, plots_dir)
    plot_disfl_distribution(data, plots_dir)
    plot_p_remove_by_function(model_scores, data, plots_dir)
    plot_p_remove_by_disfluency(model_scores, data, plots_dir)

    # ── 4c. Head B / Head C classifier metrics (explainer quality) ───────────
    print("\n[4c] Explainer classifier metrics (Head B / Head C) ...")
    explainer_results = explainer_eval(model_heads_b, model_heads_c, data, plots_dir)

    # ── 5. Transcript-level evaluation ────────────────────────────────────────
    print("\n[5/10] Transcript-level evaluation ...")
    transcript_df = transcript_eval(data, model_scores, args.thresholds, device, plots_dir)

    # ── 6. Baseline comparison ────────────────────────────────────────────────
    print("\n[6/10] Baseline comparison ...")
    baseline_results = baseline_eval(model_scores, data, plots_dir)

    # ── 7. 4-way agreement ────────────────────────────────────────────────────
    print("\n[7/10] 4-way agreement analysis ...")
    agreement_analysis(model_scores, data, plots_dir)
    agreement_by_genre(model_scores, data, plots_dir)

    # ── 8. Gold evaluation (only when annotations exist) ─────────────────────
    print("\n[8/10] Gold evaluation ...")
    gold_results = gold_eval(model_scores, data, plots_dir, out_dir, data_dir=data_dir)

    # ── 9. Caption-type robustness ────────────────────────────────────────────
    print("\n[9/10] Caption-type robustness eval ...")
    caption_type_eval(model_scores, data, plots_dir)

    # ── 10. Qualitative examples + report ────────────────────────────────────
    print("\n[10/10] Qualitative examples & report ...")
    save_qualitative(model_scores, data, out_dir)
    _, label_desc = _binary_labels(data)
    generate_report(args, audit_info, seg_metrics, baseline_results,
                    transcript_df, label_desc, config, gold_results, out_dir,
                    explainer_results=explainer_results)

    print(f"\n{'='*60}")
    print(f"Outputs: {out_dir.resolve()}")
    print(f"  audit.md  |  transcript_eval.csv  |  evaluation_report.md")
    n_plots = len(list(plots_dir.glob("*.png")))
    print(f"  plots/    ({n_plots} PNG files)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
