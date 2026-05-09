"""
src/eval.py — Evaluation suite for segremover.

Sections
────────
1  Data loading & audit
2  Model inference        (GPU required; skipped with --no-model or missing checkpoint)
3  Segment-level metrics  (ROC, PR-AUC, ECE, threshold sweep, confusion matrix)
4  Transcript-level       (compression vs SBERT-preservation, by genre/length)
5  Baseline comparison    (model vs random/heuristic/tfidf/sbert)
6  4-way agreement        (model / sbert / heuristic / weak-label proxy)
7  Qualitative examples   (CSV + Markdown)
8  Markdown report

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
import textwrap
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
)
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ── Dark-theme constants ───────────────────────────────────────────────────────
BG, PANEL, GRID = "#0f1117", "#1a1f2e", "#2d3748"
TEXT            = "#e2e8f0"
BLUE, GREEN, RED, ORANGE, PURPLE = "#63b3ed", "#68d391", "#fc8181", "#f6ad55", "#b794f4"

GENRE_PALETTE = {
    "commentary":    "#4C72B0",
    "entertainment": "#937860",
    "lectures":      "#DD8452",
    "podcasts":      "#55A868",
    "ted":           "#C44E52",
    "tv_series":     "#8172B3",
}
BUCKET_COLORS  = {"short": BLUE, "medium": ORANGE, "long": GREEN}
BASELINE_COLORS = {
    "model":     BLUE,
    "sbert":     GREEN,
    "heuristic": ORANGE,
    "tfidf":     PURPLE,
    "random":    GRID,
}

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
    return ax


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_all_data(data_dir: Path, max_docs: int | None = None) -> dict:
    """Load and cross-reference all processed data files."""
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

    weak  = _key_map(_load_jsonl(data_dir / "weak_labels.jsonl"),       "p_remove")
    fn    = _key_map(_load_jsonl(data_dir / "function_labels.jsonl"),    "label")
    disfl = _key_map(_load_jsonl(data_dir / "disfluency_labels.jsonl"),  "label")

    base_cols = ["random", "heuristic", "tfidf", "sbert"]
    baselines: dict[tuple, dict] = {}
    for r in _load_jsonl(data_dir / "baselines.jsonl"):
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

    # ── Gold labels (null until humans annotate) ──────────────────────────────
    gold:  dict[tuple, float] = {}   # soft target from gold_removability ordinal
    gold_b: dict[tuple, int]  = {}   # binary remove/keep from gold_function
    gold_path = data_dir.parent / "gold" / "annotation_set.json"
    if gold_path.exists():
        removability_map = {}
        for entry in json.loads(gold_path.read_text(encoding="utf-8")):
            rmap = entry.get("removability_to_soft_target", {})
            fmap = entry.get("function_to_binary_remove", {})
            for seg in entry["segments"]:
                k = (entry["video_id"], seg["seg_idx"])
                if k not in in_scope:
                    continue
                if seg.get("gold_removability") is not None:
                    gold[k] = float(rmap.get(seg["gold_removability"], 0.5))
                if seg.get("gold_function") is not None:
                    gold_b[k] = int(fmap.get(seg["gold_function"], 0))

    return dict(
        docs=docs, video_ids=video_ids, genre=genre, length_bucket=length_bucket,
        weak=weak, fn=fn, disfl=disfl, baselines=baselines,
        gold=gold, gold_b=gold_b, summary=summary,
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

    scores: dict[tuple, float] = {}
    doc_cursor = 0
    for batch in tqdm(loader, desc="inference"):
        with torch.no_grad():
            la, _, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["doc_lengths"],
                seg_batch=32,
            )
        probs = torch.sigmoid(la / T).cpu().tolist()
        seg_ptr = 0
        for i, L in enumerate(batch["doc_lengths"]):
            vid = video_ids[doc_cursor]
            for j in range(L):
                scores[(vid, docs[vid][j]["seg_idx"])] = round(probs[seg_ptr + j], 4)
            seg_ptr   += L
            doc_cursor += 1

    print(f"  Scored {len(scores):,} segments")
    return scores


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
    for g, n in sorted(genre_counts.items()):
        lines.append(f"  {g:20s}  {n:5d} videos")
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
        genre_counts=dict(genre_counts),
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
    fig.suptitle(f"Segment-level curves  [{label_desc}]", color=TEXT, fontsize=14)

    ax = _ax_style(axes[0])
    ax.plot(fpr, tpr, color=BLUE, label=f"AUC-ROC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=GRID, linewidth=1)
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC curve")
    ax.legend()

    ax = _ax_style(axes[1])
    ax.plot(rec, prec, color=GREEN, label=f"AUC-PR = {pr_auc:.3f}")
    base = labels.mean()
    ax.axhline(base, linestyle="--", color=GRID, linewidth=1,
               label=f"baseline = {base:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="PR curve")
    ax.legend()

    plt.tight_layout()
    _save(fig, plots_dir, f"{prefix}_roc_pr")
    return {"roc_auc": round(roc_auc, 4), "pr_auc": round(pr_auc, 4)}


def plot_reliability(scores: np.ndarray, labels: np.ndarray,
                     plots_dir: Path, prefix: str = "model") -> float:
    n_bins = 10
    ece    = _compute_ece(scores, labels, n_bins)
    bins   = np.linspace(0, 1, n_bins + 1)
    mids   = (bins[:-1] + bins[1:]) / 2
    accs, confs, ns = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            accs.append(np.nan); confs.append(np.nan); ns.append(0)
        else:
            accs.append(labels[mask].mean())
            confs.append(scores[mask].mean())
            ns.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(7, 6))
    _ax_style(ax)
    ax.plot([0, 1], [0, 1], "--", color=GRID, linewidth=1, label="perfect calibration")
    valid = [i for i, a in enumerate(accs) if not np.isnan(a)]
    ax.bar([mids[i] for i in valid],
           [accs[i] for i in valid],
           width=0.08, color=BLUE, alpha=0.8, label="accuracy per bin")
    ax.plot([confs[i] for i in valid],
            [accs[i]  for i in valid],
            "o-", color=ORANGE, label="mean confidence")
    ax.set(xlabel="Mean predicted probability", ylabel="Fraction of positives",
           title=f"Reliability diagram  (ECE = {ece:.4f})", xlim=(0, 1), ylim=(0, 1))
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
    fig.suptitle("Threshold sweep", color=TEXT, fontsize=14)

    ax = _ax_style(axes[0])
    for col, color in [("precision", BLUE), ("recall", GREEN), ("f1", ORANGE)]:
        ax.plot(df["threshold"], df[col], "o-", color=color, label=col)
    ax.set(xlabel="Threshold", ylabel="Score", title="P / R / F1 vs threshold",
           xlim=(min(thresholds) - 0.02, max(thresholds) + 0.02))
    ax.legend()

    ax = _ax_style(axes[1])
    ax.plot(df["threshold"], df["pct_removed"] * 100, "o-", color=PURPLE)
    ax.set(xlabel="Threshold", ylabel="% segments removed",
           title="Compression rate vs threshold")

    plt.tight_layout()
    _save(fig, plots_dir, f"{prefix}_threshold_sweep")
    return df


def plot_confidence_buckets(scores: np.ndarray, labels: np.ndarray,
                            plots_dir: Path, prefix: str = "model") -> dict:
    """p>=0.95 definitely remove  |  0.70<=p<0.95 probably remove  |  p<0.70 keep."""
    buckets = {
        "keep (p<0.70)":             scores < 0.70,
        "prob. remove (0.70–0.95)": (scores >= 0.70) & (scores < 0.95),
        "def.  remove (p>=0.95)":   scores >= 0.95,
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
    colors = [GREEN, ORANGE, RED]
    bars = ax.bar(df["bucket"], df["precision"], color=colors, alpha=0.85)
    for bar, row in zip(bars, df.itertuples()):
        if not np.isnan(row.precision):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{row.precision:.2f}\n(n={row.n:,})",
                    ha="center", va="bottom", color=TEXT, fontsize=10)
    ax.set(ylim=(0, 1.15), ylabel="Fraction truly removed",
           title="Confidence bucket precision")
    ax.tick_params(axis="x", rotation=10)
    plt.tight_layout()
    _save(fig, plots_dir, f"{prefix}_confidence_buckets")
    return result


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

    plot_confusion(scores, labs, threshold=0.5, plots_dir=plots_dir)
    sweep_df = plot_threshold_sweep(scores, labs, thresholds, plots_dir)
    plot_confidence_buckets(scores, labs, plots_dir)

    # Per-genre AUC
    genre_aucs = {}
    for g in sorted(set(data["genre"].values())):
        g_keys = {k for k, v in data["genre"].items() if v == g}
        s, l = _aligned_arrays(
            {k: v for k, v in model_scores.items() if k[0] in g_keys},
            {k: v for k, v in labels.items()       if k[0] in g_keys},
        )
        if len(set(l)) < 2 or len(s) < 10:
            continue
        genre_aucs[g] = round(roc_auc_score(l, s), 4)

    if genre_aucs:
        _plot_genre_bar(genre_aucs, "AUC-ROC by genre", plots_dir, "model_genre_auc")
    metrics["genre_auc"] = genre_aucs

    print(f"  ROC-AUC={metrics['roc_auc']}  PR-AUC={metrics['pr_auc']}  "
          f"ECE={metrics['ece']}  F1@0.5={metrics['f1_at_0.5']}")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 4. Transcript-level evaluation
# ══════════════════════════════════════════════════════════════════════════════

def transcript_eval(data: dict, model_scores: dict, thresholds: list[float],
                    device: str, plots_dir: Path) -> pd.DataFrame:
    """Sweep thresholds; for each doc compute compression + SBERT cosine."""
    from sentence_transformers import SentenceTransformer

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

        genre  = data["genre"].get(vid, "unknown")
        bucket = data["length_bucket"].get(vid, "unknown")

        for t in thresholds:
            keep_mask = seg_scores < t
            kept_words = sum(w for w, k in zip(wcs, keep_mask) if k)
            compression = kept_words / total_words

            if keep_mask.sum() == 0:
                sbert_cos = 0.0
            else:
                centroid_kept = embs[keep_mask].mean(axis=0)
                nk = np.linalg.norm(centroid_kept)
                if nk > 0:
                    centroid_kept /= nk
                sbert_cos = float(np.dot(centroid_full, centroid_kept))

            rows.append({
                "video_id":    vid,
                "genre":       genre,
                "length_bucket": bucket,
                "threshold":   t,
                "compression": round(compression, 4),
                "sbert_cosine": round(sbert_cos, 4),
                "segs_kept":   int(keep_mask.sum()),
                "segs_total":  len(segs),
            })

    df = pd.DataFrame(rows)
    df.to_csv(plots_dir.parent / "transcript_eval.csv", index=False)
    _plot_compression_curves(df, plots_dir)
    return df


def _plot_compression_curves(df: pd.DataFrame, plots_dir: Path) -> None:
    if df.empty:
        return

    # ── Overall compression vs SBERT cosine ──────────────────────────────────
    avg = df.groupby("threshold")[["compression", "sbert_cosine"]].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Transcript-level: compression vs semantic preservation", color=TEXT)

    ax = _ax_style(axes[0])
    ax.plot(avg["compression"] * 100, avg["sbert_cosine"],
            "o-", color=BLUE, markersize=7)
    for _, row in avg.iterrows():
        ax.annotate(f"t={row['threshold']:.2f}",
                    (row["compression"] * 100, row["sbert_cosine"]),
                    textcoords="offset points", xytext=(5, 4),
                    color=TEXT, fontsize=9)
    ax.set(xlabel="Words kept (%)", ylabel="SBERT cosine (full vs reduced)",
           title="Compression–preservation trade-off")

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
               title="Compression–preservation by length bucket")
        ax.legend()
        plt.tight_layout()
        _save(fig, plots_dir, "transcript_by_length")


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
        results[name] = {
            "n":       len(s),
            "roc_auc": round(roc_auc_score(l, s), 4),
            "pr_auc":  round(average_precision_score(l, s), 4),
        }

    if not results:
        return {}

    # Print table
    print(f"\n  {'Method':12s}  {'ROC-AUC':>8s}  {'PR-AUC':>8s}  {'n':>7s}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*7}")
    for name, m in results.items():
        print(f"  {name:12s}  {m['roc_auc']:>8.4f}  {m['pr_auc']:>8.4f}  {m['n']:>7,}")

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Model vs baselines  [{label_desc}]", color=TEXT, fontsize=13)
    for ax, metric in zip(axes, ["roc_auc", "pr_auc"]):
        _ax_style(ax)
        names = list(results.keys())
        vals  = [results[n][metric] for n in names]
        colors = [BASELINE_COLORS.get(n, BLUE) for n in names]
        bars  = ax.bar(names, vals, color=colors, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom",
                    color=TEXT, fontsize=10)
        ax.set(ylim=(0, 1.1), ylabel=metric.upper().replace("_", "-"),
               title=metric.upper().replace("_", "-"))
        ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    _save(fig, plots_dir, "baseline_comparison")

    return results


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
           xticklabels=names, yticklabels=names,
           title="4-way agreement  (Spearman ρ)")
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


# ══════════════════════════════════════════════════════════════════════════════
# 7. Qualitative examples
# ══════════════════════════════════════════════════════════════════════════════

def save_qualitative(model_scores: dict, data: dict, out_dir: Path) -> None:
    scores = model_scores or data["weak"]
    if not scores:
        return

    labels, _ = _binary_labels(data)

    rows = []
    for (vid, sidx), p in scores.items():
        segs = data["docs"].get(vid, [])
        seg  = next((s for s in segs if s["seg_idx"] == sidx), None)
        if seg is None:
            continue
        key = (vid, sidx)
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
            "text_snippet": seg["text"][:140].replace("\n", " "),
        })

    df = pd.DataFrame(rows).sort_values("p_remove")

    def _block(label: str, subset: pd.DataFrame) -> str:
        lines = [f"\n## {label}\n",
                 subset[["genre", "seg_idx", "p_remove", "fn_label",
                          "text_snippet"]].to_markdown(index=False)]
        return "\n".join(lines)

    top_remove = df[df["p_remove"] >= 0.85].tail(10)
    top_keep   = df[df["p_remove"] <= 0.15].head(10)

    # False positives / negatives if gold available
    fp_rows = df[(df["p_remove"] >= 0.70) & (df["gold_label"] == 0)]
    fn_rows = df[(df["p_remove"] <= 0.30) & (df["gold_label"] == 1)]

    md = ["# Qualitative examples\n",
          _block("High-confidence REMOVE (p ≥ 0.85)", top_remove),
          _block("High-confidence KEEP (p ≤ 0.15)",   top_keep)]
    if not fp_rows.empty:
        md.append(_block("False positives — model says REMOVE, gold says KEEP", fp_rows.head(10)))
    if not fn_rows.empty:
        md.append(_block("False negatives — model says KEEP, gold says REMOVE", fn_rows.head(10)))

    (out_dir / "qualitative_examples.md").write_text("\n".join(md), encoding="utf-8")
    df.to_csv(out_dir / "qualitative_examples.csv", index=False)
    print(f"  Saved qualitative examples ({len(df):,} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# Shared genre bar helper
# ══════════════════════════════════════════════════════════════════════════════

def _plot_genre_bar(values: dict, title: str, plots_dir: Path, name: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    _ax_style(ax)
    genres = list(values.keys())
    vals   = [values[g] for g in genres]
    colors = [GENRE_PALETTE.get(g, BLUE) for g in genres]
    bars   = ax.bar(genres, vals, color=colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom",
                color=TEXT, fontsize=10)
    ax.set(title=title, ylim=(0, 1.1))
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    _save(fig, plots_dir, name)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Markdown report
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(args, audit_info: dict, seg_metrics: dict,
                    baseline_results: dict, transcript_df: pd.DataFrame,
                    label_desc: str, config: dict | None, out_dir: Path) -> None:
    plots_rel = "plots"

    def _img(name: str) -> str:
        return f"![{name}]({plots_rel}/{name}.png)"

    lines = [
        "# segremover — Evaluation Report\n",
        f"**Checkpoint:** `{args.checkpoint}`  ",
        f"**Data dir:** `{args.data_dir}`  ",
        f"**Label source:** {label_desc}  \n",

        "## 1. Dataset summary",
        f"| Item | Value |",
        f"|---|---|",
        f"| Documents | {audit_info['n_docs']:,} |",
        f"| Segments  | {audit_info['n_segs']:,} |",
        f"| Gold labels | {'✓ human-annotated' if audit_info['has_gold'] else '✗ null (unannotated)'} |",
        f"| Model checkpoint | {'✓ loaded' if audit_info['has_model'] else '✗ missing'} |",
        f"| Baselines | {'✓ loaded' if audit_info['has_baselines'] else '✗ missing (run src/baselines.py)'} |",
    ]

    for g, n in sorted(audit_info.get("genre_counts", {}).items()):
        lines.append(f"| Genre: {g} | {n} videos |")

    if seg_metrics:
        lines += [
            "\n## 2. Segment-level metrics",
            f"| Metric | Value |",
            f"|---|---|",
            f"| ROC-AUC  | {seg_metrics.get('roc_auc', '—')} |",
            f"| PR-AUC   | {seg_metrics.get('pr_auc', '—')} |",
            f"| ECE      | {seg_metrics.get('ece', '—')} |",
            f"| F1 @0.5  | {seg_metrics.get('f1_at_0.5', '—')} |",
            f"| P  @0.5  | {seg_metrics.get('precision_at_0.5', '—')} |",
            f"| R  @0.5  | {seg_metrics.get('recall_at_0.5', '—')} |",
            "",
            _img("model_roc_pr"),
            _img("model_reliability"),
            _img("model_confusion"),
            _img("model_threshold_sweep"),
            _img("model_confidence_buckets"),
        ]
        if seg_metrics.get("genre_auc"):
            lines += ["\n### AUC by genre", _img("model_genre_auc")]

    if not transcript_df.empty:
        avg = transcript_df.groupby("threshold")[
            ["compression", "sbert_cosine"]].mean().round(4)
        lines += [
            "\n## 3. Transcript-level (compression vs SBERT preservation)",
            avg.to_markdown(),
            "",
            _img("transcript_compression"),
        ]

    if baseline_results:
        lines += ["\n## 4. Baseline comparison"]
        header = f"| {'Method':12s} | {'ROC-AUC':>8s} | {'PR-AUC':>8s} |"
        sep    = f"|{'-'*14}|{'-'*10}|{'-'*10}|"
        lines += [header, sep]
        for name, m in baseline_results.items():
            lines.append(f"| {name:12s} | {m['roc_auc']:>8.4f} | {m['pr_auc']:>8.4f} |")
        lines += ["", _img("baseline_comparison")]

    lines += [
        "\n## 5. 4-way agreement",
        _img("agreement_heatmap"),
        "\n## 6. Qualitative examples",
        "See `qualitative_examples.md` and `qualitative_examples.csv`.",
        "\n## 7. Limitations",
        "- Segment metrics evaluated against **weak labels** (proxy) unless gold annotations exist.",
        "- Transcript SBERT cosine uses centroid-level similarity; does not capture fact-level preservation.",
        "- ROUGE / QA-preservation metrics not computed (requires LLM API key).",
        "- Model scores missing for segments beyond `max_segs` truncation limit.",
        "\n## 8. Run commands",
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
    p.add_argument("--checkpoint",  default="models/best.pt",
                   help="Path to trained model checkpoint")
    p.add_argument("--models-dir",  default="models",       dest="models_dir")
    p.add_argument("--data-dir",    default="data/processed", dest="data_dir")
    p.add_argument("--output-dir",  default="outputs/eval",   dest="output_dir")
    p.add_argument("--device",      default=None,
                   help="cuda / cpu (default: auto-detect)")
    p.add_argument("--thresholds",  nargs="+", type=float,
                   default=[0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
                   metavar="T")
    p.add_argument("--max-docs",    type=int, default=None,  dest="max_docs",
                   help="Limit to N documents (smoke test)")
    p.add_argument("--no-model",    action="store_true",     dest="no_model",
                   help="Skip model inference (baselines + transcript only)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

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
    print("\n[1/8] Loading data ...")
    data = load_all_data(data_dir, args.max_docs)

    # ── 2. Model inference ────────────────────────────────────────────────────
    model_scores: dict = {}
    config: dict | None = None
    config_path = models_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())

    print("\n[2/8] Model inference ...")
    if args.no_model:
        print("  --no-model: skipping")
    elif not checkpoint.exists():
        print(f"  Checkpoint not found at {checkpoint} — skipping")
    elif config is None:
        print("  models/config.json not found — cannot reconstruct model")
    else:
        model_scores = run_inference(checkpoint, config, data, device)

    # ── 3. Data audit ─────────────────────────────────────────────────────────
    print("\n[3/8] Data audit ...")
    audit_info = audit(data, config, model_scores, out_dir)

    # ── 4. Segment-level evaluation ───────────────────────────────────────────
    print("\n[4/8] Segment-level evaluation ...")
    seg_metrics = segment_eval(model_scores, data, args.thresholds, config, plots_dir)

    # ── 5. Transcript-level evaluation ────────────────────────────────────────
    print("\n[5/8] Transcript-level evaluation ...")
    transcript_df = transcript_eval(data, model_scores, args.thresholds, device, plots_dir)

    # ── 6. Baseline comparison ────────────────────────────────────────────────
    print("\n[6/8] Baseline comparison ...")
    baseline_results = baseline_eval(model_scores, data, plots_dir)

    # ── 7. 4-way agreement ────────────────────────────────────────────────────
    print("\n[7/8] 4-way agreement analysis ...")
    agreement_analysis(model_scores, data, plots_dir)

    # ── 8. Qualitative examples + report ─────────────────────────────────────
    print("\n[8/8] Qualitative examples & report ...")
    save_qualitative(model_scores, data, out_dir)
    _, label_desc = _binary_labels(data)
    generate_report(args, audit_info, seg_metrics, baseline_results,
                    transcript_df, label_desc, config, out_dir)

    print(f"\n{'='*60}")
    print(f"Outputs: {out_dir.resolve()}")
    print(f"  audit.md  |  transcript_eval.csv  |  evaluation_report.md")
    print(f"  plots/    ({len(list(plots_dir.glob('*.png')))} PNG files)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
