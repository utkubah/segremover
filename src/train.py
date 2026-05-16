"""
src/train.py — Multi-head BERTSum (segremover)

Architecture (Liu & Lapata, BERTSum, EMNLP 2019):
  Stage 1  roberta-base applied per segment → one [CLS] embedding per segment
  Stage 2  inter-segment Transformer (2–4 layers) → each segment sees the full doc
  Head B   function 6-class   CE  loss    (explainer — semantic function)
  Head C   disfluency 5-class CE  loss    (explainer — disfluency type)
  Head A   p_remove ∈ [0,1]   BCE loss    (ranker — conditioned on Head B & C logits)

Cascade: Head A input = [h ‖ logit_B ‖ logit_C], making the removal decision
explicitly dependent on the predicted function class and disfluency type.

Loss: 1.0 * BCE(A) + 1.0 * CE(B) + 1.0 * CE(C)

Defaults: roberta-base, inter_layers=4, max_segs=256, max_tok=384, seg_batch=32.
For roberta-large: pass --encoder roberta-large --seg-batch 8 --batch-size 1 --grad-accum 16

On HPC, pre-download the encoder on the login node first:
    python -c "from transformers import AutoTokenizer, RobertaModel; \
               AutoTokenizer.from_pretrained('roberta-base'); \
               RobertaModel.from_pretrained('roberta-base')"

Usage:
    python src/train.py
    python src/train.py --encoder roberta-large --seg-batch 16 --batch-size 1 --grad-accum 16
    python src/train.py --max-docs 10   # smoke test
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, RobertaModel, get_linear_schedule_with_warmup
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
SEGMENTS_PATH    = Path("data/processed/segments_topic.jsonl")
WEAK_LABELS_PATH = Path("data/processed/weak_labels.jsonl")
FUNCTION_PATH    = Path("data/processed/function_labels.jsonl")
DISFLUENCY_PATH  = Path("data/processed/disfluency_labels.jsonl")
MODELS_DIR       = Path("models")

N_FUNCTION   = 6
N_DISFLUENCY = 5
MAX_POS      = 512  # positional embedding table size for inter-segment stage


# ── Data loading ──────────────────────────────────────────────────────────────

def load_docs(max_docs=None):
    """Return list of documents, each with its segments and all three label types.

    Each document: {video_id, genre, segments: [{text, p_remove, fn_label, disfl_label}]}
    Segments without a weak label are skipped; missing function/disfluency labels
    fall back to class 0 (new_information / clean).
    """
    weak = {}
    with WEAK_LABELS_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            weak[(r["video_id"], r["seg_idx"])] = r["p_remove"]

    fn = {}
    if FUNCTION_PATH.exists():
        with FUNCTION_PATH.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                fn[(r["video_id"], r["seg_idx"])] = r["label"]

    disfl = {}
    if DISFLUENCY_PATH.exists():
        with DISFLUENCY_PATH.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                disfl[(r["video_id"], r["seg_idx"])] = r["label"]

    docs_raw: dict[str, dict] = defaultdict(lambda: {"genre": None, "segments": []})
    with SEGMENTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            seg = json.loads(line)
            key = (seg["video_id"], seg["seg_idx"])
            if key not in weak:
                continue
            vid = seg["video_id"]
            docs_raw[vid]["genre"] = seg["genre"]
            docs_raw[vid]["segments"].append({
                "seg_idx":    seg["seg_idx"],
                "text":       seg["text"],
                "p_remove":   weak[key],
                "fn_label":   fn.get(key, 0),
                "disfl_label": disfl.get(key, 0),
            })

    docs = []
    for vid, d in docs_raw.items():
        if d["segments"]:
            # Keep segments in order
            d["segments"].sort(key=lambda s: s["seg_idx"])
            docs.append({"video_id": vid, "genre": d["genre"], "segments": d["segments"]})

    docs.sort(key=lambda x: x["video_id"])
    random.shuffle(docs)

    if max_docs:
        docs = docs[:max_docs]

    total_segs = sum(len(d["segments"]) for d in docs)
    print(f"  {len(docs)} documents, {total_segs:,} segments")
    return docs


def train_dev_split(docs, dev_frac=0.2):
    """Genre-stratified split by video_id."""
    by_genre: dict[str, list] = defaultdict(list)
    for d in docs:
        by_genre[d["genre"]].append(d)

    train_docs, dev_docs = [], []
    for lst in by_genre.values():
        if len(lst) < 2:
            # Only one doc for this genre — put it in training, skip dev
            train_docs.extend(lst)
            continue
        n_dev = max(1, int(len(lst) * dev_frac))
        dev_docs.extend(lst[:n_dev])
        train_docs.extend(lst[n_dev:])
    return train_docs, dev_docs


# ── Dataset ───────────────────────────────────────────────────────────────────

class DocDataset(Dataset):
    """One item = one document (list of segments, tokenized on the fly)."""

    def __init__(self, docs, tokenizer, max_tok: int, max_segs: int):
        self.docs     = docs
        self.tok      = tokenizer
        self.max_tok  = max_tok
        self.max_segs = max_segs

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc  = self.docs[idx]
        segs = doc["segments"][:self.max_segs]
        texts = [s["text"] for s in segs]

        enc = self.tok(
            texts,
            max_length=self.max_tok,
            truncation=True,
            padding=True,          # pad to longest segment in this doc
            return_tensors="pt",
        )
        return {
            "input_ids":     enc["input_ids"],        # (n_segs, tok)
            "attention_mask": enc["attention_mask"],   # (n_segs, tok)
            "p_remove":   torch.tensor([s["p_remove"]   for s in segs], dtype=torch.float),
            "fn_label":   torch.tensor([s["fn_label"]   for s in segs], dtype=torch.long),
            "disfl_label": torch.tensor([s["disfl_label"] for s in segs], dtype=torch.long),
        }


def collate_fn(batch):
    """Flatten all segments from all docs, pad to the batch-level max token length."""
    doc_lengths = [item["input_ids"].size(0) for item in batch]
    max_tok     = max(item["input_ids"].size(1) for item in batch)

    all_ids, all_mask = [], []
    all_p, all_fn, all_disfl = [], [], []

    for item in batch:
        n, t = item["input_ids"].shape
        if t < max_tok:
            pad = torch.zeros(n, max_tok - t, dtype=torch.long)
            ids  = torch.cat([item["input_ids"],     pad], dim=1)
            mask = torch.cat([item["attention_mask"], pad], dim=1)
        else:
            ids, mask = item["input_ids"], item["attention_mask"]
        all_ids.append(ids)
        all_mask.append(mask)
        all_p.append(item["p_remove"])
        all_fn.append(item["fn_label"])
        all_disfl.append(item["disfl_label"])

    return {
        "input_ids":     torch.cat(all_ids,   dim=0),   # (total_segs, max_tok)
        "attention_mask": torch.cat(all_mask,  dim=0),
        "p_remove":   torch.cat(all_p,    dim=0),   # (total_segs,)
        "fn_label":   torch.cat(all_fn,   dim=0),
        "disfl_label": torch.cat(all_disfl, dim=0),
        "doc_lengths": doc_lengths,                      # list[int]
    }


# ── Model ─────────────────────────────────────────────────────────────────────

class SegRemover(nn.Module):
    """
    BERTSum-style hierarchical model.

    Stage 1 — roberta-base encodes each segment independently; the [CLS] token
              embedding summarises the segment. Long documents are encoded in
              sub-batches of `seg_batch` to control peak GPU memory.

    Stage 2 — A standard Transformer encoder runs over the sequence of [CLS]
              embeddings for a document. All segments now attend to each other,
              giving document-level context (catches cross-segment repetition).

    Heads   — Head B (function) and Head C (disfluency) are linear classifiers.
              Head A (p_remove) is a 2-layer MLP conditioned on [h ‖ logit_B ‖ logit_C].
    """

    def __init__(self, encoder_name: str = "roberta-base",
                 n_inter_layers: int = 2, dropout: float = 0.1,
                 grad_checkpoint: bool = False):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(encoder_name)
        if grad_checkpoint:
            # Recompute activations on backward instead of storing them.
            # Cuts Stage-1 activation memory by ~10× at the cost of ~25% slower backward.
            self.encoder.gradient_checkpointing_enable()
        d = self.encoder.config.hidden_size  # 768 (base) or 1024 (large)

        self.pos_emb = nn.Embedding(MAX_POS, d)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=8, dim_feedforward=d * 4,
            dropout=dropout, batch_first=True,
        )
        self.inter_tf = nn.TransformerEncoder(enc_layer, num_layers=n_inter_layers)

        self.head_b = nn.Linear(d, N_FUNCTION)   # function 6-class (explainer)
        self.head_c = nn.Linear(d, N_DISFLUENCY) # disfluency 5-class (explainer)
        # Head A is conditioned on Head B and C logits — the "ranker" is explicitly
        # driven by the model's own explanations of function and disfluency type.
        self.head_a = nn.Sequential(
            nn.Linear(d + N_FUNCTION + N_DISFLUENCY, d // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d // 4, 1),
        )
        self.drop   = nn.Dropout(dropout)

    def _encode_segments(self, input_ids, attention_mask, seg_batch: int):
        """Run roberta in sub-batches → (total_segs, d). Keeps full gradient."""
        parts = []
        for s in range(0, input_ids.size(0), seg_batch):
            out = self.encoder(
                input_ids=input_ids[s:s + seg_batch],
                attention_mask=attention_mask[s:s + seg_batch],
            )
            parts.append(out.last_hidden_state[:, 0])  # [CLS]
        return torch.cat(parts, dim=0)

    def forward(self, input_ids, attention_mask, doc_lengths, seg_batch: int = 32,
                ablate_cascade: bool = False):
        # Stage 1 — one [CLS] embedding per segment
        cls_embs = self._encode_segments(input_ids, attention_mask, seg_batch)  # (N, d)

        # Stage 2 — pack into (B, max_L, d), apply inter-segment transformer
        B      = len(doc_lengths)
        max_L  = max(doc_lengths)
        d      = cls_embs.size(-1)

        padded   = cls_embs.new_zeros(B, max_L, d)
        key_mask = torch.ones(B, max_L, dtype=torch.bool, device=cls_embs.device)

        offset = 0
        for i, L in enumerate(doc_lengths):
            padded[i, :L]   = cls_embs[offset:offset + L]
            key_mask[i, :L] = False   # False = real token, True = padding
            offset += L

        pos    = torch.arange(max_L, device=cls_embs.device).unsqueeze(0)
        padded = self.drop(padded + self.pos_emb(pos))
        inter  = self.inter_tf(padded, src_key_padding_mask=key_mask)  # (B, max_L, d)

        # Unpack back to (total_segs, d)
        segs_ctx = torch.cat([inter[i, :L] for i, L in enumerate(doc_lengths)], dim=0)
        segs_ctx = self.drop(segs_ctx)

        # Explainer heads run first; ranker head is conditioned on their outputs
        logit_b = self.head_b(segs_ctx)                              # (N, 6)
        logit_c = self.head_c(segs_ctx)                              # (N, 5)
        b_in = torch.zeros_like(logit_b) if ablate_cascade else logit_b
        c_in = torch.zeros_like(logit_c) if ablate_cascade else logit_c
        combined = torch.cat([segs_ctx, b_in, c_in], dim=-1)         # (N, d+11)
        logit_a  = self.head_a(combined).squeeze(-1)                 # (N,)
        return logit_a, logit_b, logit_c


# ── Loss & utilities ──────────────────────────────────────────────────────────

def total_loss(logit_a, logit_b, logit_c, p_remove, fn_label, disfl_label,
               w_b, w_c, weight_b, weight_c):
    la = F.binary_cross_entropy_with_logits(logit_a, p_remove)
    lb = F.cross_entropy(logit_b, fn_label,    weight=weight_b)
    lc = F.cross_entropy(logit_c, disfl_label, weight=weight_c)
    return la + w_b * lb + w_c * lc, la.item(), lb.item(), lc.item()


def class_weights(labels: torch.Tensor, n: int, device) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=n).float().clamp(min=1)
    return (counts.sum() / (n * counts)).to(device)


def compute_ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width bins, 10-bin default)."""
    bins = torch.linspace(0.0, 1.0, n_bins + 1)
    ece  = torch.tensor(0.0)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = targets[mask].float().mean()
        conf = probs[mask].mean()
        ece += mask.float().sum() * (acc - conf).abs()
    return (ece / len(probs)).item()


def dev_auc(model, loader, device, seg_batch):
    """AUC-ROC of Head A on the dev set (binary targets: p_remove > 0.5)."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    scores, targets = [], []
    with torch.no_grad():
        for batch in loader:
            la, _, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["doc_lengths"],
                seg_batch,
            )
            scores.extend(torch.sigmoid(la).cpu().tolist())
            targets.extend((batch["p_remove"] > 0.5).long().tolist())

    try:
        return roc_auc_score(targets, scores)
    except ValueError:
        return 0.5   # fallback: single class in dev set (only for tiny smoke tests)


# ── Calibration ───────────────────────────────────────────────────────────────

def fit_temperature(model, loader, device, seg_batch) -> float:
    """Grid-search temperature T ∈ (0.1, 5.0] that minimises Head-A NLL on dev."""
    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for batch in loader:
            la, _, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["doc_lengths"],
                seg_batch,
            )
            logits.extend(la.cpu().tolist())
            targets.extend(batch["p_remove"].tolist())

    logits  = torch.tensor(logits)
    targets = torch.tensor(targets)

    best_T, best_nll = 1.0, float("inf")
    for T in [t / 10 for t in range(1, 51)]:   # 0.1, 0.2, … 5.0
        nll = F.binary_cross_entropy_with_logits(logits / T, targets).item()
        if nll < best_nll:
            best_nll, best_T = nll, T
    return best_T


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train segremover 3-head model")
    parser.add_argument("--encoder",      default="roberta-base",
                        help="HuggingFace encoder (roberta-base or roberta-large)")
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--batch-size",   type=int,   default=2,   dest="batch_size",
                        help="Documents per gradient step")
    parser.add_argument("--seg-batch",    type=int,   default=32,  dest="seg_batch",
                        help="Segments per roberta forward pass (memory control)")
    parser.add_argument("--lr",           type=float, default=2e-5)
    parser.add_argument("--inter-layers", type=int,   default=4,   dest="inter_layers")
    parser.add_argument("--max-segs",     type=int,   default=256, dest="max_segs",
                        help="Max segments per document (truncates longer docs)")
    parser.add_argument("--max-tok",      type=int,   default=384, dest="max_tok")
    parser.add_argument("--grad-checkpoint", action="store_true", default=True,
                        dest="grad_checkpoint",
                        help="Enable gradient checkpointing in the roberta encoder "
                             "(saves ~10x activation memory, ~25%% slower backward; "
                             "use --no-grad-checkpoint to disable)")
    parser.add_argument("--w-b",          type=float, default=1.0,
                        help="Function head loss weight")
    parser.add_argument("--w-c",          type=float, default=1.0,
                        help="Disfluency head loss weight")
    parser.add_argument("--dev-frac",     type=float, default=0.2, dest="dev_frac")
    parser.add_argument("--dropout",      type=float, default=0.1)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--grad-accum",   type=int,   default=8,   dest="grad_accum",
                        help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    parser.add_argument("--max-docs",     type=int,   default=None, dest="max_docs",
                        help="Limit to N documents (smoke test)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    models_dir = MODELS_DIR / "smoke" if args.max_docs else MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading data...")
    docs = load_docs(args.max_docs)
    train_docs, dev_docs = train_dev_split(docs, args.dev_frac)
    print(f"  train={len(train_docs)} docs  dev={len(dev_docs)} docs")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    train_ds  = DocDataset(train_docs, tokenizer, args.max_tok, args.max_segs)
    dev_ds    = DocDataset(dev_docs,   tokenizer, args.max_tok, args.max_segs)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Class weights from training set
    all_fn    = torch.tensor([s["fn_label"]    for d in train_docs for s in d["segments"]])
    all_disfl = torch.tensor([s["disfl_label"] for d in train_docs for s in d["segments"]])
    weight_b  = class_weights(all_fn,    N_FUNCTION,   device)
    weight_c  = class_weights(all_disfl, N_DISFLUENCY, device)
    print(f"  function weights:   {weight_b.cpu().tolist()}")
    print(f"  disfluency weights: {weight_c.cpu().tolist()}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Loading encoder ({args.encoder}) ...")
    model = SegRemover(args.encoder, args.inter_layers, args.dropout,
                       grad_checkpoint=args.grad_checkpoint).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {n_params:.0f}M parameters")

    # optimizer steps (not forward passes) — accounts for gradient accumulation
    n_steps   = (len(train_loader) // args.grad_accum + 1) * args.epochs
    warmup    = n_steps // 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, n_steps)

    eff_batch = args.batch_size * args.grad_accum
    print(f"\nTraining: {args.epochs} epochs, {n_steps} optimizer steps")
    print(f"Batch: {args.batch_size} docs x {args.grad_accum} accum = {eff_batch} effective")
    print(f"max_segs={args.max_segs}  max_tok={args.max_tok}  "
          f"grad_checkpoint={args.grad_checkpoint}")
    print(f"Warmup: {warmup} steps  |  Loss weights: A=1.0  B={args.w_b}  C={args.w_c}\n")

    best_auc  = 0.0
    best_path = models_dir / "best.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_l = total_la = total_lb = total_lc = 0.0
        n_batches = 0

        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            p_r  = batch["p_remove"].to(device)
            fn   = batch["fn_label"].to(device)
            ds   = batch["disfl_label"].to(device)

            la, lb, lc = model(ids, mask, batch["doc_lengths"], args.seg_batch)
            loss, la_v, lb_v, lc_v = total_loss(
                la, lb, lc, p_r, fn, ds,
                args.w_b, args.w_c, weight_b, weight_c,
            )
            # scale loss before accumulation so gradient magnitude is consistent
            (loss / args.grad_accum).backward()

            total_l  += loss.item()
            total_la += la_v
            total_lb += lb_v
            total_lc += lc_v
            n_batches += 1

            last_batch = (step + 1 == len(train_loader))
            if (step + 1) % args.grad_accum == 0 or last_batch:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        train_loss = total_l / max(n_batches, 1)
        auc = dev_auc(model, dev_loader, device, args.seg_batch)

        nb = max(n_batches, 1)
        print(
            f"Epoch {epoch}  "
            f"train_loss={train_loss:.4f} "
            f"(A={total_la/nb:.4f} "
            f"B={total_lb/nb:.4f} "
            f"C={total_lc/nb:.4f})  "
            f"dev_auc={auc:.4f}"
        )

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best_path)
            print(f"  [saved] Checkpoint (AUC={best_auc:.4f})")

    # ── Temperature calibration + ECE ─────────────────────────────────────────
    print("\nCalibrating temperature on dev set ...")
    if not best_path.exists():
        print(f"  No checkpoint saved (best_auc never improved from 0.0) — skipping calibration")
        return
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    T = fit_temperature(model, dev_loader, device, args.seg_batch)
    print(f"  Temperature T = {T:.2f}")

    # ECE at calibrated temperature (binary target: p_remove > 0.5)
    model.eval()
    cal_logits, cal_targets = [], []
    with torch.no_grad():
        for batch in dev_loader:
            la, _, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["doc_lengths"],
                args.seg_batch,
            )
            cal_logits.extend(la.cpu().tolist())
            cal_targets.extend((batch["p_remove"] > 0.5).long().tolist())
    probs_cal = torch.sigmoid(torch.tensor(cal_logits) / T)
    ece_val   = compute_ece(probs_cal, torch.tensor(cal_targets))
    print(f"  ECE (calibrated)  = {ece_val:.4f}")

    # ── Save config ───────────────────────────────────────────────────────────
    config = {
        "encoder":      args.encoder,
        "inter_layers": args.inter_layers,
        "dropout":      args.dropout,
        "max_segs":     args.max_segs,
        "max_tok":      args.max_tok,
        "seg_batch":    args.seg_batch,
        "temperature":  T,
        "best_dev_auc": round(best_auc, 4),
        "dev_ece":      round(ece_val, 4),
    }
    (models_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"\nSaved {best_path} and {models_dir / 'config.json'}")
    print(f"Best dev AUC: {best_auc:.4f}  |  Temperature: {T:.2f}  |  ECE: {ece_val:.4f}")


if __name__ == "__main__":
    main()
