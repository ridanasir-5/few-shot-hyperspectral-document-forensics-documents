"""
===========================================================================
  HYPERSPECTRAL DOCUMENT FORENSICS — PIPELINE v4
===========================================================================
  DESIGN
  ──────
  iVision HHID backbone is ALWAYS loaded from the saved checkpoint
  (./best_protonet_crosssplit.pt).  No retraining of iVision ever happens.

  Two separate evaluation pipelines:

  ┌─────────────────────────────────────────────────────────────────┐
  │  PIPELINE A — iVision HHID                                      │
  │  Backbone : loaded from ./best_protonet_crosssplit.pt (frozen)  │
  │  Tasks    : Writer ID · Ink Mismatch · Verification             │
  │             Forgery Detection · Age & Gender                     │
  │  Outputs  : ./results_ivisio/                                    │
  └─────────────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────────────┐
  │  PIPELINE B — UWA WIHSI V1.0                                    │
  │  Backbone : NEW ProtoNet trained ONLY on UWA WIHSI data         │
  │             Warm-started from iVision weights (transfer init)   │
  │             Saved to ./best_protonet_uwa.pt                     │
  │  Tasks    : Writer ID · Ink Mismatch · Verification             │
  │             Forgery Detection  (Age/Gender skipped — no labels) │
  │  Outputs  : ./results_uwa/                                       │
  └─────────────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────────────┐
  │  CROSS-DATASET (always runs if both datasets available)         │
  │  iVision backbone → zero-shot classify UWA pages               │
  │  UWA backbone     → zero-shot classify iVision pages           │
  └─────────────────────────────────────────────────────────────────┘

  Usage
  ─────
  # Run both pipelines (iVision from ckpt, UWA trained fresh or from ckpt):
      python hyperspectral_forensics_pipeline_v4.py

  # Run only iVision tasks:
      python hyperspectral_forensics_pipeline_v4.py --dataset ivisio

  # Run only UWA tasks:
      python hyperspectral_forensics_pipeline_v4.py --dataset uwa

  NOTE: iVision HHID backbone is ALWAYS loaded from checkpoint.
        There is NO option to retrain it — this is intentional.
        UWA backbone is auto-detected: if ./best_protonet_uwa.pt
        exists it is loaded; otherwise it is trained from scratch
        using iVision weights as transfer initialisation.

  Required files
  ──────────────
  ./best_protonet_crosssplit.pt     ← iVision checkpoint (MUST exist)
  ./data_preprocessed/preprocess_index.csv
  ./uwa_preprocessed/preprocess_index.csv
===========================================================================
"""

import os, re, math, time, random, argparse, warnings, collections, json, datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from torchvision import models
    from torchvision.models import ResNet18_Weights
except ImportError:
    raise ImportError("pip install torchvision")

try:
    from sklearn.metrics import (
        confusion_matrix, classification_report,
        roc_curve, auc, precision_recall_curve,
        average_precision_score, matthews_corrcoef, f1_score,
    )
    from sklearn.preprocessing import label_binarize
except ImportError:
    raise ImportError("pip install scikit-learn")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

warnings.filterwarnings("ignore")

# ============================================================
#  DISPLAY NAMES  (tag → human-readable label for titles/prints)
# ============================================================
TAG_DISPLAY = {
    "ivisio_hhid": "iVision HHID",
    "uwa":         "UWA WIHSI",
}
def dtag(tag: str) -> str:
    """Return display name for a dataset tag."""
    return TAG_DISPLAY.get(tag, tag)




# ============================================================
#  CONFIG
# ============================================================
@dataclass
class CFG:
    # ── Paths ────────────────────────────────────────────────
    ivisio_hhid_index:   str = "./data_preprocessed/preprocess_index.csv"
    ivisio_hhid_dir:     str = "./data_preprocessed"
    uwa_index:      str = "./uwa_preprocessed/preprocess_index.csv"
    uwa_dir:        str = "./uwa_preprocessed"

    # iVision checkpoint — ALWAYS loaded, never retrained
    ivisio_hhid_ckpt:    str = "./best_protonet_crosssplit.pt"
    # UWA checkpoint — trained fresh on UWA data, or auto-loaded if it already exists
    uwa_ckpt:       str = "./best_protonet_uwa.pt"

    ivisio_hhid_out:     str = "./results_ivisio_hhid"
    uwa_out:        str = "./results_uwa"

    # ── Device ───────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed:   int = 42

    # ── Embed dim (must match iVision checkpoint) ────────────
    embed_dim:  int = 256

    # ── ProtoNet — iVision proven values (Exp 5) ────────────
    # (used only for UWA training; iVision is never retrained)
    n_way:                    int   = 5
    k_shot:                   int   = 1
    train_q_query:            int   = 2
    eval_q_query:             int   = 1
    test_episodes:            int   = 200
    max_cache:                int   = 16
    fp16_cache:               bool  = True

    # ── UWA ProtoNet training ────────────────────────────────
    # UWA has only 7 writers → use 3-way episodes
    uwa_n_way:                int   = 3
    uwa_epochs:               int   = 60    # more epochs: small dataset
    uwa_episodes_per_epoch:   int   = 100
    uwa_lr:                   float = 5e-5  # lower LR: transferring weights
    uwa_wd:                   float = 1e-4

    # ── Downstream fine-tune ─────────────────────────────────
    patch_size:  int   = 128
    k_patches:   int   = 24
    min_ink:     float = 0.12
    cls_epochs:  int   = 40
    cls_batch:   int   = 8
    cls_lr:      float = 3e-4
    cls_wd:      float = 1e-4
    patience:    int   = 10
    amp:         bool  = True
    verif_pairs: int   = 1000


cfg = CFG()


# ============================================================
#  UTILS
# ============================================================
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [plot] {path}")

def compute_ci95(accs):
    if len(accs) < 2: return 0.0
    return 1.96 * float(np.std(accs, ddof=1)) / math.sqrt(len(accs))

def pairwise_sq_dists(a, b):
    a2 = (a*a).sum(1, keepdim=True)
    b2 = (b*b).sum(1, keepdim=True).t()
    return (a2 + b2 - 2*a@b.t()).clamp(min=0)

def prototypical_logits(z_s, y_s, z_q, n_way):
    protos = torch.stack([z_s[y_s==c].mean(0) for c in range(n_way)])
    return -pairwise_sq_dists(z_q, protos)

def parse_writer_id(name):
    m = re.match(r"w(\d+)", str(name))
    return int(m.group(1)) if m else -1

def parse_page_id(name):
    m = re.search(r"p(\d+)", str(name))
    return int(m.group(1)) if m else -1


# ============================================================
#  LRU CUBE CACHE
# ============================================================
def _load_cube_lazy(path: str, keep_fp16: bool = True) -> torch.Tensor:
    arr = np.load(path, mmap_mode="r")
    x   = np.asarray(arr, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected 3-D cube, got {x.shape}")
    if x.shape[-1] <= 512 and x.shape[0] > 16 and x.shape[1] > 16:
        x = np.transpose(x, (2, 0, 1))   # HWC → CHW
    mu  = float(x.mean()); sig = float(x.std()) + 1e-6
    x   = (x - mu) / sig
    t   = torch.from_numpy(np.ascontiguousarray(x))
    return t.half() if keep_fp16 else t


class CubeCache:
    def __init__(self, maxsize=16, fp16=True):
        self._d = collections.OrderedDict()
        self.maxsize = maxsize; self.fp16 = fp16

    def get(self, path: str) -> torch.Tensor:
        if path in self._d:
            self._d.move_to_end(path); return self._d[path]
        t = _load_cube_lazy(path, self.fp16)
        self._d[path] = t
        if len(self._d) > self.maxsize:
            self._d.popitem(last=False)
        return t


_CACHE = CubeCache(maxsize=cfg.max_cache, fp16=cfg.fp16_cache)


# ============================================================
#  DATASET LOADING
# ============================================================
def load_index(index_csv: str, prep_dir: str, tag: str) -> pd.DataFrame:
    if not os.path.exists(index_csv):
        raise FileNotFoundError(f"Index not found: {index_csv}")
    df = pd.read_csv(index_csv)
    def resolve(p):
        p = str(p)
        return p if os.path.isabs(p) else os.path.join(prep_dir, p)
    df["full_cube"] = df["cube_npy"].apply(resolve)
    df["full_mask"] = df["mask_png"].apply(resolve) if "mask_png" in df.columns else ""
    df["writer_id"] = df["name"].apply(parse_writer_id)
    df["page_id"]   = df["name"].apply(parse_page_id)
    df["dataset"]   = tag
    df = df[df["full_cube"].apply(os.path.exists)].reset_index(drop=True)
    in_ch = _CACHE.get(str(df.iloc[0]["full_cube"])).shape[0]
    print(f"  [{dtag(tag)}] {len(df)} cubes | {df['writer_id'].nunique()} writers | {in_ch} bands")
    return df


# ============================================================
#  ENCODER
# ============================================================
class ResNet18Encoder(nn.Module):
    """
    Input : (B, in_ch, H, W)
    Output: (B, embed_dim)  L2-normalised
    """
    def __init__(self, in_ch: int, pretrained: bool = True, embed_dim: int = 256):
        super().__init__()
        base = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        old  = base.conv1
        base.conv1 = nn.Conv2d(in_ch, old.out_channels,
                               kernel_size=old.kernel_size,
                               stride=old.stride, padding=old.padding, bias=False)
        if pretrained and in_ch != 3:
            with torch.no_grad():
                if in_ch > 3:
                    base.conv1.weight[:, :3].copy_(old.weight)
                    mean_w = old.weight.mean(dim=1, keepdim=True)
                    for c in range(3, in_ch):
                        base.conv1.weight[:, c:c+1].copy_(mean_w)
                else:
                    base.conv1.weight.copy_(
                        old.weight.mean(dim=1, keepdim=True).repeat(1, in_ch, 1, 1))
        base.fc        = nn.Identity()
        self.backbone  = base
        self.proj      = nn.Sequential(
            nn.Linear(512, embed_dim), nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim))
        self.in_ch     = in_ch
        self.embed_dim = embed_dim

    def forward(self, x):
        return F.normalize(self.proj(self.backbone(x)), dim=-1)


# ============================================================
#  LOAD iVision PRETRAINED BACKBONE  (never retrained)
# ============================================================
def load_ivisio_hhid_encoder(ckpt_path: str, in_ch: int) -> ResNet18Encoder:
    """
    Always loads from checkpoint — iVision backbone is NEVER retrained.
    Aborts clearly if checkpoint is missing.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"\n  ERROR: iVision checkpoint not found: {ckpt_path}\n"
            f"  This pipeline requires a pre-trained iVision backbone.\n"
            f"  Train it once with the v3 pipeline (no --use_pretrained flag),\n"
            f"  then run this script.")
    ckpt    = torch.load(ckpt_path, map_location="cpu")
    encoder = ResNet18Encoder(in_ch=in_ch, pretrained=False,
                              embed_dim=cfg.embed_dim)
    encoder.load_state_dict(ckpt["state_dict"], strict=True)
    acc = float(ckpt.get("acc", 0.0))
    print(f"  [iVision] Loaded backbone from {ckpt_path}  (acc={acc:.3f})")
    print(f"  [iVision] Backbone is FROZEN — no retraining will happen.")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


# ============================================================
#  TRANSFER iVision WEIGHTS → UWA ENCODER
# ============================================================
def transfer_to_uwa(ivisio_hhid_encoder: ResNet18Encoder,
                    uwa_in_ch: int) -> ResNet18Encoder:
    """
    Build a new ResNet18Encoder for UWA's band count (33).
    Copy all layers from iVision encoder except conv1 (different bands).
    conv1 is initialised by averaging iVision's channel weights.
    The returned encoder has requires_grad=True on all params — it will
    be trained on UWA data only.
    """
    uwa_enc = ResNet18Encoder(in_ch=uwa_in_ch, pretrained=False,
                              embed_dim=cfg.embed_dim)
    src_sd  = ivisio_hhid_encoder.state_dict()
    tgt_sd  = uwa_enc.state_dict()
    copied = skipped = 0
    for k in tgt_sd:
        if k not in src_sd:
            skipped += 1; continue
        if tgt_sd[k].shape == src_sd[k].shape:
            tgt_sd[k].copy_(src_sd[k]); copied += 1
        elif "conv1.weight" in k:
            # iVision: (64, 149, 7, 7)  →  UWA: (64, 33, 7, 7)
            # Average across iVision channels, replicate for UWA bands
            mean_w = src_sd[k].mean(dim=1, keepdim=True)
            tgt_sd[k].copy_(mean_w.repeat(1, uwa_in_ch, 1, 1))
            copied += 1
        else:
            skipped += 1
    uwa_enc.load_state_dict(tgt_sd)
    for p in uwa_enc.parameters():
        p.requires_grad_(True)   # UWA encoder IS trainable
    print(f"  [UWA transfer] Copied {copied} tensors, skipped {skipped}  "
          f"(iVision {ivisio_hhid_encoder.in_ch}ch → UWA {uwa_in_ch}ch)")
    return uwa_enc


# ============================================================
#  PROTONET DATASETS & SAMPLERS
# ============================================================
class LazyLRUCubeDataset:
    def __init__(self, items, label_to_int, keep_fp16, max_cache_items):
        self.items           = items
        self.label_to_int    = dict(label_to_int)
        self.keep_fp16       = keep_fp16
        self.max_cache_items = int(max_cache_items)
        self.y               = [self.label_to_int[str(x["label"])] for x in items]
        self.class_to_indices: Dict[int, List[int]] = {}
        for i, yi in enumerate(self.y):
            self.class_to_indices.setdefault(yi, []).append(i)
        self._cache = collections.OrderedDict()
        t0 = _load_cube_lazy(self.items[0]["path"], keep_fp16=keep_fp16)
        self.in_ch = int(t0.shape[0])
        self._cache[0] = t0

    def __len__(self): return len(self.items)

    def get_x(self, idx: int) -> torch.Tensor:
        if idx in self._cache:
            t = self._cache.pop(idx); self._cache[idx] = t; return t
        t = _load_cube_lazy(self.items[idx]["path"], keep_fp16=self.keep_fp16)
        self._cache[idx] = t
        while len(self._cache) > self.max_cache_items:
            self._cache.popitem(last=False)
        return t

    def get_y(self, idx: int): return self.y[idx]

def pad_to_same_size(tensors: list) -> list:
    """
    Pad a list of (C, H, W) tensors to the same spatial size (max H, max W).
    Required when cubes have slightly different spatial dimensions (e.g. UWA).
    """
    if len(tensors) == 0:
        return tensors
    max_H = max(t.shape[-2] for t in tensors)
    max_W = max(t.shape[-1] for t in tensors)
    padded = []
    for t in tensors:
        H, W = t.shape[-2], t.shape[-1]
        pad_H = max_H - H
        pad_W = max_W - W
        if pad_H > 0 or pad_W > 0:
            t = F.pad(t.float(), (0, pad_W, 0, pad_H))
        padded.append(t)
    return padded



class TrainEpisodeSampler:
    def __init__(self, tr, seed=0):
        self.rng = random.Random(seed); self.tr = tr
        self.classes = sorted(tr.keys())

    def eligible(self, k, q):
        return [c for c in self.classes if len(self.tr[c]) >= k+q]

    def sample_episode(self, n_way, k_shot, q_query):
        elig   = self.eligible(k_shot, q_query)
        n      = min(n_way, len(elig))
        chosen = self.rng.sample(elig, n)
        s_idx, q_idx, y_s, y_q = [], [], [], []
        for ny, cls in enumerate(chosen):
            picks  = self.rng.sample(self.tr[cls], k_shot + q_query)
            s_idx += picks[:k_shot]; q_idx += picks[k_shot:]
            y_s   += [ny]*k_shot;   y_q   += [ny]*q_query
        return n, s_idx, q_idx, torch.tensor(y_s), torch.tensor(y_q)


class CrossSplitEvalSampler:
    def __init__(self, tr, te, seed=0):
        self.rng = random.Random(seed); self.tr = tr; self.te = te
        self.common = sorted(set(tr.keys()) & set(te.keys()))

    def eligible(self, k, q):
        return [c for c in self.common if len(self.tr[c])>=k and len(self.te[c])>=q]

    def sample_episode(self, n_way, k_shot, q_query):
        elig   = self.eligible(k_shot, q_query)
        n      = min(n_way, len(elig))
        chosen = self.rng.sample(elig, n)
        s_idx, q_idx, y_s, y_q = [], [], [], []
        for ny, cls in enumerate(chosen):
            s_idx += self.rng.sample(self.tr[cls], k_shot)
            q_idx += self.rng.sample(self.te[cls], q_query)
            y_s   += [ny]*k_shot; y_q += [ny]*q_query
        return n, s_idx, q_idx, torch.tensor(y_s), torch.tensor(y_q), chosen


def build_crosssplit_items(index_csv, prep_dir, seed=0, test_per_class=1):
    df = pd.read_csv(index_csv)
    def resolve(p):
        p = str(p)
        return p if os.path.isabs(p) else os.path.join(prep_dir, p)
    df["full_path"] = df["cube_npy"].apply(resolve)
    df = df[df["full_path"].apply(os.path.exists)].copy()
    def infer_label(name):
        m = re.search(r"([wW]\d{1,4})", str(name))
        return m.group(1).lower() if m else str(name).lower()
    df["label"] = df["name"].apply(infer_label)
    rng = random.Random(seed)
    train_rows, test_rows = [], []
    for cls, g in df.groupby("label"):
        idxs = list(g.index); rng.shuffle(idxs)
        if len(idxs) >= test_per_class + 1:
            test_rows  += idxs[:test_per_class]
            train_rows += idxs[test_per_class:]
        else:
            train_rows += idxs
    train_items = [{"path": df.loc[i,"full_path"], "label": df.loc[i,"label"]} for i in train_rows]
    test_items  = [{"path": df.loc[i,"full_path"], "label": df.loc[i,"label"]} for i in test_rows]
    return train_items, test_items


# ============================================================
#  UWA PROTONET TRAINING
# ============================================================
@torch.no_grad()
def episodic_eval(encoder, train_ds, test_ds, ev_samp,
                  n_way, k_shot, q_query, n_episodes, device):
    encoder.eval(); accs = []
    for _ in range(n_episodes):
        n, s_idx, q_idx, y_s, y_q, _ = ev_samp.sample_episode(n_way, k_shot, q_query)
        xs = torch.stack(pad_to_same_size([train_ds.get_x(i) for i in s_idx])).to(device).float()
        xq = torch.stack(pad_to_same_size([test_ds.get_x(i)  for i in q_idx])).to(device).float()
        y_s = y_s.to(device); y_q = y_q.to(device)
        logits = prototypical_logits(encoder(xs), y_s, encoder(xq), n)
        accs.append((logits.argmax(1)==y_q).float().mean().item())
    return float(np.mean(accs)), compute_ci95(accs)


def train_uwa_protonet(uwa_enc: ResNet18Encoder,
                       train_items, test_items) -> Tuple[ResNet18Encoder, float]:
    """
    Train ProtoNet backbone ONLY on UWA WIHSI data.
    uwa_enc is already warm-started from iVision weights via transfer_to_uwa().
    Saves best model to cfg.uwa_ckpt.
    """
    print(f"\n{'='*60}")
    print(f"  [UWA ProtoNet] Training on UWA WIHSI ONLY")
    print(f"  n_way={cfg.uwa_n_way}  k={cfg.k_shot}  epochs={cfg.uwa_epochs}"
          f"  eps/epoch={cfg.uwa_episodes_per_epoch}  lr={cfg.uwa_lr}")
    print(f"  Warm-started from iVision backbone (transfer learning)")
    print(f"{'='*60}")

    device       = torch.device(cfg.device)
    set_seed(cfg.seed)
    all_labels   = sorted(set(x["label"] for x in train_items + test_items))
    label_to_int = {c:i for i,c in enumerate(all_labels)}

    train_ds = LazyLRUCubeDataset(train_items, label_to_int, cfg.fp16_cache, cfg.max_cache)
    test_ds  = LazyLRUCubeDataset(test_items,  label_to_int, cfg.fp16_cache, cfg.max_cache)

    uwa_enc  = uwa_enc.to(device)
    opt      = torch.optim.AdamW(uwa_enc.parameters(),
                                 lr=cfg.uwa_lr, weight_decay=cfg.uwa_wd)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.uwa_epochs)

    tr_samp  = TrainEpisodeSampler(train_ds.class_to_indices, seed=cfg.seed)
    ev_samp  = CrossSplitEvalSampler(train_ds.class_to_indices,
                                     test_ds.class_to_indices, seed=cfg.seed)

    n_tr_elig = len(tr_samp.eligible(cfg.k_shot, cfg.train_q_query))
    n_ev_elig = len(ev_samp.eligible(cfg.k_shot, cfg.eval_q_query))
    print(f"  Train eligible: {n_tr_elig}  Eval eligible: {n_ev_elig}  "
          f"Classes: {len(all_labels)}")

    best, best_state = -1.0, None

    for epoch in range(1, cfg.uwa_epochs + 1):
        uwa_enc.train(); losses, accs = [], []
        for epi in range(1, cfg.uwa_episodes_per_epoch + 1):
            n, s_idx, q_idx, y_s, y_q = tr_samp.sample_episode(
                cfg.uwa_n_way, cfg.k_shot, cfg.train_q_query)
            xs = torch.stack(pad_to_same_size([train_ds.get_x(i) for i in s_idx])).to(device).float()
            xq = torch.stack(pad_to_same_size([train_ds.get_x(i) for i in q_idx])).to(device).float()
            y_s = y_s.to(device); y_q = y_q.to(device)
            z_s = uwa_enc(xs); z_q = uwa_enc(xq)
            logits = prototypical_logits(z_s, y_s, z_q, n)
            loss   = F.cross_entropy(logits, y_q)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(loss.item())
            accs.append((logits.argmax(1)==y_q).float().mean().item())
        sched.step()

        te_acc, te_ci = episodic_eval(
            uwa_enc, train_ds, test_ds, ev_samp,
            cfg.uwa_n_way, cfg.k_shot, cfg.eval_q_query,
            cfg.test_episodes, device)
        print(f"Epoch {epoch:02d}/{cfg.uwa_epochs} | tr_acc={np.mean(accs):.3f} | "
              f"test_ep_acc={te_acc:.3f}±{te_ci:.3f} | best={best:.3f}")
        if te_acc > best:
            best       = te_acc
            best_state = {k: v.detach().cpu().clone()
                          for k,v in uwa_enc.state_dict().items()}

    uwa_enc.load_state_dict(best_state)
    print(f"\n  [UWA] Best episodic acc = {best:.3f}")
    torch.save({"state_dict": uwa_enc.state_dict(), "acc": best,
                "in_ch": uwa_enc.in_ch, "embed_dim": uwa_enc.embed_dim,
                "trained_on": "UWA_WIHSI",
                "transfer_from": "iVision_HHID"}, cfg.uwa_ckpt)
    print(f"  Saved: {cfg.uwa_ckpt}")
    return uwa_enc, best


def load_uwa_encoder(ckpt_path: str, uwa_in_ch: int) -> ResNet18Encoder:
    ckpt    = torch.load(ckpt_path, map_location="cpu")
    encoder = ResNet18Encoder(in_ch=uwa_in_ch, pretrained=False,
                              embed_dim=cfg.embed_dim)
    encoder.load_state_dict(ckpt["state_dict"], strict=True)
    acc = float(ckpt.get("acc", 0.0))
    print(f"  [UWA] Loaded backbone from {ckpt_path}  (acc={acc:.3f})")
    encoder.eval()
    return encoder


# ============================================================
#  CUBE EMBEDDING  (full-cube, OOM-safe fallback to patches)
# ============================================================
def embed_cube(encoder: ResNet18Encoder, path: str,
               enc_in_ch: int, device: torch.device,
               n_crops: int = 8, seed: int = 0) -> torch.Tensor:
    """
    Embeds one cube with the full-cube forward pass (Exp-5 style).
    Zero-pads or truncates spectral bands to enc_in_ch.
    Falls back to averaged patch crops on GPU OOM.
    """
    cube = _CACHE.get(str(path)).float()
    C    = cube.shape[0]
    if C < enc_in_ch:
        cube = F.pad(cube, (0,0,0,0,0,enc_in_ch-C))
    elif C > enc_in_ch:
        cube = cube[:enc_in_ch]
    # Full-cube forward
    try:
        with torch.no_grad():
            z = encoder(cube.unsqueeze(0).to(device))
        return F.normalize(z.squeeze(0), dim=-1).cpu()
    except RuntimeError:                 # OOM — fall back to crops
        torch.cuda.empty_cache()
        rng = random.Random(seed)
        ps  = cfg.patch_size
        _,H,W = cube.shape
        if H<ps or W<ps: cube=F.pad(cube,(0,max(0,ps-W),0,max(0,ps-H)))
        _,H,W = cube.shape
        patches = torch.stack([
            cube[:, rng.randint(0,H-ps):rng.randint(0,H-ps)+ps,
                    rng.randint(0,W-ps):rng.randint(0,W-ps)+ps].float()
            for _ in range(n_crops)])
        with torch.no_grad():
            z = encoder(patches.to(device))
        return F.normalize(z.mean(0), dim=-1).cpu()


# ============================================================
#  PATCH SAMPLING  (for downstream MIL tasks)
# ============================================================
def sample_patches(cube_t: torch.Tensor, n: int,
                   rng: random.Random, augment: bool = False) -> torch.Tensor:
    C,H,W = cube_t.shape
    ps    = cfg.patch_size
    if H<ps or W<ps:
        cube_t = F.pad(cube_t,(0,max(0,ps-W),0,max(0,ps-H)))
        C,H,W  = cube_t.shape
    gray = cube_t.float().mean(0)
    patches, tries = [], 0
    while len(patches) < n and tries < n*200:
        tries += 1
        y = rng.randint(0,H-ps); x = rng.randint(0,W-ps)
        p = cube_t[:,y:y+ps,x:x+ps].float()
        if gray[y:y+ps,x:x+ps].mean() >= cfg.min_ink:
            patches.append(p)
    if not patches:
        y=(H-ps)//2; x=(W-ps)//2
        patches=[cube_t[:,y:y+ps,x:x+ps].float() for _ in range(n)]
    while len(patches) < n:
        patches.append(patches[rng.randint(0,len(patches)-1)])
    bag = torch.stack(patches[:n])
    if augment:
        if rng.random()<0.5: bag=torch.flip(bag,dims=[3])
        if rng.random()<0.2: bag=torch.flip(bag,dims=[2])
        a=0.9+0.2*rng.random(); b=-0.05+0.1*rng.random()
        bag=(a*bag+b).clamp(0,1)
    return bag


# ============================================================
#  MIL ENCODER + CLASSIFIER  (downstream fine-tune)
# ============================================================
class MILDataset(Dataset):
    def __init__(self, df, label_col, label_map, enc_in_ch, training=True, seed=0):
        self.df        = df.reset_index(drop=True)
        self.label_col = label_col; self.label_map = label_map
        self.enc_in_ch = enc_in_ch; self.training  = training
        self.rng       = random.Random(seed + (0 if training else 9999))

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        cube = _CACHE.get(str(row["full_cube"])).float()
        C = cube.shape[0]
        if C < self.enc_in_ch: cube=F.pad(cube,(0,0,0,0,0,self.enc_in_ch-C))
        elif C > self.enc_in_ch: cube=cube[:self.enc_in_ch]
        bag = sample_patches(cube, cfg.k_patches, self.rng, augment=self.training)
        y   = self.label_map[row[self.label_col]]
        return bag, torch.tensor(y, dtype=torch.long), str(row["name"])


def collate_bags(batch):
    bags,ys,names = zip(*batch)
    return torch.stack(bags), torch.tensor(ys,dtype=torch.long), list(names)


class MILEncoder(nn.Module):
    def __init__(self, encoder: ResNet18Encoder):
        super().__init__()
        self.enc = encoder; D = encoder.embed_dim
        self.attn = nn.Sequential(nn.Linear(D,D//2),nn.Tanh(),nn.Linear(D//2,1))

    def forward(self, bag):
        B,K,C,H,W = bag.shape
        feats = self.enc(bag.view(B*K,C,H,W)).view(B,K,-1)
        a     = torch.softmax(self.attn(feats).squeeze(-1),dim=1)
        return (a.unsqueeze(-1)*feats).sum(1)


class ClassifierHead(nn.Module):
    def __init__(self,in_dim,num_classes,dropout=0.3):
        super().__init__()
        self.head=nn.Sequential(nn.LayerNorm(in_dim),nn.Dropout(dropout),nn.Linear(in_dim,num_classes))
    def forward(self,x): return self.head(x)


# ============================================================
#  SPLIT HELPERS
# ============================================================
def leave_one_page_out(df, seed=42):
    rng=random.Random(seed); tr,te=[],[]
    for _,g in df.groupby("writer_id"):
        idxs=list(g.index)
        if len(idxs)==1: tr.extend(idxs); continue
        pick=rng.choice(idxs); te.append(pick)
        tr.extend(i for i in idxs if i!=pick)
    return df.loc[tr].reset_index(drop=True), df.loc[te].reset_index(drop=True)


def kfold_split(df, k_test=1, seed=42):
    rng=random.Random(seed); tr,te=[],[]
    for _,g in df.groupby("writer_id"):
        idxs=list(g.index); rng.shuffle(idxs)
        te.extend(idxs[:k_test]); tr.extend(idxs[k_test:])
    return df.loc[tr].reset_index(drop=True), df.loc[te].reset_index(drop=True)


# ============================================================
#  GENERIC MIL FINE-TUNE LOOP
# ============================================================
def fine_tune(encoder, train_df, test_df, label_col,
              label_map, enc_in_ch, tag):
    device = torch.device(cfg.device)
    # Repeat small datasets
    if len(train_df) < 200:
        reps     = math.ceil(200/len(train_df))
        train_df = pd.concat([train_df]*reps, ignore_index=True)

    # Build a COPY of encoder for MIL (keeps original intact for other tasks)
    import copy
    enc_copy = copy.deepcopy(encoder)

    mil  = MILEncoder(enc_copy).to(device)
    head = ClassifierHead(cfg.embed_dim, len(label_map)).to(device)

    tr_ds  = MILDataset(train_df,label_col,label_map,enc_in_ch,training=True)
    te_ds  = MILDataset(test_df, label_col,label_map,enc_in_ch,training=False)
    tr_ldr = DataLoader(tr_ds,batch_size=cfg.cls_batch,shuffle=True, num_workers=0,collate_fn=collate_bags)
    te_ldr = DataLoader(te_ds,batch_size=cfg.cls_batch,shuffle=False,num_workers=0,collate_fn=collate_bags)

    # Phase 1: backbone frozen
    for p in mil.enc.parameters(): p.requires_grad_(False)
    phase1 = [p for p in mil.parameters() if p.requires_grad] + list(head.parameters())
    opt    = torch.optim.AdamW(phase1, lr=cfg.cls_lr*3, weight_decay=cfg.cls_wd)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and cfg.device=="cuda")
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.cls_epochs)

    best_acc, bad      = -1.0, 0
    best_yt = best_yp  = None
    unfreeze_ep        = cfg.cls_epochs // 2

    for ep in range(1, cfg.cls_epochs+1):
        # Phase 2: unfreeze at midpoint
        if ep == unfreeze_ep + 1:
            for p in mil.enc.parameters(): p.requires_grad_(True)
            opt   = torch.optim.AdamW(list(mil.parameters())+list(head.parameters()),
                                      lr=cfg.cls_lr, weight_decay=cfg.cls_wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt, T_max=cfg.cls_epochs - unfreeze_ep)

        mil.train(); head.train()
        for bag,y,_ in tr_ldr:
            bag,y = bag.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp and cfg.device=="cuda"):
                loss = F.cross_entropy(head(mil(bag)), y, label_smoothing=0.05)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()

        mil.eval(); head.eval(); yt,yp = [],[]
        with torch.no_grad():
            for bag,y,_ in te_ldr:
                bag=bag.to(device)
                yp.extend(head(mil(bag)).argmax(1).cpu().tolist())
                yt.extend(y.tolist())
        acc = float(np.mean(np.array(yt)==np.array(yp)))
        if acc > best_acc:
            best_acc=acc; bad=0; best_yt=yt; best_yp=yp
        else:
            bad += 1
        if ep%10==0:
            print(f"    [{dtag(tag)}] ep{ep:02d} val_acc={acc:.4f} best={best_acc:.4f}")
        if bad >= cfg.patience: break

    return best_acc, np.array(best_yt), np.array(best_yp)


# ============================================================
#  VISUALISATION HELPERS
# ============================================================
def plot_pr(yt, score, pos_label, tag, name, out_dir):
    prec,rec,_ = precision_recall_curve(yt,score,pos_label=pos_label)
    ap = average_precision_score(yt,score)
    fig,ax=plt.subplots(figsize=(5,4))
    ax.plot(rec,prec,lw=2); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"PR — {name} [{dtag(tag)}]  AP={ap:.3f}")
    ax.set_xlim(0,1); ax.set_ylim(0,1.05)
    save_fig(fig,os.path.join(out_dir,f"{name}_pr_{tag}.png")); return ap

def plot_det(fpr,fnr,tag,name,out_dir):
    try:
        from scipy.special import ndtri
        nd = lambda x: ndtri(np.clip(x,1e-6,1-1e-6))
        fig,ax=plt.subplots(figsize=(5,4)); ax.plot(nd(fpr),nd(fnr),lw=2)
        tks=[0.001,0.01,0.05,0.1,0.2,0.5]
        ax.set_xticks([nd(t) for t in tks]); ax.set_xticklabels([f"{t*100:.1f}" for t in tks],fontsize=7)
        ax.set_yticks([nd(t) for t in tks]); ax.set_yticklabels([f"{t*100:.1f}" for t in tks],fontsize=7)
        ax.set_xlabel("FPR (%)"); ax.set_ylabel("FNR (%)")
        ax.set_title(f"DET — {name} [{dtag(tag)}]")
        save_fig(fig,os.path.join(out_dir,f"{name}_det_{tag}.png"))
    except Exception: pass

def plot_score_hist(pos,neg,thr,tag,name,out_dir,xlabel,plbl,nlbl):
    fig,ax=plt.subplots(figsize=(6,4))
    bins=np.linspace(min(pos.min(),neg.min()),max(pos.max(),neg.max()),60)
    ax.hist(pos,bins=bins,alpha=0.55,label=plbl,color="#2196F3")
    ax.hist(neg,bins=bins,alpha=0.55,label=nlbl,color="#F44336")
    ax.axvline(thr,color="black",linestyle="--",lw=1.5,label=f"thr={thr:.3f}")
    ax.set_xlabel(xlabel); ax.set_ylabel("Count")
    ax.set_title(f"Score Dist — {name} [{dtag(tag)}]"); ax.legend(fontsize=8)
    save_fig(fig,os.path.join(out_dir,f"{name}_score_hist_{tag}.png"))

def plot_f1_bar(yt,yp,cnames,tag,name,out_dir):
    f1s=f1_score(yt,yp,average=None,labels=list(range(len(cnames))),zero_division=0)
    fig,ax=plt.subplots(figsize=(max(6,len(cnames)*0.35),4))
    colors=["#4CAF50" if v>=0.7 else "#FF9800" if v>=0.4 else "#F44336" for v in f1s]
    ax.bar(cnames,f1s,color=colors)
    ax.axhline(f1s.mean(),color="navy",linestyle="--",label=f"mean={f1s.mean():.2f}")
    ax.set_ylim(0,1.05); ax.set_ylabel("F1"); ax.set_title(f"Per-class F1 — {name} [{dtag(tag)}]")
    ax.legend(fontsize=8)
    if len(cnames)>20: ax.set_xticks([]); ax.set_xlabel("Writers →")
    else: plt.xticks(rotation=45,ha="right",fontsize=7)
    plt.tight_layout(); save_fig(fig,os.path.join(out_dir,f"{name}_f1bar_{tag}.png"))

def plot_tsne(embs,lbls,lnames,tag,name,out_dir,n_max=20):
    if not HAS_TSNE: return
    uniq=sorted(set(lbls))
    if len(uniq)>n_max:
        chosen=uniq[:n_max]; mask=np.isin(lbls,chosen)
        embs=embs[mask]; lbls=lbls[mask]; uniq=chosen
    try:
        z2d=TSNE(n_components=2,perplexity=min(30,len(embs)-1),
                 random_state=42,n_iter=1000).fit_transform(embs)
        cmap=plt.cm.get_cmap("tab20",len(uniq))
        fig,ax=plt.subplots(figsize=(8,7))
        for i,lbl in enumerate(uniq):
            m=lbls==lbl; nm=lnames[lbl] if lbl<len(lnames) else str(lbl)
            ax.scatter(z2d[m,0],z2d[m,1],s=30,alpha=0.7,color=cmap(i),label=nm)
        ax.set_title(f"t-SNE — {name} [{dtag(tag)}]")
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        if len(uniq)<=20: ax.legend(fontsize=6,ncol=2)
        plt.tight_layout(); save_fig(fig,os.path.join(out_dir,f"{name}_tsne_{tag}.png"))
    except Exception as e: print(f"  [warn] t-SNE: {e}")

def plot_sim_heatmap(proto_mat,proto_keys,tag,out_dir):
    P=F.normalize(proto_mat,dim=1); sim=(P@P.T).numpy(); n=sim.shape[0]
    fig,ax=plt.subplots(figsize=(max(5,n//4),max(4,n//4)))
    im=ax.imshow(sim,cmap="RdYlGn",vmin=-1,vmax=1)
    ax.set_title(f"Proto Cosine Sim — {dtag(tag)}"); fig.colorbar(im,ax=ax,fraction=0.03)
    if n<=60:
        ax.set_xticks(range(n)); ax.set_xticklabels([str(k) for k in proto_keys],fontsize=5,rotation=90)
        ax.set_yticks(range(n)); ax.set_yticklabels([str(k) for k in proto_keys],fontsize=5)
    plt.tight_layout(); save_fig(fig,os.path.join(out_dir,f"proto_sim_heatmap_{tag}.png"))

def plot_roc(fpr,tpr,roc_auc,tag,name,out_dir):
    fig,ax=plt.subplots(figsize=(5,4))
    ax.plot(fpr,tpr,lw=2,label=f"AUC={roc_auc:.3f}"); ax.plot([0,1],[0,1],"--",color="grey")
    ax.legend(); ax.set_title(f"{name} ROC — {dtag(tag)}")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    save_fig(fig,os.path.join(out_dir,f"{name}_roc_{tag}.png"))


# ============================================================
#  MODEL PROFILING
# ============================================================
def profile_encoder(encoder, in_ch, device, spatial, out_dir, tag, n_runs=50):
    encoder=encoder.to(device).eval()
    H,W=spatial
    total_p=sum(p.numel() for p in encoder.parameters())
    train_p=sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    dummy=torch.zeros(1,in_ch,H,W).to(device); gflops=None
    if HAS_THOP:
        try: macs,_=thop_profile(encoder,inputs=(dummy,),verbose=False); gflops=macs/1e9
        except Exception: pass
    if gflops is None:
        gflops=1.814*(H*W*in_ch)/(224*224*3)
    with torch.no_grad():
        for _ in range(5): encoder(dummy)
    if device=="cuda": torch.cuda.synchronize()
    times=[]
    with torch.no_grad():
        for _ in range(n_runs):
            t0=time.perf_counter(); encoder(dummy)
            if device=="cuda": torch.cuda.synchronize()
            times.append((time.perf_counter()-t0)*1000)
    times=np.array(times)
    res={"params_total_M":total_p/1e6,"params_train_M":train_p/1e6,
         "gflops":gflops,"infer_ms_mean":float(times.mean()),
         "infer_ms_std":float(times.std()),"infer_ms_p95":float(np.percentile(times,95))}
    print(f"\n  [{dtag(tag)} profile] Params={res['params_total_M']:.2f}M  "
          f"GFLOPs={res['gflops']:.2f}  "
          f"Infer={res['infer_ms_mean']:.1f}±{res['infer_ms_std']:.1f}ms")
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    axes[0].bar(["Total\nParams","Trainable\nParams"],
                [res["params_total_M"],res["params_train_M"]],
                color=["#1565C0","#42A5F5"])
    for i,v in enumerate([res["params_total_M"],res["params_train_M"]]):
        axes[0].text(i,v+0.05,f"{v:.2f}M",ha="center",fontsize=9)
    axes[0].set_title(f"Model Size [{dtag(tag)}]"); axes[0].set_ylabel("Params (M)")
    axes[1].bar(["Full-cube\ninference"],[res["infer_ms_mean"]],
                yerr=[res["infer_ms_std"]],color="#388E3C",capsize=6,
                error_kw={"elinewidth":2})
    axes[1].text(0,res["infer_ms_mean"]+res["infer_ms_std"]+0.5,
                 f"mean={res['infer_ms_mean']:.1f}ms\np95={res['infer_ms_p95']:.1f}ms",
                 ha="center",fontsize=9)
    axes[1].set_title(f"Inference — {res['gflops']:.2f} GFLOPs [{dtag(tag)}]")
    axes[1].set_ylabel("ms / sample")
    plt.tight_layout()
    save_fig(fig,os.path.join(out_dir,f"model_profile_{tag}.png"))
    return res


# ============================================================
#  TASK 1: WRITER IDENTIFICATION  (prototype nearest-neighbour)
# ============================================================
def task_writer_identification(df, encoder, enc_in_ch, tag, out_dir):
    print(f"\n{'='*60}\n  TASK 1 — Writer ID [{dtag(tag)}]\n{'='*60}")
    device=torch.device(cfg.device); encoder=encoder.to(device).eval()
    writers=sorted(df["writer_id"].unique())
    w2i={w:i for i,w in enumerate(writers)}; i2w={i:w for w,i in w2i.items()}
    train_df,test_df=leave_one_page_out(df)
    print(f"  Train={len(train_df)}  Test={len(test_df)}")

    print("  Building prototypes…")
    protos={}
    for wid,grp in train_df.groupby("writer_id"):
        zs=[embed_cube(encoder,row["full_cube"],enc_in_ch,device)
            for _,row in grp.iterrows()]
        protos[wid]=torch.stack(zs).mean(0)
    proto_keys   = sorted(protos.keys())
    proto_matrix = torch.stack([protos[k] for k in proto_keys])

    yt,yp=[],[]
    for _,row in test_df.iterrows():
        z     = embed_cube(encoder,row["full_cube"],enc_in_ch,device).unsqueeze(0)
        dists = pairwise_sq_dists(z,proto_matrix).squeeze(0)
        pred_w= proto_keys[int(dists.argmin().item())]
        yt.append(w2i[int(row["writer_id"])]); yp.append(w2i[pred_w])

    yt=np.array(yt); yp=np.array(yp)
    acc=float((yt==yp).mean()); mcc=float(matthews_corrcoef(yt,yp))
    print(f"\n  Accuracy={acc:.4f}  MCC={mcc:.4f}")
    print(classification_report(yt,yp,
          target_names=[f"w{i2w[i]:02d}" for i in range(len(writers))],zero_division=0))

    n_cls=len(writers); cnames=[f"w{i2w[i]:02d}" for i in range(n_cls)]

    # Confusion matrix
    cm=confusion_matrix(yt,yp); sz=max(8,n_cls//2)
    fig,ax=plt.subplots(figsize=(sz,sz))
    im=ax.imshow(cm,cmap="Blues")
    ax.set_title(f"Writer ID — {dtag(tag)}  acc={acc:.3f}  MCC={mcc:.3f}")
    fig.colorbar(im,ax=ax); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    save_fig(fig,os.path.join(out_dir,f"writer_id_cm_{tag}.png"))

    # Micro ROC
    yt_b=label_binarize(yt,classes=list(range(n_cls)))
    yp_b=label_binarize(yp,classes=list(range(n_cls)))
    fpr_m,tpr_m,_=roc_curve(yt_b.ravel(),yp_b.ravel()); roc_auc_m=auc(fpr_m,tpr_m)
    plot_roc(fpr_m,tpr_m,roc_auc_m,tag,"writer_id",out_dir)

    # Per-class F1 bar
    plot_f1_bar(yt,yp,cnames,tag,"writer_id",out_dir)

    # t-SNE — all cubes
    all_embs,all_lbls=[],[]
    for wid,grp in train_df.groupby("writer_id"):
        for _,row in grp.iterrows():
            all_embs.append(embed_cube(encoder,row["full_cube"],enc_in_ch,device).numpy())
            all_lbls.append(w2i[wid])
    for _,row in test_df.iterrows():
        all_embs.append(embed_cube(encoder,row["full_cube"],enc_in_ch,device).numpy())
        all_lbls.append(w2i[int(row["writer_id"])])
    plot_tsne(np.array(all_embs),np.array(all_lbls),cnames,tag,"writer_id",out_dir)

    # Similarity heatmap between writer prototypes
    plot_sim_heatmap(proto_matrix,proto_keys,tag,out_dir)

    return {"writer_id_acc":acc,"writer_id_mcc":mcc,"writer_id_roc_auc":roc_auc_m}


# ============================================================
#  TASK 2: INK MISMATCH
# ============================================================
def task_ink_mismatch(df, encoder, enc_in_ch, tag, out_dir):
    print(f"\n{'='*60}\n  TASK 2 — Ink Mismatch [{dtag(tag)}]\n{'='*60}")
    device=torch.device(cfg.device); encoder=encoder.to(device).eval()
    rng=random.Random(cfg.seed+2)
    by_w={w:list(g.index) for w,g in df.groupby("writer_id")}
    writers=list(by_w.keys()); n_pairs=min(2000,len(writers)*20)

    sims,labels=[],[]
    for _ in range(n_pairs):
        w1=rng.choice(writers)
        genuine=rng.random()<0.5 and len(by_w[w1])>=2
        if genuine: i1,i2=rng.sample(by_w[w1],2); lbl=0
        else:
            w2=rng.choice([w for w in writers if w!=w1])
            i1=rng.choice(by_w[w1]); i2=rng.choice(by_w[w2]); lbl=1
        z1=embed_cube(encoder,df.iloc[i1]["full_cube"],enc_in_ch,device,n_crops=4,seed=i1)
        z2=embed_cube(encoder,df.iloc[i2]["full_cube"],enc_in_ch,device,n_crops=4,seed=i2)
        sims.append(F.cosine_similarity(z1.unsqueeze(0),z2.unsqueeze(0)).item())
        labels.append(lbl)

    sims=np.array(sims); labels=np.array(labels)
    scores=1.0-sims
    fpr,tpr,ths=roc_curve(labels,scores); roc_auc=auc(fpr,tpr)
    fnr=1-tpr; eer_idx=np.argmin(np.abs(fpr-fnr))
    eer=float((fpr[eer_idx]+fnr[eer_idx])/2); thr=float(ths[eer_idx])
    yp=(scores>=thr).astype(int); acc=float((yp==labels).mean())
    print(f"  AUC={roc_auc:.4f}  EER={eer:.4f}  acc@EER={acc:.4f}")
    print(classification_report(labels,yp,target_names=["same_ink","diff_ink"],zero_division=0))

    plot_roc(fpr,tpr,roc_auc,tag,"ink_mismatch",out_dir)
    plot_pr(labels,scores,pos_label=1,tag=tag,name="ink_mismatch",out_dir=out_dir)
    plot_det(fpr,1-tpr,tag,name="ink_mismatch",out_dir=out_dir)
    plot_score_hist(sims[labels==0],sims[labels==1],thr=1-thr,
                    tag=tag,name="ink_mismatch",out_dir=out_dir,
                    xlabel="Cosine Sim",plbl="Same Ink",nlbl="Diff Ink")
    cm=confusion_matrix(labels,yp); fig,ax=plt.subplots()
    im=ax.imshow(cm,cmap="Oranges")
    ax.set_xticks([0,1]); ax.set_xticklabels(["same","diff"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["same","diff"])
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    ax.set_title(f"Ink Mismatch CM — {dtag(tag)}  acc={acc:.3f}"); fig.colorbar(im,ax=ax)
    save_fig(fig,os.path.join(out_dir,f"ink_mismatch_cm_{tag}.png"))
    return {"ink_mismatch_acc":acc,"ink_mismatch_auc":roc_auc,"ink_mismatch_eer":eer}


# ============================================================
#  TASK 3: WRITER VERIFICATION
# ============================================================
def task_writer_verification(df, encoder, enc_in_ch, tag, out_dir):
    print(f"\n{'='*60}\n  TASK 3 — Writer Verification [{dtag(tag)}]\n{'='*60}")
    device=torch.device(cfg.device); encoder=encoder.to(device).eval()
    rng=random.Random(cfg.seed+3)
    by_w={w:list(g.index) for w,g in df.groupby("writer_id")}; writers=list(by_w.keys())
    n_pairs=min(cfg.verif_pairs,len(writers)*(len(writers)-1)*2)

    sims,labels=[],[]
    for _ in range(n_pairs):
        w1=rng.choice(writers)
        genuine=rng.random()<0.5 and len(by_w[w1])>=2
        if genuine: i1,i2=rng.sample(by_w[w1],2); lbl=1
        else:
            w2=rng.choice([w for w in writers if w!=w1])
            i1=rng.choice(by_w[w1]); i2=rng.choice(by_w[w2]); lbl=0
        z1=embed_cube(encoder,df.iloc[i1]["full_cube"],enc_in_ch,device,n_crops=8,seed=i1)
        z2=embed_cube(encoder,df.iloc[i2]["full_cube"],enc_in_ch,device,n_crops=8,seed=i2)
        sims.append(F.cosine_similarity(z1.unsqueeze(0),z2.unsqueeze(0)).item())
        labels.append(lbl)

    sims=np.array(sims); labels=np.array(labels)
    fpr,tpr,ths=roc_curve(labels,sims); roc_auc=auc(fpr,tpr)
    fnr=1-tpr; eer_idx=np.argmin(np.abs(fpr-fnr))
    eer=float((fpr[eer_idx]+fnr[eer_idx])/2); best_thr=float(ths[eer_idx])
    acc=float(((sims>=best_thr).astype(int)==labels).mean())
    print(f"  AUC={roc_auc:.4f}  EER={eer:.4f}  acc={acc:.4f}  thr={best_thr:.3f}")

    plot_roc(fpr,tpr,roc_auc,tag,"writer_verif",out_dir)
    plot_pr(labels,sims,pos_label=1,tag=tag,name="writer_verif",out_dir=out_dir)
    plot_det(fpr,1-tpr,tag,name="writer_verif",out_dir=out_dir)
    plot_score_hist(sims[labels==1],sims[labels==0],thr=best_thr,
                    tag=tag,name="writer_verif",out_dir=out_dir,
                    xlabel="Cosine Sim",plbl="Genuine",nlbl="Impostor")
    return {"verif_auc":roc_auc,"verif_eer":eer,"verif_acc":acc}


# ============================================================
#  TASK 4: FORGERY DETECTION
# ============================================================
def task_forgery_detection(df, encoder, enc_in_ch, tag, out_dir):
    print(f"\n{'='*60}\n  TASK 4 — Forgery Detection [{dtag(tag)}]\n{'='*60}")
    df2=df.copy()
    df2["forgery_label"]=df2["page_id"].apply(lambda p:0 if p<=2 else 1)
    counts=df2["forgery_label"].value_counts()
    print(f"  genuine={counts.get(0,0)}  forged={counts.get(1,0)}")
    if counts.get(0,0)==0 or counts.get(1,0)==0:
        print("  [warn] Insufficient forgery labels — skipping."); return {}
    f2i={0:0,1:1}; train_df,test_df=kfold_split(df2,k_test=1)
    acc,yt,yp=fine_tune(encoder,train_df,test_df,"forgery_label",f2i,enc_in_ch,"forgery")
    print(f"\n  Accuracy={acc:.4f}")
    print(classification_report(yt,yp,target_names=["genuine","forged"],zero_division=0))
    cm=confusion_matrix(yt,yp); fig,ax=plt.subplots()
    im=ax.imshow(cm,cmap="Reds"); ax.set_xticks([0,1]); ax.set_xticklabels(["genuine","forged"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["genuine","forged"])
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center",
                                    color="white" if cm[i,j]>cm.max()/2 else "black")
    ax.set_title(f"Forgery CM — {dtag(tag)}  acc={acc:.3f}"); fig.colorbar(im,ax=ax)
    save_fig(fig,os.path.join(out_dir,f"forgery_cm_{tag}.png"))
    fpr_f,tpr_f,_=roc_curve(yt,yp); auc_f=auc(fpr_f,tpr_f)
    plot_roc(fpr_f,tpr_f,auc_f,tag,"forgery",out_dir)
    plot_pr(yt,yp,pos_label=1,tag=tag,name="forgery",out_dir=out_dir)
    plot_det(fpr_f,1-tpr_f,tag,"forgery",out_dir)
    return {"forgery_acc":acc,"forgery_auc":auc_f}


# ============================================================
#  TASK 5: AGE & GENDER  (iVision ONLY — UWA has no labels)
# ============================================================
IVISIO_META={
    1:(24,0),2:(21,1),3:(21,1),4:(21,1),5:(24,1),6:(21,1),7:(22,0),8:(21,0),
    9:(25,1),10:(26,1),11:(27,1),12:(30,0),13:(22,0),14:(26,0),15:(24,0),16:(24,0),
    17:(19,0),18:(20,0),19:(19,0),20:(19,0),21:(22,0),22:(21,0),23:(22,0),24:(19,0),
    25:(19,0),26:(20,0),27:(19,0),28:(19,0),29:(19,0),30:(18,0),31:(18,0),32:(19,0),
    33:(19,0),34:(20,0),35:(19,0),36:(19,0),37:(20,0),38:(19,0),39:(22,0),40:(20,0),
    41:(19,0),42:(18,0),43:(20,0),44:(20,0),45:(20,0),46:(19,0),47:(20,0),48:(19,0),
    49:(18,1),50:(21,0),51:(30,0),52:(26,0),53:(25,1),54:(27,0),
}
AGE_NAMES=["18-19","20-21","22-23","24-25","26+"]
def age_bin(a):
    return 0 if a<=19 else 1 if a<=21 else 2 if a<=23 else 3 if a<=25 else 4

def task_age_gender(df, encoder, enc_in_ch, tag, out_dir):
    print(f"\n{'='*60}\n  TASK 5 — Age & Gender [{dtag(tag)}]\n{'='*60}")
    df2=df.copy()
    df2["gender_label"]=df2["writer_id"].map(lambda w:IVISIO_META.get(w,(-1,-1))[1])
    df2["age_bin"]     =df2["writer_id"].map(lambda w:age_bin(IVISIO_META.get(w,(20,-1))[0]))
    df2=df2[df2["gender_label"]>=0].reset_index(drop=True)
    if len(df2)==0:
        print("  [warn] No age/gender metadata — skipping."); return {}
    results={}
    for tcol,n_cls,cnames in [("gender_label",2,["Male","Female"]),("age_bin",5,AGE_NAMES)]:
        print(f"\n  — {tcol} —")
        lmap={v:v for v in range(n_cls)}; train_df,test_df=kfold_split(df2,k_test=1)
        acc,yt,yp=fine_tune(encoder,train_df,test_df,tcol,lmap,enc_in_ch,tcol)
        print(f"    Accuracy={acc:.4f}")
        print(classification_report(yt,yp,target_names=cnames[:n_cls],zero_division=0))
        cm=confusion_matrix(yt,yp,labels=list(range(n_cls)))
        fig,ax=plt.subplots(figsize=(max(4,n_cls),max(4,n_cls)))
        im=ax.imshow(cm,cmap="Purples"); ax.set_xticks(range(n_cls)); ax.set_xticklabels(cnames[:n_cls])
        ax.set_yticks(range(n_cls)); ax.set_yticklabels(cnames[:n_cls])
        for i in range(n_cls):
            for j in range(n_cls): ax.text(j,i,str(cm[i,j]),ha="center",va="center")
        ax.set_title(f"{tcol} [{dtag(tag)}]  acc={acc:.3f}"); fig.colorbar(im,ax=ax); plt.tight_layout()
        save_fig(fig,os.path.join(out_dir,f"{tcol}_cm_{tag}.png"))
        plot_f1_bar(yt,yp,cnames[:n_cls],tag,tcol,out_dir)
        from sklearn.metrics import roc_curve as _rc,auc as _au
        from sklearn.preprocessing import label_binarize as _lb
        yt_b=_lb(yt,classes=list(range(n_cls))); yp_b=_lb(yp,classes=list(range(n_cls)))
        if yt_b.shape[1]>1:
            fpr_t,tpr_t,_=_rc(yt_b.ravel(),yp_b.ravel()); auc_t=_au(fpr_t,tpr_t)
            plot_roc(fpr_t,tpr_t,auc_t,tag,tcol,out_dir); results[f"{tcol}_roc_auc"]=auc_t
        results[f"{tcol}_acc"]=acc
    return results


# ============================================================
#  CROSS-DATASET ZERO-SHOT EVAL
# ============================================================
def cross_dataset_eval(df_src, df_tgt, encoder_src, enc_in_ch_src,
                       tag_src, tag_tgt, out_dir):
    """
    Build per-writer prototypes from df_src using encoder_src,
    then classify each page in df_tgt by nearest prototype.
    Spectral band mismatch is handled automatically (zero-pad / truncate).
    """
    print(f"\n{'='*60}\n  CROSS-DATASET: {tag_src} → {tag_tgt}\n{'='*60}")
    device=torch.device(cfg.device); encoder_src=encoder_src.to(device).eval()

    print(f"  Building {tag_src} prototypes…")
    by_w={w:list(g.index) for w,g in df_src.groupby("writer_id")}
    protos={}
    for w,idxs in by_w.items():
        zs=[embed_cube(encoder_src,df_src.iloc[i]["full_cube"],enc_in_ch_src,device)
            for i in idxs]
        protos[w]=torch.stack(zs).mean(0)
    proto_keys  =sorted(protos.keys())
    proto_tensor=torch.stack([protos[k] for k in proto_keys]).to(device)

    in_ch_tgt=_CACHE.get(str(df_tgt.iloc[0]["full_cube"])).shape[0]
    print(f"  Src bands={enc_in_ch_src}  Tgt bands={in_ch_tgt}  "
          f"(embed_cube auto-adapts)")

    yt,yp=[],[]
    for idx in range(len(df_tgt)):
        z   =embed_cube(encoder_src,df_tgt.iloc[idx]["full_cube"],
                        enc_in_ch_src,device).unsqueeze(0).to(device)
        pred=proto_keys[int(pairwise_sq_dists(z,proto_tensor).argmin(1).item())]
        yt.append(int(df_tgt.iloc[idx]["writer_id"])); yp.append(pred)

    yt=np.array(yt); yp=np.array(yp)
    acc=float((yt==yp).mean())
    print(f"  Zero-shot acc ({tag_src}→{tag_tgt}): {acc:.4f}")
    print(classification_report(yt,yp,zero_division=0))
    return {f"cross_{tag_src}_to_{tag_tgt}_acc":acc}


# ============================================================
#  RADAR SUMMARY PLOT
# ============================================================
def plot_radar(metrics, title, out_path):
    names=list(metrics.keys()); vals=[metrics[n] for n in names]
    N=len(names); angles=[n/N*2*math.pi for n in range(N)]+[0]; vals2=vals+[vals[0]]
    fig,ax=plt.subplots(subplot_kw={"projection":"polar"},figsize=(7,7))
    ax.set_theta_offset(math.pi/2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(names,size=9); ax.set_ylim(0,1)
    ax.plot(angles,vals2,"o-",lw=2,color="#1f77b4")
    ax.fill(angles,vals2,alpha=0.2,color="#1f77b4")
    ax.set_title(title,size=12,pad=20)
    save_fig(fig,out_path)


# ============================================================
#  RESULTS WRITER  (txt + json)
# ============================================================
def save_results(all_results, profile, tag, out_dir, in_ch, n_writers, n_cubes, spatial):
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    H,W = spatial
    task_map=[
        ("ProtoNet Backbone",         "protonet",      [("Episodic Acc","proto_acc")]),
        ("Task 1: Writer ID",         "writer_id",     [("Accuracy","writer_id_acc"),
                                                         ("MCC","writer_id_mcc"),
                                                         ("ROC AUC","writer_id_roc_auc")]),
        ("Task 2: Ink Mismatch",      "ink_mismatch",  [("Acc@EER","ink_mismatch_acc"),
                                                         ("AUC","ink_mismatch_auc"),
                                                         ("EER","ink_mismatch_eer")]),
        ("Task 3: Writer Verif",      "writer_verif",  [("AUC","verif_auc"),
                                                         ("EER","verif_eer"),
                                                         ("Acc@EER","verif_acc")]),
        ("Task 4: Forgery Det",       "forgery",       [("Accuracy","forgery_acc"),
                                                         ("AUC","forgery_auc")]),
        ("Task 5: Gender Pred",       "age_gender",    [("Accuracy","gender_label_acc"),
                                                         ("ROC AUC","gender_label_roc_auc")]),
        ("Task 5: Age Pred",          "age_gender",    [("Accuracy","age_bin_acc"),
                                                         ("ROC AUC","age_bin_roc_auc")]),
        ("Cross: iVision HHID→UWA",        "cross",         [("Zero-shot Acc","cross_ivisio_hhid_to_uwa_acc")]),
        ("Cross: UWA→iVision HHID",        "cross",         [("Zero-shot Acc","cross_uwa_to_ivisio_hhid_acc")]),
    ]
    lines=[
        "="*70,
        f"  HYPERSPECTRAL FORENSICS v4 — {tag.upper()} RESULTS",
        f"  Generated  : {ts}",
        f"  Dataset    : {dtag(tag)}  ({n_writers} writers, {n_cubes} cubes, "
        f"{in_ch} bands, {H}×{W})",
        f"  Device     : {cfg.device}",
        "="*70,"",
        "── MODEL PROFILE ──────────────────────────────────────────────────",
        f"  Architecture     : ResNet18Encoder (ProtoNet, Exp-5 style)",
        f"  Input            : (1, {in_ch}, {H}, {W})",
        f"  Total params     : {profile['params_total_M']:.4f} M",
        f"  Trainable params : {profile['params_train_M']:.4f} M",
        f"  GFLOPs           : {profile['gflops']:.4f}",
        f"  Inference mean   : {profile['infer_ms_mean']:.2f} ms",
        f"  Inference std    : {profile['infer_ms_std']:.2f} ms",
        f"  Inference p95    : {profile['infer_ms_p95']:.2f} ms",
        "","── TASK RESULTS ─────────────────────────────────────────────────────",
    ]
    for title_t,key,metrics_l in task_map:
        res=all_results.get(key,{})
        if not res: continue
        lines.append(f"\n  {title_t}")
        lines.append(f"  {'─'*50}")
        for label,mkey in metrics_l:
            val=res.get(mkey,None)
            if val is not None:
                lines.append(f"    {label:<30}: {val:.4f}")
    lines+=[
        "","── OUTPUT FILES ────────────────────────────────────────────────────",
        f"  Directory : {out_dir}/",
        "  Plots     : *_cm_*.png  *_roc_*.png  *_pr_*.png  *_det_*.png",
        "              *_score_hist_*.png  *_f1bar_*.png  *_tsne_*.png",
        "              proto_sim_heatmap_*.png  model_profile_*.png  pipeline_radar.png",
        f"  Results   : results_{tag}.txt  results_{tag}.json",
        "","="*70,
    ]
    txt_path=os.path.join(out_dir,f"results_{tag}.txt")
    with open(txt_path,"w") as f: f.write("\n".join(lines)+"\n")
    print(f"  [saved] {txt_path}")

    json_path=os.path.join(out_dir,f"results_{tag}.json")
    with open(json_path,"w") as f:
        json.dump({
            "timestamp":ts,"dataset":tag,"device":cfg.device,
            "model_profile":{k:float(v) for k,v in profile.items()},
            "results":{k:{kk:float(vv) for kk,vv in v.items()}
                       for k,v in all_results.items()},
        },f,indent=2)
    print(f"  [saved] {json_path}")


# ============================================================
#  RUN ONE COMPLETE PIPELINE FOR A DATASET
# ============================================================
def run_pipeline(df, encoder, enc_in_ch, tag, out_dir,
                 proto_acc, spatial, run_age_gender=False):
    """
    Runs all tasks on df using encoder, saves all plots & results.
    Returns all_results dict.
    """
    ensure_dir(out_dir)
    all_results={"protonet":{"proto_acc":proto_acc}}

    # Model profile
    profile=profile_encoder(encoder,enc_in_ch,cfg.device,spatial,out_dir,tag)

    # Tasks
    all_results["writer_id"]   =task_writer_identification(df,encoder,enc_in_ch,tag,out_dir)
    all_results["ink_mismatch"]=task_ink_mismatch(df,encoder,enc_in_ch,tag,out_dir)
    all_results["writer_verif"]=task_writer_verification(df,encoder,enc_in_ch,tag,out_dir)
    all_results["forgery"]     =task_forgery_detection(df,encoder,enc_in_ch,tag,out_dir)
    if run_age_gender:
        all_results["age_gender"]=task_age_gender(df,encoder,enc_in_ch,tag,out_dir)

    # Radar
    radar_metrics={
        "Writer ID":    all_results["writer_id"].get("writer_id_acc",0),
        "Ink Mismatch": all_results["ink_mismatch"].get("ink_mismatch_auc",0),
        "Verif AUC":    all_results["writer_verif"].get("verif_auc",0),
        "Forgery":      all_results["forgery"].get("forgery_acc",0) if all_results["forgery"] else 0,
        "Proto Acc":    proto_acc,
    }
    if run_age_gender and all_results.get("age_gender"):
        radar_metrics["Gender Acc"]=all_results["age_gender"].get("gender_label_acc",0)
        radar_metrics["Age Acc"]   =all_results["age_gender"].get("age_bin_acc",0)
    plot_radar(radar_metrics,f"Forensics Pipeline — {dtag(tag).upper()}",
               os.path.join(out_dir,"pipeline_radar.png"))

    # Save results
    save_results(all_results,profile,tag,out_dir,
                 enc_in_ch,df["writer_id"].nunique(),len(df),spatial)

    print(f"\n  {'='*55}")
    print(f"  {dtag(tag).upper()} PIPELINE COMPLETE — outputs in {out_dir}/")
    print(f"  {'='*55}")
    return all_results, profile


# ============================================================
#  MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Hyperspectral Forensics v4 — iVision HHID pretrained + UWA WIHSI trained here")
    parser.add_argument("--dataset", default="both",
        choices=["both", "ivisio_hhid", "uwa"],
        help="Which pipeline(s) to run  [default: both]")
    args = parser.parse_args()

    set_seed(cfg.seed)
    ensure_dir(cfg.ivisio_hhid_out)
    ensure_dir(cfg.uwa_out)

    run_ivisio_hhid = args.dataset in ("both", "ivisio_hhid")
    run_uwa    = args.dataset in ("both", "uwa")

    print("\n" + "="*65)
    print("  HYPERSPECTRAL DOCUMENT FORENSICS — PIPELINE v4")
    print("="*65)
    print(f"  Device       : {cfg.device}")
    print(f"  Pipeline(s)  : {args.dataset}")
    print(f"  iVision ckpt : {cfg.ivisio_hhid_ckpt}")
    print(f"               -> ALWAYS loaded from disk, never retrained")
    print(f"  UWA ckpt     : {cfg.uwa_ckpt}")
    uwa_ckpt_exists = os.path.exists(cfg.uwa_ckpt)
    print(f"               -> {'Found — will load' if uwa_ckpt_exists else 'Not found — will train on UWA data only'}")
    print("="*65 + "\n")

    # ----------------------------------------------------------
    #  Load iVision dataset (always needed — even for UWA pipeline,
    #  iVision weights are used as transfer init if UWA must train)
    # ----------------------------------------------------------
    df_ivisio_hhid    = load_index(cfg.ivisio_hhid_index, cfg.ivisio_hhid_dir, "iVision_HHID")
    in_ch_ivisio_hhid = _CACHE.get(str(df_ivisio_hhid.iloc[0]["full_cube"])).shape[0]

    # ----------------------------------------------------------
    #  iVision backbone — ALWAYS loaded from checkpoint.
    #  There is NO code path that retrains it.
    # ----------------------------------------------------------
    iv_hhid_encoder   = load_ivisio_hhid_encoder(cfg.ivisio_hhid_ckpt, in_ch_ivisio_hhid)
    iv_hhid_proto_acc = float(torch.load(cfg.ivisio_hhid_ckpt,
                                    map_location="cpu").get("acc", 0.955))
    iv_hhid_encoder.to(cfg.device)

    # ==========================================================
    #  PIPELINE A — iVision HHID
    #  Uses the pretrained iVision backbone, no training at all
    # ==========================================================
    if run_ivisio_hhid:
        print("\n" + "#"*65)
        print("  PIPELINE A — iVision HHID")
        print("  Backbone : pretrained checkpoint (no retraining)")
        print("  Tasks    : Writer ID / Ink Mismatch / Verification")
        print("             Forgery Detection / Age & Gender")
        print("#"*65)
        run_pipeline(
            df             = df_ivisio_hhid,
            encoder        = iv_hhid_encoder,
            enc_in_ch      = in_ch_ivisio_hhid,
            tag            = "ivisio_hhid",
            out_dir        = cfg.ivisio_hhid_out,
            proto_acc      = iv_hhid_proto_acc,
            spatial        = (512, 650),
            run_age_gender = True,    # iVision has age/gender metadata
        )

    # ----------------------------------------------------------
    #  Load UWA dataset
    # ----------------------------------------------------------
    try:
        df_uwa    = load_index(cfg.uwa_index, cfg.uwa_dir, "UWA_WIHSI")
        in_ch_uwa = _CACHE.get(str(df_uwa.iloc[0]["full_cube"])).shape[0]
        has_uwa   = True
    except FileNotFoundError:
        print("\n  [warn] UWA index not found — skipping UWA pipeline.")
        has_uwa   = False
        in_ch_uwa = 0
        df_uwa    = None
        uwa_encoder = None

    # ==========================================================
    #  PIPELINE B — UWA WIHSI V1.0
    #
    #  The UWA backbone is completely separate from iVision.
    #  Two sub-cases, fully automatic — no flags needed:
    #
    #  Case 1 — UWA checkpoint already exists (./best_protonet_uwa.pt):
    #            Load it directly. No training. Fast.
    #
    #  Case 2 — UWA checkpoint does not exist:
    #            a. Copy iVision weights into a new 33-band encoder
    #               (conv1 is averaged across bands; rest is copied)
    #            b. Train ProtoNet on UWA data ONLY (iVision is frozen)
    #            c. Save checkpoint so future runs hit Case 1
    #
    #  In both cases, the iVision backbone is NOT modified.
    # ==========================================================
    if run_uwa and has_uwa:
        print("\n" + "#"*65)
        print("  PIPELINE B — UWA WIHSI V1.0")
        print("  Tasks    : Writer ID / Ink Mismatch / Verification")
        print("             Forgery Detection")
        print("#"*65)

        if uwa_ckpt_exists:
            # Case 1: checkpoint already trained — load and run tasks
            print(f"\n  UWA checkpoint found — loading (no retraining)")
            uwa_encoder   = load_uwa_encoder(cfg.uwa_ckpt, in_ch_uwa)
            uwa_proto_acc = float(torch.load(cfg.uwa_ckpt,
                                             map_location="cpu").get("acc", 0.0))
        else:
            # Case 2: train UWA backbone from scratch
            print(f"\n  No UWA checkpoint at {cfg.uwa_ckpt}")
            print(f"\n  Step 1/2  Transfer iVision weights -> UWA encoder")
            print(f"            iVision: {in_ch_ivisio_hhid} bands  ->  UWA: {in_ch_uwa} bands")
            print(f"            (conv1 averaged; all other layers copied directly)")
            uwa_encoder = transfer_to_uwa(iv_hhid_encoder, uwa_in_ch=in_ch_uwa)

            print(f"\n  Step 2/2  Train ProtoNet on UWA WIHSI ONLY")
            train_items, test_items = build_crosssplit_items(
                cfg.uwa_index, cfg.uwa_dir, seed=cfg.seed, test_per_class=1)
            print(f"            Split: {len(train_items)} train  /  {len(test_items)} test")
            uwa_encoder, uwa_proto_acc = train_uwa_protonet(
                uwa_encoder, train_items, test_items)
            print(f"\n  UWA backbone saved -> {cfg.uwa_ckpt}")
            print(f"  Future runs will load it automatically.")

        uwa_encoder.to(cfg.device).eval()

        run_pipeline(
            df             = df_uwa,
            encoder        = uwa_encoder,
            enc_in_ch      = in_ch_uwa,
            tag            = "uwa",
            out_dir        = cfg.uwa_out,
            proto_acc      = uwa_proto_acc,
            spatial        = (480, 752),
            run_age_gender = False,   # UWA has no age/gender metadata
        )

    # ==========================================================
    #  CROSS-DATASET ZERO-SHOT EVAL
    #  Runs automatically when both pipelines were active
    # ==========================================================
    if run_ivisio_hhid and run_uwa and has_uwa:
        print("\n" + "#"*65)
        print("  CROSS-DATASET EVALUATION")
        print("  iVision encoder -> zero-shot classify UWA pages")
        print("  UWA encoder     -> zero-shot classify iVision pages")
        print("#"*65)

        cross_dir = "./results_cross"
        ensure_dir(cross_dir)

        r1 = cross_dataset_eval(df_ivisio_hhid,  df_uwa, iv_hhid_encoder,  in_ch_ivisio_hhid,
                                 "ivisio_hhid", "uwa",    cross_dir)
        r2 = cross_dataset_eval(df_uwa, df_ivisio_hhid,  uwa_encoder, in_ch_uwa,
                                 "uwa",    "ivisio_hhid", cross_dir)

        cross_results = {**r1, **r2}
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        txt_path  = os.path.join(cross_dir, "cross_dataset_results.txt")
        json_path = os.path.join(cross_dir, "cross_dataset_results.json")
        with open(txt_path, "w") as f:
            f.write("="*60 + "\n")
            f.write(f"  CROSS-DATASET RESULTS  [{ts}]\n")
            f.write("="*60 + "\n\n")
            for k, v in cross_results.items():
                f.write(f"  {k:<44}: {v:.4f}\n")
        with open(json_path, "w") as f:
            json.dump({"timestamp": ts,
                       "results": {k: float(v) for k,v in cross_results.items()}},
                      f, indent=2)
        print(f"\n  [saved] {txt_path}")
        print(f"  [saved] {json_path}")

    # ----------------------------------------------------------
    #  DONE
    # ----------------------------------------------------------
    print("\n" + "="*65)
    print("  ALL PIPELINES COMPLETE")
    print("="*65)
    if run_ivisio_hhid:
        print(f"  iVision results  -> {cfg.ivisio_hhid_out}/")
    if run_uwa and has_uwa:
        print(f"  UWA results      -> {cfg.uwa_out}/")
    if run_ivisio_hhid and run_uwa and has_uwa:
        print(f"  Cross-dataset    -> ./results_cross/")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()
