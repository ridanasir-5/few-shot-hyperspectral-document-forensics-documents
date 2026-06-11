"""
===========================================================================
  HYPERSPECTRAL DOCUMENT FORENSICS — ABLATION STUDY
===========================================================================
  Purpose
  ───────
  Systematically measure the contribution of every major design choice
  in the forensics pipeline.  Each ablation axis isolates one variable
  while keeping everything else at the "full model" (v4) configuration.

  Ablation Axes  (7 axes, ~24 conditions total)
  ─────────────────────────────────────────────
  A. BACKBONE ARCHITECTURE
       A1  ResNet18  (full model — baseline)
       A2  ResNet50
       A3  EfficientNet-B0
       A4  MobileNetV3-Small  (lightweight)

  B. SPECTRAL BAND INPUT STRATEGY
       B1  Full bands (149 iVision / 33 UWA)  ← baseline
       B2  PCA to 3 components, treat as RGB
       B3  Uniform band selection → 3 channels (R/G/B analogues)
       B4  Random subset of 16 bands

  C. PROJECTION HEAD EMBEDDING DIM
       C1  64
       C2  128
       C3  256  ← baseline
       C4  512

  D. WEIGHT INITIALISATION STRATEGY
       D1  iVision transfer + ImageNet conv1  ← baseline
       D2  ImageNet pretrained only (no iVision transfer)
       D3  Random initialisation (no pretrained weights at all)
       D4  iVision transfer, proj head random only (freeze backbone)

  E. EPISODIC TRAINING LOSS
       E1  Prototypical (squared-L2 distance)  ← baseline
       E2  Cosine prototypical (cosine distance)
       E3  Supervised Contrastive (SupCon, temperature=0.07)
       E4  Batch-hard Triplet loss (margin=0.3)

  F. K-SHOT (support size at meta-train time)
       F1  k=1  ← baseline
       F2  k=2
       F3  k=3
       F4  k=5

  G. EMBEDDING NORMALISATION
       G1  L2-norm after projection  ← baseline
       G2  No normalisation
       G3  BatchNorm1d on projection output
       G4  LayerNorm on projection output

  For each condition the script:
    1. Builds the encoder variant
    2. Runs episodic evaluation on the iVision test split
       (using the iVision pretrained backbone as init where possible,
        otherwise trains from scratch for N episodes as proxy)
    3. Runs Writer-ID prototype-NN on iVision
    4. Runs Ink-Mismatch AUC on iVision
    5. Records all metrics in a unified results table

  Output
  ──────
  ./ablation_results/
    ablation_table.csv               ← full numeric table
    ablation_table.txt               ← formatted table for paper
    ablation_summary.json            ← machine-readable
    plots/
      ablation_A_backbone.png
      ablation_B_bands.png
      ablation_C_embdim.png
      ablation_D_init.png
      ablation_E_loss.png
      ablation_F_kshot.png
      ablation_G_norm.png
      ablation_heatmap_all.png       ← all axes × all metrics
      ablation_radar_full_vs_best.png

  Usage
  ─────
  # Full ablation (all axes, both datasets):
      python hyperspectral_forensics_ablation.py

  # Single axis:
      python hyperspectral_forensics_ablation.py --axis A
      python hyperspectral_forensics_ablation.py --axis E

  # Quick mode — fewer eval episodes (faster, noisier):
      python hyperspectral_forensics_ablation.py --quick

  # iVision only:
      python hyperspectral_forensics_ablation.py --dataset ivisio

  Required files
  ──────────────
  ./best_protonet_crosssplit.pt          ← iVision backbone (must exist)
  ./data_preprocessed/preprocess_index.csv
  ./uwa_preprocessed/preprocess_index.csv   (optional)
===========================================================================
"""

import os, re, copy, math, time, random, argparse, warnings
import collections, json, datetime, itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from torchvision import models
    from torchvision.models import (ResNet18_Weights, ResNet50_Weights,
                                    EfficientNet_B0_Weights,
                                    MobileNet_V3_Small_Weights)
except ImportError:
    raise ImportError("pip install torchvision>=0.13")

try:
    from sklearn.metrics import (roc_curve, auc, matthews_corrcoef,
                                  classification_report)
    from sklearn.decomposition import PCA
except ImportError:
    raise ImportError("pip install scikit-learn")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")


# ============================================================
#  BASE CONFIG  (full-model / baseline settings)
# ============================================================
@dataclass
class CFG:
    # Paths
    ivisio_index:  str  = "./data_preprocessed/preprocess_index.csv"
    ivisio_dir:    str  = "./data_preprocessed"
    uwa_index:     str  = "./uwa_preprocessed/preprocess_index.csv"
    uwa_dir:       str  = "./uwa_preprocessed"
    ivisio_ckpt:   str  = "./best_protonet_crosssplit.pt"
    out_dir:       str  = "./ablation_results"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed:   int = 42

    # Baseline embed dim
    embed_dim: int = 256

    # ProtoNet eval episodes (set lower with --quick)
    test_episodes:  int = 200
    # Proxy training episodes per condition (for conditions that need training)
    proxy_episodes: int = 500   # reduced from full 6000 for ablation speed
    proxy_lr:       float = 1e-4
    proxy_wd:       float = 1e-4

    # Episodic config — baseline
    n_way:         int = 5
    k_shot:        int = 1
    train_q_query: int = 2
    eval_q_query:  int = 1
    max_cache:     int = 16
    fp16_cache:    bool = True

    # Downstream (Writer-ID + Ink-Mismatch only for ablation speed)
    patch_size:   int   = 128
    k_patches:    int   = 24
    min_ink:      float = 0.12
    verif_pairs:  int   = 500    # reduced from 1000 for ablation speed


cfg = CFG()


# ============================================================
#  SEED & UTILS
# ============================================================
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")

def compute_ci95(accs):
    if len(accs) < 2: return 0.0
    return 1.96 * float(np.std(accs, ddof=1)) / math.sqrt(len(accs))

def pairwise_sq_dists(a, b):
    a2 = (a*a).sum(1, keepdim=True)
    b2 = (b*b).sum(1, keepdim=True).t()
    return (a2 + b2 - 2*a@b.t()).clamp(min=0)

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
        x = np.transpose(x, (2, 0, 1))
    mu = float(x.mean()); sig = float(x.std()) + 1e-6
    x  = (x - mu) / sig
    t  = torch.from_numpy(np.ascontiguousarray(x))
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
    df["writer_id"] = df["name"].apply(parse_writer_id)
    df["page_id"]   = df["name"].apply(parse_page_id)
    df["dataset"]   = tag
    df = df[df["full_cube"].apply(os.path.exists)].reset_index(drop=True)
    in_ch = _CACHE.get(str(df.iloc[0]["full_cube"])).shape[0]
    print(f"  [{tag}] {len(df)} cubes | {df['writer_id'].nunique()} writers | {in_ch} bands")
    return df


# ============================================================
#  SPECTRAL BAND PREPROCESSING  (Ablation B)
# ============================================================
def preprocess_bands_full(cube: torch.Tensor, n_ch: int) -> torch.Tensor:
    """B1 — identity: use all bands as-is."""
    return cube  # (C, H, W)

def preprocess_bands_pca3(cube: torch.Tensor, n_ch: int) -> torch.Tensor:
    """B2 — PCA to 3 principal components (treat as pseudo-RGB)."""
    C, H, W = cube.shape
    X = cube.float().reshape(C, H*W).T.numpy()   # (H*W, C)
    n_comp = min(3, C)
    pca = PCA(n_components=n_comp, random_state=42)
    Xp  = pca.fit_transform(X)                   # (H*W, 3)
    Xp  = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-6)
    out = torch.from_numpy(Xp.T.reshape(n_comp, H, W).astype(np.float32))
    if n_comp < 3:
        out = out.repeat(3 // n_comp + 1, 1, 1)[:3]
    return out  # (3, H, W)

def preprocess_bands_uniform3(cube: torch.Tensor, n_ch: int) -> torch.Tensor:
    """B3 — uniform band selection → 3 channels (low / mid / high band)."""
    C, H, W = cube.shape
    idxs = [0, C // 2, C - 1]
    return cube[idxs].float()   # (3, H, W)

def preprocess_bands_random16(cube: torch.Tensor, n_ch: int,
                               seed: int = 42) -> torch.Tensor:
    """B4 — random subset of 16 bands (same subset every time for reproducibility)."""
    C, H, W = cube.shape
    rng  = random.Random(seed)
    k    = min(16, C)
    idxs = sorted(rng.sample(range(C), k))
    out  = cube[idxs].float()   # (16, H, W)
    if out.shape[0] < 16:
        out = F.pad(out, (0,0,0,0,0,16-out.shape[0]))
    return out

BAND_STRATEGIES = {
    "full":      (preprocess_bands_full,      None),   # out_ch = original
    "pca3":      (preprocess_bands_pca3,      3),
    "uniform3":  (preprocess_bands_uniform3,  3),
    "random16":  (preprocess_bands_random16,  16),
}


# ============================================================
#  ENCODER VARIANTS  (Ablation A — backbone, C — embed_dim,
#                     G — normalisation)
# ============================================================
class BaseEncoder(nn.Module):
    """Shared interface for all encoder variants."""
    in_ch:     int
    embed_dim: int

    def forward(self, x) -> torch.Tensor:
        raise NotImplementedError


class ResNet18Encoder(BaseEncoder):
    def __init__(self, in_ch: int, embed_dim: int = 256,
                 pretrained: bool = True, norm: str = "l2"):
        super().__init__()
        base = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self._adapt_conv1(base, in_ch, pretrained)
        base.fc    = nn.Identity()
        self.backbone  = base
        self.proj      = self._make_proj(512, embed_dim)
        self.norm_mode = norm
        self.in_ch     = in_ch
        self.embed_dim = embed_dim

    @staticmethod
    def _adapt_conv1(base, in_ch, pretrained):
        old = base.conv1
        base.conv1 = nn.Conv2d(in_ch, old.out_channels,
                               kernel_size=old.kernel_size,
                               stride=old.stride, padding=old.padding, bias=False)
        if pretrained and in_ch != 3:
            with torch.no_grad():
                if in_ch > 3:
                    base.conv1.weight[:, :3].copy_(old.weight)
                    mw = old.weight.mean(dim=1, keepdim=True)
                    for c in range(3, in_ch):
                        base.conv1.weight[:, c:c+1].copy_(mw)
                else:
                    base.conv1.weight.copy_(
                        old.weight.mean(dim=1, keepdim=True).repeat(1, in_ch, 1, 1))

    @staticmethod
    def _make_proj(in_feat: int, embed_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_feat, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def _apply_norm(self, z: torch.Tensor) -> torch.Tensor:
        if self.norm_mode == "l2":
            return F.normalize(z, dim=-1)
        elif self.norm_mode == "none":
            return z
        elif self.norm_mode == "bn":
            # Applied at build time via proj head — here just return
            return z
        elif self.norm_mode == "ln":
            return z
        return F.normalize(z, dim=-1)

    def forward(self, x):
        return self._apply_norm(self.proj(self.backbone(x)))


class ResNet50Encoder(BaseEncoder):
    def __init__(self, in_ch: int, embed_dim: int = 256,
                 pretrained: bool = True, norm: str = "l2"):
        super().__init__()
        base = models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        ResNet18Encoder._adapt_conv1(base, in_ch, pretrained)
        base.fc    = nn.Identity()
        self.backbone  = base
        self.proj      = ResNet18Encoder._make_proj(2048, embed_dim)
        self.norm_mode = norm
        self.in_ch     = in_ch
        self.embed_dim = embed_dim

    def forward(self, x):
        z = self.proj(self.backbone(x))
        return F.normalize(z, dim=-1) if self.norm_mode == "l2" else z


class EfficientNetB0Encoder(BaseEncoder):
    def __init__(self, in_ch: int, embed_dim: int = 256,
                 pretrained: bool = True, norm: str = "l2"):
        super().__init__()
        base  = models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        # Replace first conv
        old   = base.features[0][0]
        base.features[0][0] = nn.Conv2d(
            in_ch, old.out_channels, kernel_size=old.kernel_size,
            stride=old.stride, padding=old.padding, bias=False)
        if pretrained and in_ch != 3:
            with torch.no_grad():
                mw = old.weight.mean(dim=1, keepdim=True)
                base.features[0][0].weight.copy_(mw.repeat(1, in_ch, 1, 1))
        base.classifier = nn.Identity()
        self.backbone   = base
        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.proj       = ResNet18Encoder._make_proj(1280, embed_dim)
        self.norm_mode  = norm
        self.in_ch      = in_ch
        self.embed_dim  = embed_dim

    def forward(self, x):
        feat = self.backbone(x)                      # (B, 1280, 1, 1) or (B,1280)
        if feat.dim() == 4: feat = feat.flatten(1)
        z = self.proj(feat)
        return F.normalize(z, dim=-1) if self.norm_mode == "l2" else z


class MobileNetV3Encoder(BaseEncoder):
    def __init__(self, in_ch: int, embed_dim: int = 256,
                 pretrained: bool = True, norm: str = "l2"):
        super().__init__()
        base = models.mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        old  = base.features[0][0]
        base.features[0][0] = nn.Conv2d(
            in_ch, old.out_channels, kernel_size=old.kernel_size,
            stride=old.stride, padding=old.padding, bias=False)
        if pretrained and in_ch != 3:
            with torch.no_grad():
                mw = old.weight.mean(dim=1, keepdim=True)
                base.features[0][0].weight.copy_(mw.repeat(1, in_ch, 1, 1))
        base.classifier = nn.Identity()
        self.backbone   = base
        self.proj       = ResNet18Encoder._make_proj(576, embed_dim)
        self.norm_mode  = norm
        self.in_ch      = in_ch
        self.embed_dim  = embed_dim

    def forward(self, x):
        feat = self.backbone(x)
        if feat.dim() == 4: feat = feat.flatten(1)
        z = self.proj(feat)
        return F.normalize(z, dim=-1) if self.norm_mode == "l2" else z


def build_encoder(arch: str, in_ch: int, embed_dim: int,
                  pretrained: bool, norm: str) -> BaseEncoder:
    """Factory — builds any encoder variant."""
    kwargs = dict(in_ch=in_ch, embed_dim=embed_dim,
                  pretrained=pretrained, norm=norm)
    if arch == "resnet18":   return ResNet18Encoder(**kwargs)
    if arch == "resnet50":   return ResNet50Encoder(**kwargs)
    if arch == "effnet_b0":  return EfficientNetB0Encoder(**kwargs)
    if arch == "mobilenet":  return MobileNetV3Encoder(**kwargs)
    raise ValueError(f"Unknown arch: {arch}")


def build_proj_with_norm(in_feat: int, embed_dim: int, norm: str) -> nn.Sequential:
    """Build projection head with the specified output normalisation layer."""
    layers: List[nn.Module] = [
        nn.Linear(in_feat, embed_dim),
        nn.ReLU(inplace=True),
        nn.Linear(embed_dim, embed_dim),
    ]
    if norm == "bn":
        layers.append(nn.BatchNorm1d(embed_dim))
    elif norm == "ln":
        layers.append(nn.LayerNorm(embed_dim))
    return nn.Sequential(*layers)


# ============================================================
#  LOAD / COPY IVISIO PRETRAINED BACKBONE
# ============================================================
def load_ivisio_encoder(ckpt_path: str, in_ch: int,
                        embed_dim: int = 256) -> ResNet18Encoder:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"iVision checkpoint not found: {ckpt_path}")
    ckpt    = torch.load(ckpt_path, map_location="cpu")
    encoder = ResNet18Encoder(in_ch=in_ch, embed_dim=embed_dim, pretrained=False)
    encoder.load_state_dict(ckpt["state_dict"], strict=True)
    acc = float(ckpt.get("acc", 0.0))
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad_(False)
    print(f"  [iVision] Loaded backbone  acc={acc:.3f}")
    return encoder


def copy_ivisio_weights(iv_encoder: ResNet18Encoder,
                        tgt_encoder: BaseEncoder) -> None:
    """
    Copy all compatible weight tensors from iv_encoder into tgt_encoder.
    Used when a variant shares the same architecture as the baseline.
    """
    src_sd = iv_encoder.state_dict()
    tgt_sd = tgt_encoder.state_dict()
    for k in tgt_sd:
        if k not in src_sd: continue
        if tgt_sd[k].shape == src_sd[k].shape:
            tgt_sd[k].copy_(src_sd[k])
        elif "conv1.weight" in k:
            mw = src_sd[k].mean(dim=1, keepdim=True)
            tgt_sd[k].copy_(mw.repeat(1, tgt_encoder.in_ch, 1, 1))
    tgt_encoder.load_state_dict(tgt_sd)


# ============================================================
#  EPISODIC DATASETS & SAMPLERS  (same as v4)
# ============================================================
class LazyLRUCubeDataset:
    def __init__(self, items, label_to_int, keep_fp16, max_cache_items,
                 band_fn=None, raw_in_ch=None):
        self.items           = items
        self.label_to_int    = dict(label_to_int)
        self.keep_fp16       = keep_fp16
        self.max_cache_items = int(max_cache_items)
        self.band_fn         = band_fn        # optional band preprocessing
        self.raw_in_ch       = raw_in_ch
        self.y               = [self.label_to_int[str(x["label"])] for x in items]
        self.class_to_indices: Dict[int, List[int]] = {}
        for i, yi in enumerate(self.y):
            self.class_to_indices.setdefault(yi, []).append(i)
        self._cache          = collections.OrderedDict()
        t0 = self._load(0)
        self.in_ch = int(t0.shape[0])
        self._cache[0] = t0

    def _load(self, idx: int) -> torch.Tensor:
        t = _load_cube_lazy(self.items[idx]["path"], keep_fp16=False)
        if self.band_fn is not None:
            t = self.band_fn(t, self.raw_in_ch)
        return t.half() if self.keep_fp16 else t

    def __len__(self): return len(self.items)

    def get_x(self, idx: int) -> torch.Tensor:
        if idx in self._cache:
            v = self._cache.pop(idx); self._cache[idx] = v; return v
        t = self._load(idx)
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



class EpisodeSampler:
    def __init__(self, class_to_indices, seed=0):
        self.rng     = random.Random(seed)
        self.c2i     = class_to_indices
        self.classes = sorted(class_to_indices.keys())

    def eligible(self, k, q):
        return [c for c in self.classes if len(self.c2i[c]) >= k+q]

    def sample(self, n_way, k_shot, q_query, split2=None):
        """split2: optional second split (test split for cross-eval)."""
        if split2 is None:
            pool = self.eligible(k_shot, q_query)
            n    = min(n_way, len(pool))
            chosen = self.rng.sample(pool, n)
            s_idx, q_idx, y_s, y_q = [], [], [], []
            for ny, cls in enumerate(chosen):
                picks  = self.rng.sample(self.c2i[cls], k_shot + q_query)
                s_idx += picks[:k_shot]; q_idx += picks[k_shot:]
                y_s   += [ny]*k_shot;   y_q   += [ny]*q_query
        else:
            common = [c for c in self.classes
                      if c in split2.c2i
                      and len(self.c2i[c]) >= k_shot
                      and len(split2.c2i[c]) >= q_query]
            n      = min(n_way, len(common))
            chosen = self.rng.sample(common, n)
            s_idx, q_idx, y_s, y_q = [], [], [], []
            for ny, cls in enumerate(chosen):
                s_idx += self.rng.sample(self.c2i[cls], k_shot)
                q_idx += self.rng.sample(split2.c2i[cls], q_query)
                y_s   += [ny]*k_shot; y_q += [ny]*q_query
        return n, s_idx, q_idx, torch.tensor(y_s), torch.tensor(y_q)


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
#  LOSS VARIANTS  (Ablation E)
# ============================================================
def proto_loss_sqL2(z_s, y_s, z_q, y_q, n_way):
    """E1 — standard prototypical (negative squared-L2)."""
    protos = torch.stack([z_s[y_s==c].mean(0) for c in range(n_way)])
    logits = -((z_q.unsqueeze(1) - protos.unsqueeze(0))**2).sum(-1)
    return F.cross_entropy(logits, y_q), (logits.argmax(1)==y_q).float().mean().item()

def proto_loss_cosine(z_s, y_s, z_q, y_q, n_way):
    """E2 — prototypical with cosine similarity."""
    protos = F.normalize(
        torch.stack([z_s[y_s==c].mean(0) for c in range(n_way)]), dim=-1)
    zq_n   = F.normalize(z_q, dim=-1)
    logits = zq_n @ protos.T * 10.0   # temperature scale
    return F.cross_entropy(logits, y_q), (logits.argmax(1)==y_q).float().mean().item()

def supcon_loss(z_s, y_s, z_q, y_q, n_way, temperature=0.07):
    """E3 — Supervised Contrastive loss on the full episode batch."""
    z   = F.normalize(torch.cat([z_s, z_q], dim=0), dim=-1)
    y   = torch.cat([y_s, y_q], dim=0)
    N   = z.shape[0]
    sim = (z @ z.T) / temperature                            # (N, N)
    mask_pos  = (y.unsqueeze(0) == y.unsqueeze(1)).float()  # (N, N)
    mask_self = torch.eye(N, device=z.device)
    mask_pos  = mask_pos * (1 - mask_self)
    log_prob  = sim - torch.logsumexp(
        sim - 1e9 * mask_self, dim=1, keepdim=True)
    n_pos     = mask_pos.sum(1).clamp(min=1)
    loss      = -(mask_pos * log_prob).sum(1) / n_pos
    # Also compute prototypical accuracy for fair comparison
    protos = torch.stack([z_s[y_s==c].mean(0) for c in range(n_way)])
    logits = -((F.normalize(z_q,dim=-1).unsqueeze(1) -
                F.normalize(protos,dim=-1).unsqueeze(0))**2).sum(-1)
    acc    = (logits.argmax(1)==y_q).float().mean().item()
    return loss.mean(), acc

def triplet_loss(z_s, y_s, z_q, y_q, n_way, margin=0.3):
    """E4 — Batch-hard triplet loss."""
    z = torch.cat([z_s, z_q], dim=0)
    y = torch.cat([y_s, y_q], dim=0)
    dist = torch.cdist(z, z)           # (N, N)
    N    = z.shape[0]
    losses = []
    for i in range(N):
        pos_mask = (y == y[i]).clone().float(); pos_mask[i] = 0
        neg_mask = (y != y[i]).clone().float()
        if pos_mask.sum() == 0 or neg_mask.sum() == 0: continue
        d_ap = (dist[i] * pos_mask).max()     # hardest positive
        d_an = (dist[i] + 1e9*pos_mask + 1e9*torch.eye(N,device=z.device)[i]).min()
        losses.append(F.relu(d_ap - d_an + margin))
    if not losses:
        dummy = z.sum() * 0
        protos = torch.stack([z_s[y_s==c].mean(0) for c in range(n_way)])
        logits = -((z_q.unsqueeze(1)-protos.unsqueeze(0))**2).sum(-1)
        return dummy, (logits.argmax(1)==y_q).float().mean().item()
    loss = torch.stack(losses).mean()
    protos = torch.stack([z_s[y_s==c].mean(0) for c in range(n_way)])
    logits = -((z_q.unsqueeze(1)-protos.unsqueeze(0))**2).sum(-1)
    acc    = (logits.argmax(1)==y_q).float().mean().item()
    return loss, acc


LOSS_FNS = {
    "proto_sqL2":   proto_loss_sqL2,
    "proto_cosine": proto_loss_cosine,
    "supcon":       supcon_loss,
    "triplet":      triplet_loss,
}


# ============================================================
#  PROXY TRAINING LOOP  (used for conditions that need training)
# ============================================================
def proxy_train(encoder: BaseEncoder,
                train_ds: LazyLRUCubeDataset,
                n_way: int, k_shot: int, q_query: int,
                n_episodes: int, lr: float, wd: float,
                loss_fn_name: str, device: torch.device) -> float:
    """
    Lightweight proxy training: run n_episodes of episodic training.
    Returns final training accuracy (not test accuracy).
    Used for ablation conditions that cannot use the pretrained checkpoint.
    """
    encoder  = encoder.to(device).train()
    opt      = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=wd)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_episodes)
    sampler  = EpisodeSampler(train_ds.class_to_indices, seed=cfg.seed)
    loss_fn  = LOSS_FNS[loss_fn_name]
    accs     = []

    for epi in range(1, n_episodes + 1):
        n, s_idx, q_idx, y_s, y_q = sampler.sample(n_way, k_shot, q_query)
        xs = torch.stack(pad_to_same_size([train_ds.get_x(i) for i in s_idx])).to(device).float()
        xq = torch.stack(pad_to_same_size([train_ds.get_x(i) for i in q_idx])).to(device).float()
        y_s = y_s.to(device); y_q = y_q.to(device)
        z_s = encoder(xs); z_q = encoder(xq)
        loss, acc = loss_fn(z_s, y_s, z_q, y_q, n)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        accs.append(acc)
        if epi % 100 == 0:
            print(f"    proxy [{epi}/{n_episodes}] acc={np.mean(accs[-50:]):.3f}")

    encoder.eval()
    return float(np.mean(accs[-100:]))


# ============================================================
#  EPISODIC EVALUATION  (clean test episodes, no grad)
# ============================================================
@torch.no_grad()
def episodic_eval(encoder: BaseEncoder,
                  train_ds: LazyLRUCubeDataset,
                  test_ds:  LazyLRUCubeDataset,
                  n_way: int, k_shot: int, q_query: int,
                  n_episodes: int, device: torch.device) -> Tuple[float, float]:
    encoder.eval()
    tr_samp = EpisodeSampler(train_ds.class_to_indices, seed=cfg.seed+1)
    te_samp = EpisodeSampler(test_ds.class_to_indices,  seed=cfg.seed+2)
    accs = []
    for _ in range(n_episodes):
        n, s_idx, q_idx, y_s, y_q = tr_samp.sample(
            n_way, k_shot, q_query, split2=te_samp)
        xs = torch.stack(pad_to_same_size([train_ds.get_x(i) for i in s_idx])).to(device).float()
        xq = torch.stack(pad_to_same_size([test_ds.get_x(i)  for i in q_idx])).to(device).float()
        y_s = y_s.to(device); y_q = y_q.to(device)
        z_s = encoder(xs); z_q = encoder(xq)
        protos = torch.stack([z_s[y_s==c].mean(0) for c in range(n)])
        dists  = ((z_q.unsqueeze(1) - protos.unsqueeze(0))**2).sum(-1)
        accs.append((dists.argmin(1)==y_q).float().mean().item())
    return float(np.mean(accs)), compute_ci95(accs)


# ============================================================
#  WRITER ID — prototype nearest-neighbour  (single eval pass)
# ============================================================
def eval_writer_id(df: pd.DataFrame,
                   encoder: BaseEncoder,
                   enc_in_ch: int,
                   device: torch.device,
                   band_fn=None) -> Dict[str, float]:
    encoder = encoder.to(device).eval()
    writers = sorted(df["writer_id"].unique())
    w2i     = {w:i for i,w in enumerate(writers)}

    def embed(path):
        cube = _CACHE.get(str(path)).float()
        if band_fn is not None:
            cube = band_fn(cube, cube.shape[0])
        C = cube.shape[0]
        if C < enc_in_ch: cube = F.pad(cube, (0,0,0,0,0,enc_in_ch-C))
        elif C > enc_in_ch: cube = cube[:enc_in_ch]
        with torch.no_grad():
            z = encoder(cube.unsqueeze(0).to(device))
        return F.normalize(z.squeeze(0), dim=-1).cpu()

    # Leave-one-page-out split
    rng = random.Random(cfg.seed)
    tr_rows, te_rows = [], []
    for _, g in df.groupby("writer_id"):
        idxs = list(g.index)
        if len(idxs) == 1: tr_rows.extend(idxs); continue
        pick = rng.choice(idxs); te_rows.append(pick)
        tr_rows.extend(i for i in idxs if i != pick)
    train_df = df.loc[tr_rows]; test_df = df.loc[te_rows]

    protos = {}
    for wid, grp in train_df.groupby("writer_id"):
        zs = [embed(row["full_cube"]) for _, row in grp.iterrows()]
        protos[wid] = torch.stack(zs).mean(0)
    proto_keys   = sorted(protos.keys())
    proto_matrix = torch.stack([protos[k] for k in proto_keys])

    yt, yp = [], []
    for _, row in test_df.iterrows():
        z     = embed(row["full_cube"]).unsqueeze(0)
        dists = ((z.unsqueeze(1) - proto_matrix.unsqueeze(0))**2).sum(-1).squeeze(0)
        pred  = proto_keys[int(dists.argmin().item())]
        yt.append(w2i[int(row["writer_id"])]); yp.append(w2i[pred])

    yt = np.array(yt); yp = np.array(yp)
    acc = float((yt == yp).mean())
    mcc = float(matthews_corrcoef(yt, yp)) if len(set(yt)) > 1 else 0.0
    return {"writer_id_acc": acc, "writer_id_mcc": mcc}


# ============================================================
#  INK MISMATCH AUC  (cosine similarity, no fine-tune)
# ============================================================
def eval_ink_mismatch(df: pd.DataFrame,
                      encoder: BaseEncoder,
                      enc_in_ch: int,
                      device: torch.device,
                      band_fn=None) -> Dict[str, float]:
    encoder = encoder.to(device).eval()
    rng     = random.Random(cfg.seed + 2)
    by_w    = {w: list(g.index) for w, g in df.groupby("writer_id")}
    writers = list(by_w.keys())
    n_pairs = min(cfg.verif_pairs, len(writers) * 20)

    def embed(path, seed_i):
        cube = _CACHE.get(str(path)).float()
        if band_fn is not None:
            cube = band_fn(cube, cube.shape[0])
        C = cube.shape[0]
        if C < enc_in_ch: cube = F.pad(cube, (0,0,0,0,0,enc_in_ch-C))
        elif C > enc_in_ch: cube = cube[:enc_in_ch]
        with torch.no_grad():
            z = encoder(cube.unsqueeze(0).to(device))
        return F.normalize(z.squeeze(0), dim=-1).cpu()

    sims, labels = [], []
    for _ in range(n_pairs):
        w1      = rng.choice(writers)
        genuine = rng.random() < 0.5 and len(by_w[w1]) >= 2
        if genuine: i1,i2 = rng.sample(by_w[w1],2); lbl = 0
        else:
            w2 = rng.choice([w for w in writers if w != w1])
            i1 = rng.choice(by_w[w1]); i2 = rng.choice(by_w[w2]); lbl = 1
        z1 = embed(df.iloc[i1]["full_cube"], i1)
        z2 = embed(df.iloc[i2]["full_cube"], i2)
        sims.append(F.cosine_similarity(z1.unsqueeze(0),
                                         z2.unsqueeze(0)).item())
        labels.append(lbl)

    sims   = np.array(sims); labels = np.array(labels)
    scores = 1.0 - sims
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc     = auc(fpr, tpr)
    fnr         = 1 - tpr
    eer_idx     = np.argmin(np.abs(fpr - fnr))
    eer         = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    return {"ink_mismatch_auc": roc_auc, "ink_mismatch_eer": eer}


# ============================================================
#  ABLATION CONDITION DEFINITION
# ============================================================
@dataclass
class AblationCondition:
    axis:        str          # "A", "B", ...
    condition:   str          # "A1", "A2", ...
    label:       str          # human-readable
    arch:        str  = "resnet18"
    embed_dim:   int  = 256
    band_key:    str  = "full"  # key into BAND_STRATEGIES
    init:        str  = "ivisio_transfer"  # ivisio_transfer|imagenet|random|ivisio_frozen
    loss_fn:     str  = "proto_sqL2"
    k_shot:      int  = 1
    norm:        str  = "l2"
    needs_train: bool = False   # True → proxy training required


def build_conditions() -> List[AblationCondition]:
    C = AblationCondition
    conditions = [
        # ── A: Backbone Architecture ─────────────────────────
        C("A","A1","ResNet18 (baseline)",     arch="resnet18",  init="ivisio_transfer"),
        C("A","A2","ResNet50",                arch="resnet50",  init="imagenet",    needs_train=True),
        C("A","A3","EfficientNet-B0",         arch="effnet_b0", init="imagenet",    needs_train=True),
        C("A","A4","MobileNetV3-Small",       arch="mobilenet", init="imagenet",    needs_train=True),

        # ── B: Spectral Band Strategy ────────────────────────
        C("B","B1","Full bands (baseline)",   band_key="full",      init="ivisio_transfer"),
        C("B","B2","PCA → 3 bands",           band_key="pca3",      init="imagenet",    needs_train=True),
        C("B","B3","Uniform 3 bands",         band_key="uniform3",  init="imagenet",    needs_train=True),
        C("B","B4","Random 16 bands",         band_key="random16",  init="ivisio_transfer"),

        # ── C: Embedding Dimension ───────────────────────────
        C("C","C1","embed_dim=64",            embed_dim=64,  needs_train=True),
        C("C","C2","embed_dim=128",           embed_dim=128, needs_train=True),
        C("C","C3","embed_dim=256 (baseline)",embed_dim=256, init="ivisio_transfer"),
        C("C","C4","embed_dim=512",           embed_dim=512, needs_train=True),

        # ── D: Weight Initialisation Strategy ───────────────
        C("D","D1","iVision transfer (baseline)",  init="ivisio_transfer"),
        C("D","D2","ImageNet pretrained only",      init="imagenet",    needs_train=True),
        C("D","D3","Random init (no pretrain)",     init="random",      needs_train=True),
        C("D","D4","iVision frozen backbone",       init="ivisio_frozen"),

        # ── E: Episodic Training Loss ────────────────────────
        C("E","E1","Proto sqL2 (baseline)",   loss_fn="proto_sqL2",   init="ivisio_transfer"),
        C("E","E2","Proto cosine",            loss_fn="proto_cosine", init="ivisio_transfer", needs_train=True),
        C("E","E3","Supervised Contrastive",  loss_fn="supcon",       init="ivisio_transfer", needs_train=True),
        C("E","E4","Batch-hard Triplet",      loss_fn="triplet",      init="ivisio_transfer", needs_train=True),

        # ── F: K-shot ────────────────────────────────────────
        C("F","F1","k=1 (baseline)",  k_shot=1),
        C("F","F2","k=2",             k_shot=2),
        C("F","F3","k=3",             k_shot=3),
        C("F","F4","k=5",             k_shot=5),

        # ── G: Embedding Normalisation ───────────────────────
        C("G","G1","L2-norm (baseline)", norm="l2",   init="ivisio_transfer"),
        C("G","G2","No normalisation",   norm="none",  needs_train=True),
        C("G","G3","BatchNorm1d",         norm="bn",    needs_train=True),
        C("G","G4","LayerNorm",           norm="ln",    needs_train=True),
    ]
    return conditions


# ============================================================
#  RUN ONE ABLATION CONDITION
# ============================================================
def run_condition(cond: AblationCondition,
                  df: pd.DataFrame,
                  iv_encoder: ResNet18Encoder,
                  train_items, test_items,
                  raw_in_ch: int,
                  device: torch.device) -> Dict[str, float]:
    """
    Evaluates one ablation condition.  Returns metric dict.
    """
    set_seed(cfg.seed)
    print(f"\n  [{cond.condition}] {cond.label}")

    # ── Determine encoder input channels ──────────────────────
    band_fn, band_out_ch = BAND_STRATEGIES[cond.band_key]
    enc_in_ch = band_out_ch if band_out_ch is not None else raw_in_ch
    if enc_in_ch is None: enc_in_ch = raw_in_ch

    # ── Build encoder ─────────────────────────────────────────
    use_pretrained_imagenet = cond.init in ("imagenet",)
    encoder = build_encoder(cond.arch, enc_in_ch, cond.embed_dim,
                             pretrained=use_pretrained_imagenet,
                             norm=cond.norm)

    # ── Rebuild projection head with correct norm if needed ───
    if cond.norm in ("bn", "ln") and hasattr(encoder, "proj"):
        in_feat = {"resnet18": 512, "resnet50": 2048,
                   "effnet_b0": 1280, "mobilenet": 576}[cond.arch]
        encoder.proj = build_proj_with_norm(in_feat, cond.embed_dim, cond.norm)

    # ── Apply initialisation strategy ─────────────────────────
    if cond.init == "ivisio_transfer" and cond.arch == "resnet18":
        copy_ivisio_weights(iv_encoder, encoder)
        for p in encoder.parameters(): p.requires_grad_(True)
        print(f"    init: iVision transfer weights")

    elif cond.init == "ivisio_frozen" and cond.arch == "resnet18":
        copy_ivisio_weights(iv_encoder, encoder)
        for p in encoder.backbone.parameters(): p.requires_grad_(False)
        for p in encoder.proj.parameters():     p.requires_grad_(True)
        print(f"    init: iVision frozen backbone")

    elif cond.init == "random":
        for p in encoder.parameters(): p.requires_grad_(True)
        print(f"    init: random (no pretrained weights)")

    elif cond.init == "imagenet":
        for p in encoder.parameters(): p.requires_grad_(True)
        print(f"    init: ImageNet pretrained conv1 adapted")

    # ── Build datasets with band preprocessing ─────────────────
    all_labels   = sorted(set(x["label"] for x in train_items + test_items))
    label_to_int = {c:i for i,c in enumerate(all_labels)}

    train_ds = LazyLRUCubeDataset(train_items, label_to_int,
                                   cfg.fp16_cache, cfg.max_cache,
                                   band_fn=band_fn if band_out_ch else None,
                                   raw_in_ch=raw_in_ch)
    test_ds  = LazyLRUCubeDataset(test_items,  label_to_int,
                                   cfg.fp16_cache, cfg.max_cache,
                                   band_fn=band_fn if band_out_ch else None,
                                   raw_in_ch=raw_in_ch)

    # ── Proxy train if this condition cannot use saved ckpt ────
    proxy_tr_acc = None
    if cond.needs_train:
        print(f"    proxy training ({cfg.proxy_episodes} episodes)…")
        proxy_tr_acc = proxy_train(
            encoder, train_ds,
            n_way=min(cfg.n_way, len(all_labels)-1),
            k_shot=cond.k_shot,
            q_query=cfg.train_q_query,
            n_episodes=cfg.proxy_episodes,
            lr=cfg.proxy_lr, wd=cfg.proxy_wd,
            loss_fn_name=cond.loss_fn,
            device=device)
        print(f"    proxy tr_acc = {proxy_tr_acc:.3f}")
    else:
        encoder = encoder.to(device).eval()

    # ── Episodic eval (test set) ───────────────────────────────
    ep_acc, ep_ci = episodic_eval(
        encoder, train_ds, test_ds,
        n_way=min(cfg.n_way, len(all_labels)-1),
        k_shot=cond.k_shot,
        q_query=cfg.eval_q_query,
        n_episodes=cfg.test_episodes,
        device=device)
    print(f"    episodic_acc = {ep_acc:.4f} ±{ep_ci:.4f}")

    # ── Writer ID ─────────────────────────────────────────────
    wid_metrics = eval_writer_id(
        df, encoder, enc_in_ch, device,
        band_fn=(band_fn if band_out_ch else None))
    print(f"    writer_id_acc = {wid_metrics['writer_id_acc']:.4f}  "
          f"MCC = {wid_metrics['writer_id_mcc']:.4f}")

    # ── Ink Mismatch ──────────────────────────────────────────
    ink_metrics = eval_ink_mismatch(
        df, encoder, enc_in_ch, device,
        band_fn=(band_fn if band_out_ch else None))
    print(f"    ink_mismatch_auc = {ink_metrics['ink_mismatch_auc']:.4f}  "
          f"EER = {ink_metrics['ink_mismatch_eer']:.4f}")

    result = {
        "episodic_acc":      ep_acc,
        "episodic_ci95":     ep_ci,
        "writer_id_acc":     wid_metrics["writer_id_acc"],
        "writer_id_mcc":     wid_metrics["writer_id_mcc"],
        "ink_mismatch_auc":  ink_metrics["ink_mismatch_auc"],
        "ink_mismatch_eer":  ink_metrics["ink_mismatch_eer"],
    }
    if proxy_tr_acc is not None:
        result["proxy_tr_acc"] = proxy_tr_acc
    return result


# ============================================================
#  PLOTTING
# ============================================================
AXIS_LABELS = {
    "A": "Backbone Architecture",
    "B": "Spectral Band Strategy",
    "C": "Embedding Dimension",
    "D": "Init Strategy",
    "E": "Training Loss",
    "F": "K-shot",
    "G": "Embedding Normalisation",
}

METRIC_COLS = ["episodic_acc", "writer_id_acc", "writer_id_mcc",
               "ink_mismatch_auc", "ink_mismatch_eer"]
METRIC_LABELS = {
    "episodic_acc":     "Episodic Acc",
    "writer_id_acc":    "Writer ID Acc",
    "writer_id_mcc":    "Writer ID MCC",
    "ink_mismatch_auc": "Ink Mismatch AUC",
    "ink_mismatch_eer": "Ink Mismatch EER (↓)",
}
METRIC_COLORS = {
    "episodic_acc":     "#1565C0",
    "writer_id_acc":    "#2E7D32",
    "writer_id_mcc":    "#558B2F",
    "ink_mismatch_auc": "#E65100",
    "ink_mismatch_eer": "#B71C1C",
}


def plot_axis(axis: str, rows: pd.DataFrame, out_dir: str):
    """Grouped bar chart for all metrics in one axis."""
    axis_rows = rows[rows["axis"] == axis].copy()
    conds     = list(axis_rows["condition"])
    labels    = [r.split("(")[0].strip() for r in axis_rows["label"]]

    metrics   = [m for m in METRIC_COLS if m in axis_rows.columns]
    n_conds   = len(conds)
    n_met     = len(metrics)
    x         = np.arange(n_conds)
    w         = 0.8 / n_met

    fig, ax = plt.subplots(figsize=(max(8, n_conds*1.8), 5))
    for j, met in enumerate(metrics):
        vals = axis_rows[met].fillna(0).values
        bars = ax.bar(x + j*w - (n_met-1)*w/2, vals, w,
                      label=METRIC_LABELS[met],
                      color=METRIC_COLORS[met], alpha=0.82)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)

    # Highlight baseline (first condition in each axis)
    ax.axvspan(-0.5, 0.5, alpha=0.07, color="gold", zorder=0, label="Baseline")

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score"); ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Ablation {axis}: {AXIS_LABELS[axis]}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"ablation_{axis}_{AXIS_LABELS[axis].replace(' ','_')}.png"))


def plot_heatmap_all(rows: pd.DataFrame, out_dir: str):
    """Heatmap: conditions (rows) × metrics (columns)."""
    met_cols = [m for m in METRIC_COLS if m in rows.columns]
    data     = rows[met_cols].fillna(0).values
    # Invert EER for coloring (lower is better)
    eer_col  = met_cols.index("ink_mismatch_eer") if "ink_mismatch_eer" in met_cols else -1
    data_disp = data.copy()
    if eer_col >= 0: data_disp[:, eer_col] = 1 - data_disp[:, eer_col]

    row_labels = [f"{r['condition']} — {r['label'][:28]}" for _, r in rows.iterrows()]
    col_labels = [METRIC_LABELS[m] for m in met_cols]

    fig, ax = plt.subplots(figsize=(len(met_cols)*2+2, max(10, len(rows)*0.35)))
    im = ax.imshow(data_disp, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(met_cols))); ax.set_xticklabels(col_labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=7)
    for i in range(len(row_labels)):
        for j in range(len(met_cols)):
            v = data[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=6.5, color="black" if 0.3 < data_disp[i,j] < 0.8 else "white")
    fig.colorbar(im, ax=ax, fraction=0.02, label="Score (EER inverted)")
    ax.set_title("Ablation Heatmap — All Conditions × All Metrics",
                 fontsize=12, fontweight="bold", pad=12)

    # Draw axis group separators
    axis_breaks = rows.groupby("axis").apply(lambda g: g.index[-1] - rows.index[0]).values
    for b in axis_breaks[:-1]:
        ax.axhline(b + 0.5, color="white", lw=2)

    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, "ablation_heatmap_all.png"))


def plot_delta_from_baseline(rows: pd.DataFrame, out_dir: str):
    """
    For each condition, show delta from its axis baseline.
    Positive = improvement. Negative = degradation.
    """
    met_cols   = ["writer_id_acc", "ink_mismatch_auc", "episodic_acc"]
    met_cols   = [m for m in met_cols if m in rows.columns]
    deltas_all = []
    for axis in rows["axis"].unique():
        ax_rows  = rows[rows["axis"] == axis]
        baseline = ax_rows.iloc[0]
        for _, row in ax_rows.iterrows():
            for met in met_cols:
                delta = row[met] - baseline[met]
                deltas_all.append({
                    "condition": row["condition"],
                    "label":     row["label"],
                    "axis":      row["axis"],
                    "metric":    METRIC_LABELS[met],
                    "delta":     delta,
                })
    ddf = pd.DataFrame(deltas_all)

    axes_list = sorted(ddf["axis"].unique())
    fig, axs  = plt.subplots(1, len(axes_list),
                              figsize=(len(axes_list)*3, 7), sharey=False)
    if len(axes_list) == 1: axs = [axs]
    colors_met = ["#1565C0", "#E65100", "#2E7D32"]

    for ax_plot, axis in zip(axs, axes_list):
        sub = ddf[ddf["axis"] == axis]
        conds = sub["condition"].unique()
        x = np.arange(len(conds))
        for j, met in enumerate(sub["metric"].unique()):
            mrows = sub[sub["metric"] == met]
            vals  = [mrows[mrows["condition"]==c]["delta"].values[0]
                     if len(mrows[mrows["condition"]==c]) > 0 else 0 for c in conds]
            ax_plot.bar(x + j*0.25 - 0.25, vals, 0.22,
                        label=met, color=colors_met[j % len(colors_met)], alpha=0.85)
        ax_plot.axhline(0, color="black", lw=1)
        ax_plot.set_xticks(x); ax_plot.set_xticklabels(conds, rotation=40, ha="right", fontsize=8)
        ax_plot.set_title(f"Axis {axis}", fontsize=10, fontweight="bold")
        ax_plot.set_ylabel("Delta from Baseline")
        ax_plot.grid(axis="y", alpha=0.3)
        if ax_plot == axs[0]: ax_plot.legend(fontsize=7)

    plt.suptitle("Delta from Baseline per Axis", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, "ablation_delta_from_baseline.png"))


def plot_radar_full_vs_best(rows: pd.DataFrame, out_dir: str):
    """Radar: full model baseline vs best-per-axis variant."""
    met_cols = ["episodic_acc", "writer_id_acc",
                "writer_id_mcc", "ink_mismatch_auc"]
    met_cols = [m for m in met_cols if m in rows.columns]

    # Baseline = first condition across all (A1 / the full model)
    baseline_row = rows.iloc[0]

    # Best per axis by writer_id_acc
    best_rows = []
    for axis in rows["axis"].unique():
        ax_rows   = rows[rows["axis"] == axis]
        best_idx  = ax_rows["writer_id_acc"].idxmax()
        best_rows.append(ax_rows.loc[best_idx])

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
    N = len(met_cols)
    angles = [n/N*2*math.pi for n in range(N)] + [0]

    def radar_vals(row):
        return [float(row[m]) if m in row.index else 0.0 for m in met_cols] + \
               [float(row[met_cols[0]]) if met_cols[0] in row.index else 0.0]

    ax.set_theta_offset(math.pi/2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS[m] for m in met_cols], size=9)
    ax.set_ylim(0, 1)

    # Baseline
    bv = radar_vals(baseline_row)
    ax.plot(angles, bv, "o-", lw=2, color="#1565C0", label=f"Baseline ({baseline_row['condition']})")
    ax.fill(angles, bv, alpha=0.15, color="#1565C0")

    # Best per axis
    cmap = plt.cm.get_cmap("Set2", len(best_rows))
    for i, row in enumerate(best_rows):
        bv = radar_vals(row)
        ax.plot(angles, bv, "s--", lw=1.5, color=cmap(i),
                label=f"Best-{row['axis']}: {row['condition']}")
        ax.fill(angles, bv, alpha=0.05, color=cmap(i))

    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.35, 1.1))
    ax.set_title("Full Model vs Best-per-Axis", size=12, pad=20, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, "ablation_radar_full_vs_best.png"))


def plot_sensitivity(rows: pd.DataFrame, out_dir: str):
    """
    Bar chart showing sensitivity of each axis: range(max-min) across conditions.
    Higher = that design choice matters more.
    """
    met_cols = ["writer_id_acc", "ink_mismatch_auc", "episodic_acc"]
    met_cols = [m for m in met_cols if m in rows.columns]
    axes_list = sorted(rows["axis"].unique())
    ranges = {m: [] for m in met_cols}
    for axis in axes_list:
        ax_rows = rows[rows["axis"] == axis]
        for met in met_cols:
            r = float(ax_rows[met].max() - ax_rows[met].min())
            ranges[met].append(r)

    x   = np.arange(len(axes_list))
    w   = 0.8 / len(met_cols)
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, met in enumerate(met_cols):
        bars = ax.bar(x + j*w - (len(met_cols)-1)*w/2,
                      ranges[met], w,
                      label=METRIC_LABELS[met],
                      color=METRIC_COLORS[met], alpha=0.85)
        for bar, v in zip(bars, ranges[met]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}: {AXIS_LABELS[a]}" for a in axes_list],
                       rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Range (max − min) across conditions")
    ax.set_title("Ablation Sensitivity — Which Design Choice Matters Most",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, "ablation_sensitivity.png"))


# ============================================================
#  RESULTS TABLE
# ============================================================
def save_results_table(rows: pd.DataFrame, out_dir: str, dataset_tag: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── CSV ────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, f"ablation_table_{dataset_tag}.csv")
    rows.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  [saved] {csv_path}")

    # ── JSON ───────────────────────────────────────────────────
    json_path = os.path.join(out_dir, f"ablation_summary_{dataset_tag}.json")
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": ts,
            "dataset":   dataset_tag,
            "conditions": rows.to_dict(orient="records"),
        }, f, indent=2)
    print(f"  [saved] {json_path}")

    # ── Formatted text table ───────────────────────────────────
    met_cols = [m for m in METRIC_COLS if m in rows.columns]
    col_w    = 28
    txt_path = os.path.join(out_dir, f"ablation_table_{dataset_tag}.txt")
    with open(txt_path, "w") as f:
        f.write("="*100 + "\n")
        f.write(f"  HYPERSPECTRAL FORENSICS — ABLATION STUDY  [{dataset_tag.upper()}]\n")
        f.write(f"  Generated : {ts}\n")
        f.write("="*100 + "\n\n")

        header = f"  {'Cond':<6} {'Label':<{col_w}}"
        for m in met_cols:
            header += f"  {METRIC_LABELS[m]:>18}"
        f.write(header + "\n")
        f.write("  " + "-"*(6 + col_w + 20*len(met_cols)) + "\n")

        current_axis = None
        for _, row in rows.iterrows():
            if row["axis"] != current_axis:
                current_axis = row["axis"]
                f.write(f"\n  ── Axis {current_axis}: {AXIS_LABELS[current_axis]} ──\n")
            line = f"  {row['condition']:<6} {row['label'][:col_w]:<{col_w}}"
            for m in met_cols:
                v = row[m] if m in row.index and pd.notna(row[m]) else float("nan")
                marker = "*" if row.get("is_baseline", False) else " "
                line += f"  {v:>17.4f}{marker}"
            f.write(line + "\n")

        # Best per axis
        f.write("\n\n  BEST PER AXIS (by Writer ID Acc)\n")
        f.write("  " + "-"*60 + "\n")
        for axis in sorted(rows["axis"].unique()):
            ax_rows  = rows[rows["axis"] == axis]
            best_idx = ax_rows["writer_id_acc"].idxmax()
            best     = ax_rows.loc[best_idx]
            f.write(f"  Axis {axis} ({AXIS_LABELS[axis]:<30}) "
                    f"→ {best['condition']:5}  {best['label'][:28]:<28}  "
                    f"Writer-ID={best['writer_id_acc']:.4f}  "
                    f"AUC={best['ink_mismatch_auc']:.4f}\n")

        f.write("\n" + "="*100 + "\n")
    print(f"  [saved] {txt_path}")


# ============================================================
#  MAIN ABLATION RUNNER
# ============================================================
def run_ablation(df: pd.DataFrame,
                 iv_encoder: ResNet18Encoder,
                 raw_in_ch: int,
                 dataset_tag: str,
                 active_axes: Optional[List[str]],
                 out_dir: str) -> pd.DataFrame:

    device       = torch.device(cfg.device)
    conditions   = build_conditions()
    if active_axes:
        conditions = [c for c in conditions if c.axis in active_axes]

    train_items, test_items = build_crosssplit_items(
        cfg.ivisio_index if dataset_tag == "ivisio" else cfg.uwa_index,
        cfg.ivisio_dir   if dataset_tag == "ivisio" else cfg.uwa_dir,
        seed=cfg.seed, test_per_class=1)

    records = []
    n_total = len(conditions)

    for idx, cond in enumerate(conditions):
        print(f"\n{'='*65}")
        print(f"  Condition {idx+1}/{n_total}  [{cond.condition}] {cond.label}")
        print(f"  Dataset={dataset_tag}  arch={cond.arch}  init={cond.init}"
              f"  bands={cond.band_key}  dim={cond.embed_dim}"
              f"  loss={cond.loss_fn}  k={cond.k_shot}  norm={cond.norm}")
        print(f"{'='*65}")

        t0 = time.perf_counter()
        try:
            metrics = run_condition(
                cond, df, iv_encoder,
                train_items, test_items, raw_in_ch, device)
            metrics["status"] = "ok"
        except Exception as e:
            print(f"  [ERROR] {cond.condition} failed: {e}")
            metrics = {m: float("nan") for m in METRIC_COLS}
            metrics["status"] = f"error: {e}"

        elapsed = time.perf_counter() - t0
        record  = {
            "axis":        cond.axis,
            "condition":   cond.condition,
            "label":       cond.label,
            "arch":        cond.arch,
            "embed_dim":   cond.embed_dim,
            "band_key":    cond.band_key,
            "init":        cond.init,
            "loss_fn":     cond.loss_fn,
            "k_shot":      cond.k_shot,
            "norm":        cond.norm,
            "needs_train": cond.needs_train,
            "elapsed_s":   round(elapsed, 1),
            **metrics,
        }
        records.append(record)
        print(f"  Done in {elapsed:.0f}s")

    rows = pd.DataFrame(records)

    # Mark baselines
    baseline_ids = {"A1","B1","C3","D1","E1","F1","G1"}
    rows["is_baseline"] = rows["condition"].isin(baseline_ids)

    return rows


# ============================================================
#  MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Hyperspectral Forensics — Ablation Study")
    parser.add_argument("--axis", default="all",
        help="Comma-separated axes to run, e.g. A,B,E  or  'all'")
    parser.add_argument("--dataset", default="ivisio",
        choices=["ivisio", "uwa", "both"],
        help="Dataset(s) to ablate on  [default: ivisio]")
    parser.add_argument("--quick", action="store_true",
        help="Reduce episodes for fast smoke-test")
    args = parser.parse_args()

    if args.quick:
        cfg.test_episodes  = 50
        cfg.proxy_episodes = 150
        cfg.verif_pairs    = 100
        print("  [quick mode] Reduced episodes for fast run")

    active_axes = None if args.axis == "all" else args.axis.upper().split(",")

    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)
    plot_dir = os.path.join(cfg.out_dir, "plots")
    ensure_dir(plot_dir)

    print("\n" + "="*65)
    print("  HYPERSPECTRAL FORENSICS — ABLATION STUDY")
    print("="*65)
    print(f"  Device        : {cfg.device}")
    print(f"  Axes          : {args.axis}")
    print(f"  Dataset(s)    : {args.dataset}")
    print(f"  Test episodes : {cfg.test_episodes}")
    print(f"  Proxy episodes: {cfg.proxy_episodes}")
    print(f"  iVision ckpt  : {cfg.ivisio_ckpt}")
    print("="*65 + "\n")

    # ── Load iVision backbone (baseline / transfer source) ─────
    df_iv    = load_index(cfg.ivisio_index, cfg.ivisio_dir, "ivisio")
    in_ch_iv = _CACHE.get(str(df_iv.iloc[0]["full_cube"])).shape[0]
    iv_enc   = load_ivisio_encoder(cfg.ivisio_ckpt, in_ch_iv)
    iv_enc.to(cfg.device)

    all_dataset_rows = []

    # ── iVision ablation ───────────────────────────────────────
    if args.dataset in ("ivisio", "both"):
        print("\n" + "#"*65)
        print("  ABLATION ON iVision HHID")
        print("#"*65)
        rows_iv = run_ablation(df_iv, iv_enc, in_ch_iv,
                               "ivisio", active_axes, cfg.out_dir)
        rows_iv["dataset"] = "ivisio"
        all_dataset_rows.append(rows_iv)

        print("\n  Generating iVision plots…")
        for axis in sorted(rows_iv["axis"].unique()):
            plot_axis(axis, rows_iv, plot_dir)
        plot_heatmap_all(rows_iv, plot_dir)
        plot_delta_from_baseline(rows_iv, plot_dir)
        plot_radar_full_vs_best(rows_iv, plot_dir)
        plot_sensitivity(rows_iv, plot_dir)
        save_results_table(rows_iv, cfg.out_dir, "ivisio")

    # ── UWA ablation ───────────────────────────────────────────
    if args.dataset in ("uwa", "both"):
        try:
            df_uwa    = load_index(cfg.uwa_index, cfg.uwa_dir, "uwa")
            in_ch_uwa = _CACHE.get(str(df_uwa.iloc[0]["full_cube"])).shape[0]
        except FileNotFoundError:
            print("  [warn] UWA index not found — skipping UWA ablation.")
            df_uwa = None

        if df_uwa is not None:
            print("\n" + "#"*65)
            print("  ABLATION ON UWA WIHSI")
            print("#"*65)
            rows_uwa = run_ablation(df_uwa, iv_enc, in_ch_uwa,
                                    "uwa", active_axes, cfg.out_dir)
            rows_uwa["dataset"] = "uwa"
            all_dataset_rows.append(rows_uwa)

            print("\n  Generating UWA plots…")
            for axis in sorted(rows_uwa["axis"].unique()):
                plot_axis(axis, rows_uwa, plot_dir)
            plot_heatmap_all(rows_uwa, plot_dir)
            plot_delta_from_baseline(rows_uwa, plot_dir)
            plot_radar_full_vs_best(rows_uwa, plot_dir)
            plot_sensitivity(rows_uwa, plot_dir)
            save_results_table(rows_uwa, cfg.out_dir, "uwa")

    # ── Combined cross-dataset comparison plot ─────────────────
    if len(all_dataset_rows) == 2:
        combined = pd.concat(all_dataset_rows, ignore_index=True)
        # Cross-dataset comparison: same condition, both datasets
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax_plot, met in zip(axes, ["writer_id_acc", "ink_mismatch_auc"]):
            for ds, color in [("ivisio","#1565C0"), ("uwa","#E65100")]:
                sub  = combined[combined["dataset"] == ds]
                cids = sub["condition"].tolist()
                vals = sub[met].fillna(0).tolist()
                ax_plot.plot(range(len(cids)), vals, "o-",
                             color=color, label=ds, lw=1.5, markersize=5)
            ax_plot.set_xticks(range(len(cids)))
            ax_plot.set_xticklabels(cids, rotation=45, ha="right", fontsize=7)
            ax_plot.set_ylabel(METRIC_LABELS[met])
            ax_plot.set_title(f"{METRIC_LABELS[met]} — iVision vs UWA")
            ax_plot.legend(); ax_plot.grid(alpha=0.3)
        plt.suptitle("Cross-Dataset Ablation Comparison", fontsize=12, fontweight="bold")
        plt.tight_layout()
        save_fig(fig, os.path.join(plot_dir, "ablation_cross_dataset_comparison.png"))

    # ── Final summary ──────────────────────────────────────────
    print("\n" + "="*65)
    print("  ABLATION COMPLETE")
    print("="*65)
    print(f"  Results  -> {cfg.out_dir}/")
    print(f"  Plots    -> {plot_dir}/")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
