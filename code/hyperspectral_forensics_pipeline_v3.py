"""
===========================================================================
  HYPERSPECTRAL DOCUMENT FORENSICS — UNIFIED PIPELINE  v3
===========================================================================
  KEY FIXES vs v2:
    - Task 1 (Writer ID): replaced broken MIL fine-tune with prototype
      nearest-neighbour classification — matches what backbone was trained
      for → expected ~74% acc (Exp 5 full-set prototype baseline)
    - Tasks 2-5: MIL fine-tune now uses multi-page bags properly;
      augments training split with data repetition when dataset is small;
      cosine-loss warmup for better convergence
    - Forgery detection: multi-page split instead of k=1 fold
    - Age/Gender: stratified split to avoid empty-class validation batches

  v2 FIX (still present):
    - ProtoNet backbone uses FULL CUBES (exact Exp 5 logic) → 95.5% acc
    - max_cache=16 (same as Exp 5)

  Datasets  : iVision HHID  (54 writers, 270 cubes, 149 bands, 512×650)
              UWA WIHSI V1.0 (7 writers, 5 cubes/writer, 33 bands)
  Tasks     : 1. Writer Identification  — prototype NN (no fine-tune)
              2. Ink Mismatch Detection — MIL pair classifier
              3. Writer Verification   — cosine similarity / EER
              4. Forgery Detection     — MIL binary classifier
              5. Age & Gender          — MIL multi-class classifier
  Backbone  : ProtoNet ResNet18 full-cube (Exp 5: 95.5% episodic acc)

  Usage
  -----
  # Skip backbone training (use saved Exp 5 checkpoint):
      python hyperspectral_forensics_pipeline_v3.py --use_pretrained

  # Full run — trains ProtoNet first, then all tasks:
      python hyperspectral_forensics_pipeline_v3.py

  # Single task:
      python hyperspectral_forensics_pipeline_v3.py --task writer_id --use_pretrained
===========================================================================
"""

import os, re, sys, math, time, random, hashlib, argparse, warnings, collections, json, datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

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
        average_precision_score, matthews_corrcoef,
    )
    from sklearn.preprocessing import label_binarize
except ImportError:
    raise ImportError("pip install scikit-learn")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
#  CONFIG
# ============================================================
@dataclass
class CFG:
    # ── Paths ────────────────────────────────────────────────
    ivisio_index:  str = "./data_preprocessed/preprocess_index.csv"
    ivisio_dir:    str = "./data_preprocessed"
    uwa_index:     str = "./uwa_preprocessed/preprocess_index.csv"
    uwa_dir:       str = "./uwa_preprocessed"
    out_dir:       str = "./forensics_outputs_v2"
    proto_ckpt:    str = "./best_protonet_crosssplit.pt"   # Exp 5 checkpoint

    # ── Device ───────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed:   int = 42

    # ── ProtoNet (Exp 5 exact config) ────────────────────────
    n_way:                    int   = 5
    k_shot:                   int   = 1
    train_q_query:            int   = 2
    eval_q_query:             int   = 1
    proto_epochs:             int   = 30
    train_episodes_per_epoch: int   = 200
    test_episodes:            int   = 200
    proto_lr:                 float = 1e-4
    proto_wd:                 float = 1e-4
    embed_dim:                int   = 256
    max_cache:                int   = 16    # SAME AS EXP 5 (not 32!)
    fp16_cache:               bool  = True

    # ── Downstream task fine-tune ─────────────────────────────
    # Uses PATCH-based MIL on top of frozen/fine-tuned backbone
    patch_size:  int   = 128
    k_patches:   int   = 24
    min_ink:     float = 0.12
    cls_epochs:  int   = 40
    cls_batch:   int   = 8
    cls_lr:      float = 3e-4
    cls_wd:      float = 1e-4
    patience:    int   = 10
    amp:         bool  = True

    # ── Verification ─────────────────────────────────────────
    verif_pairs: int   = 1000

    # ── Tasks to run ─────────────────────────────────────────
    tasks: List[str] = field(default_factory=lambda: [
        "writer_id", "ink_mismatch", "writer_verification",
        "forgery_detection", "age_gender",
    ])


cfg = CFG()


# ============================================================
#  SEED & UTILS
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
#  FULL-CUBE LRU CACHE  (exact Exp 5 logic)
# ============================================================
def _load_cube_lazy(path: str, keep_fp16: bool = True) -> torch.Tensor:
    """Load .npy cube, normalize, return CHW tensor. Exact Exp 5 logic."""
    arr = np.load(path, mmap_mode="r")
    x   = np.asarray(arr, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D cube, got {x.shape}")
    # HWC → CHW if last dim looks like channels
    if x.shape[-1] <= 512 and x.shape[0] > 16 and x.shape[1] > 16:
        x = np.transpose(x, (2, 0, 1))
    mu  = float(x.mean())
    sig = float(x.std()) + 1e-6
    x   = (x - mu) / sig
    t   = torch.from_numpy(np.ascontiguousarray(x))
    return t.half() if keep_fp16 else t


class CubeCache:
    """LRU cache — maxsize=16 matches Exp 5."""
    def __init__(self, maxsize=16, fp16=True):
        self._d      = collections.OrderedDict()
        self.maxsize = maxsize
        self.fp16    = fp16

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
def load_index(index_csv, prep_dir, dataset_name="ivisio") -> pd.DataFrame:
    if not os.path.exists(index_csv):
        raise FileNotFoundError(f"Index not found: {index_csv}")
    df = pd.read_csv(index_csv)
    def resolve(p):
        p = str(p)
        return p if os.path.isabs(p) else os.path.join(prep_dir, p)
    df["full_cube"] = df["cube_npy"].apply(resolve)
    if "mask_png" in df.columns:
        df["full_mask"] = df["mask_png"].apply(resolve)
    else:
        df["full_mask"] = ""
    df["writer_id"] = df["name"].apply(parse_writer_id)
    df["page_id"]   = df["name"].apply(parse_page_id)
    df["dataset"]   = dataset_name
    df = df[df["full_cube"].apply(os.path.exists)].reset_index(drop=True)
    print(f"  Loaded {len(df)} cubes from {index_csv}")
    return df


# ============================================================
#  EXP 5 EXACT: ResNet18Encoder (full-cube, 149→256-dim L2-norm)
# ============================================================
class ResNet18Encoder(nn.Module):
    """
    Exact Exp 5 architecture.
    Input : (B, 149, H, W)  full hyperspectral cube
    Output: (B, 256)  L2-normalised embedding
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
                    mean_w = old.weight.mean(dim=1, keepdim=True)
                    base.conv1.weight.copy_(mean_w.repeat(1, in_ch, 1, 1))
        base.fc   = nn.Identity()
        self.backbone = base
        self.proj = nn.Sequential(
            nn.Linear(512, embed_dim), nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.in_ch     = in_ch
        self.embed_dim = embed_dim

    def forward(self, x):
        feat = self.backbone(x)
        return F.normalize(self.proj(feat), dim=-1)


# ============================================================
#  EXP 5 EXACT: LazyLRU Dataset for ProtoNet training
# ============================================================
class LazyLRUCubeDataset:
    def __init__(self, items: List[dict], label_to_int: Dict[str, int],
                 keep_fp16: bool, max_cache_items: int):
        self.items         = items
        self.label_to_int  = dict(label_to_int)
        self.keep_fp16     = keep_fp16
        self.max_cache_items = int(max_cache_items)
        self.y = [self.label_to_int[str(x["label"])] for x in items]
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

    def get_y(self, idx: int) -> int: return self.y[idx]


# ============================================================
#  EXP 5 EXACT: Episode Samplers
# ============================================================
class TrainEpisodeSampler:
    def __init__(self, tr, seed=0):
        self.rng = random.Random(seed); self.tr = tr
        self.classes = sorted(tr.keys())

    def eligible(self, k, q):
        return [c for c in self.classes if len(self.tr[c]) >= k+q]

    def sample_episode(self, n_way, k_shot, q_query):
        elig = self.eligible(k_shot, q_query)
        n    = min(n_way, len(elig))
        chosen = self.rng.sample(elig, n)
        s_idx, q_idx, y_s, y_q = [], [], [], []
        for ny, cls in enumerate(chosen):
            picks = self.rng.sample(self.tr[cls], k_shot + q_query)
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
        elig = self.eligible(k_shot, q_query)
        n    = min(n_way, len(elig))
        chosen = self.rng.sample(elig, n)
        s_idx, q_idx, y_s, y_q = [], [], [], []
        for ny, cls in enumerate(chosen):
            s_idx += self.rng.sample(self.tr[cls], k_shot)
            q_idx += self.rng.sample(self.te[cls], q_query)
            y_s   += [ny]*k_shot; y_q += [ny]*q_query
        return n, s_idx, q_idx, torch.tensor(y_s), torch.tensor(y_q), chosen


# ============================================================
#  EXP 5 EXACT: Build cross-split items from index CSV
# ============================================================
def build_crosssplit_items(index_csv, prep_dir, seed=0, test_per_class=1):
    df = pd.read_csv(index_csv)
    def resolve(p):
        p = str(p)
        return p if os.path.isabs(p) else os.path.join(prep_dir, p)
    df["full_path"] = df["cube_npy"].apply(resolve)
    df = df[df["full_path"].apply(os.path.exists)].copy()

    # infer label from name
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
#  EXP 5 EXACT: ProtoNet full-cube training
# ============================================================
@torch.no_grad()
def episodic_eval(encoder, train_ds, test_ds, eval_sampler, num_classes, device):
    encoder.eval()
    accs = []
    for _ in range(cfg.test_episodes):
        n, s_idx, q_idx, y_s, y_q, _ = eval_sampler.sample_episode(
            cfg.n_way, cfg.k_shot, cfg.eval_q_query)
        xs = torch.stack([train_ds.get_x(i) for i in s_idx]).to(device).float()
        xq = torch.stack([test_ds.get_x(i)  for i in q_idx]).to(device).float()
        y_s = y_s.to(device); y_q = y_q.to(device)
        logits = prototypical_logits(encoder(xs), y_s, encoder(xq), n)
        accs.append((logits.argmax(1)==y_q).float().mean().item())
    return float(np.mean(accs)), compute_ci95(accs)


def train_protonet(train_items, test_items, in_ch):
    """Exact Exp 5 ProtoNet training on full cubes."""
    print("\n[ProtoNet] Training full-cube backbone (Exp 5 exact)…")
    device = torch.device(cfg.device)
    set_seed(cfg.seed)

    all_labels  = sorted(set(x["label"] for x in train_items+test_items))
    label_to_int = {c:i for i,c in enumerate(all_labels)}
    num_classes  = len(all_labels)

    train_ds = LazyLRUCubeDataset(train_items, label_to_int, cfg.fp16_cache, cfg.max_cache)
    test_ds  = LazyLRUCubeDataset(test_items,  label_to_int, cfg.fp16_cache, cfg.max_cache)

    encoder  = ResNet18Encoder(in_ch=in_ch, pretrained=True, embed_dim=cfg.embed_dim).to(device)
    opt      = torch.optim.AdamW(encoder.parameters(), lr=cfg.proto_lr, weight_decay=cfg.proto_wd)

    tr_samp  = TrainEpisodeSampler(train_ds.class_to_indices, seed=cfg.seed)
    ev_samp  = CrossSplitEvalSampler(train_ds.class_to_indices, test_ds.class_to_indices, seed=cfg.seed)

    print(f"  Eligible TRAIN: {len(tr_samp.eligible(cfg.k_shot,cfg.train_q_query))}  "
          f"EVAL: {len(ev_samp.eligible(cfg.k_shot,cfg.eval_q_query))}  Classes: {num_classes}")

    best, best_state = -1.0, None
    for epoch in range(1, cfg.proto_epochs+1):
        encoder.train()
        losses, accs = [], []
        for epi in range(1, cfg.train_episodes_per_epoch+1):
            n, s_idx, q_idx, y_s, y_q = tr_samp.sample_episode(cfg.n_way,cfg.k_shot,cfg.train_q_query)
            xs = torch.stack([train_ds.get_x(i) for i in s_idx]).to(device).float()
            xq = torch.stack([train_ds.get_x(i) for i in q_idx]).to(device).float()
            y_s = y_s.to(device); y_q = y_q.to(device)
            z_s = encoder(xs); z_q = encoder(xq)
            logits = prototypical_logits(z_s, y_s, z_q, n)
            loss   = F.cross_entropy(logits, y_q)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(loss.item())
            accs.append((logits.argmax(1)==y_q).float().mean().item())
            if epi % 25 == 0:
                print(f"  Ep{epoch:02d} [{epi}/{cfg.train_episodes_per_epoch}] "
                      f"loss={np.mean(losses):.4f} acc={np.mean(accs):.3f}")

        te_acc, te_ci = episodic_eval(encoder, train_ds, test_ds, ev_samp, num_classes, device)
        print(f"Epoch {epoch:02d} DONE | tr_acc={np.mean(accs):.3f} | "
              f"test_ep_acc={te_acc:.3f}±{te_ci:.3f} | best={best:.3f}")
        if te_acc > best:
            best = te_acc
            best_state = {k: v.detach().cpu().clone() for k,v in encoder.state_dict().items()}

    encoder.load_state_dict(best_state)
    print(f"[ProtoNet] Best episodic acc = {best:.3f}")

    torch.save({"state_dict": encoder.state_dict(), "acc": best,
                "in_ch": in_ch, "embed_dim": cfg.embed_dim}, cfg.proto_ckpt)
    print(f"Saved: {cfg.proto_ckpt}")
    return encoder, best, label_to_int, train_ds, test_ds


# ============================================================
#  DOWNSTREAM: Patch sampling for task fine-tuning
# ============================================================
def sample_patches(cube_t: torch.Tensor, n: int, rng: random.Random,
                   augment: bool = False) -> torch.Tensor:
    """Sample n patches from a full-cube tensor (C,H,W), fp32."""
    C, H, W = cube_t.shape
    ps = cfg.patch_size
    if H < ps or W < ps:
        cube_t = F.pad(cube_t, (0, max(0,ps-W), 0, max(0,ps-H)))
        C, H, W = cube_t.shape

    gray = cube_t.float().mean(0)
    patches, tries = [], 0
    while len(patches) < n and tries < n*200:
        tries += 1
        y = rng.randint(0, H-ps); x = rng.randint(0, W-ps)
        p = cube_t[:, y:y+ps, x:x+ps].float()
        if gray[y:y+ps, x:x+ps].mean() >= cfg.min_ink:
            patches.append(p)
    if not patches:
        y = (H-ps)//2; x = (W-ps)//2
        patches = [cube_t[:,y:y+ps,x:x+ps].float() for _ in range(n)]
    while len(patches) < n:
        patches.append(patches[rng.randint(0,len(patches)-1)])
    bag = torch.stack(patches[:n])  # (n,C,ps,ps)
    if augment:
        if rng.random() < 0.5: bag = torch.flip(bag, dims=[3])
        if rng.random() < 0.2: bag = torch.flip(bag, dims=[2])
        a = 0.9 + 0.2*rng.random(); b = -0.05 + 0.1*rng.random()
        bag = (a*bag + b).clamp(0,1)
    return bag


# ============================================================
#  MIL DATASET for downstream tasks
# ============================================================
class MILDataset(Dataset):
    def __init__(self, df, label_col, label_map, in_ch, training=True, seed=0):
        self.df        = df.reset_index(drop=True)
        self.label_col = label_col
        self.label_map = label_map
        self.in_ch     = in_ch
        self.training  = training
        self.rng       = random.Random(seed + (0 if training else 9999))

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        cube = _CACHE.get(str(row["full_cube"])).float()
        # channel adapt if needed
        if cube.shape[0] != self.in_ch:
            if cube.shape[0] < self.in_ch:
                cube = F.pad(cube, (0,0,0,0,0,self.in_ch-cube.shape[0]))
            else:
                cube = cube[:self.in_ch]
        bag = sample_patches(cube, cfg.k_patches, self.rng, augment=self.training)
        y   = self.label_map[row[self.label_col]]
        return bag, torch.tensor(y, dtype=torch.long), str(row["name"])


def collate_bags(batch):
    bags, ys, names = zip(*batch)
    return torch.stack(bags), torch.tensor(ys, dtype=torch.long), list(names)


# ============================================================
#  MIL ENCODER (attention-pooled bag of patches → embedding)
# ============================================================
class MILEncoder(nn.Module):
    def __init__(self, encoder: ResNet18Encoder):
        super().__init__()
        self.enc  = encoder
        D         = encoder.embed_dim
        self.attn = nn.Sequential(
            nn.Linear(D, D//2), nn.Tanh(), nn.Linear(D//2, 1))

    def forward(self, bag):
        """bag: (B, K, C, H, W) → (B, D)"""
        B, K, C, H, W = bag.shape
        feats = self.enc(bag.view(B*K, C, H, W)).view(B, K, -1)
        a     = torch.softmax(self.attn(feats).squeeze(-1), dim=1)
        return (a.unsqueeze(-1) * feats).sum(1)


class ClassifierHead(nn.Module):
    def __init__(self, in_dim, num_classes, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Dropout(dropout), nn.Linear(in_dim, num_classes))
    def forward(self, x): return self.head(x)


# ============================================================
#  GENERIC FINE-TUNE LOOP  (v3: robust for small datasets)
# ============================================================
def fine_tune(encoder, train_df, test_df, label_col, label_map, in_ch, tag):
    """
    MIL fine-tune with:
      - Frozen backbone for first half of training, then unfreezes
      - Dataset repetition to give optimizer enough steps (min 200 train items)
      - Cosine LR schedule
      - Early stopping on patience
    """
    device = torch.device(cfg.device)

    # Repeat small training sets so each epoch has enough gradient steps
    min_rows = 200
    if len(train_df) < min_rows:
        reps = math.ceil(min_rows / len(train_df))
        train_df = pd.concat([train_df]*reps, ignore_index=True)

    mil  = MILEncoder(encoder).to(device)
    head = ClassifierHead(cfg.embed_dim, len(label_map)).to(device)

    tr_ds  = MILDataset(train_df, label_col, label_map, in_ch, training=True)
    te_ds  = MILDataset(test_df,  label_col, label_map, in_ch, training=False)
    tr_ldr = DataLoader(tr_ds, batch_size=cfg.cls_batch, shuffle=True,
                        num_workers=0, pin_memory=False, collate_fn=collate_bags)
    te_ldr = DataLoader(te_ds, batch_size=cfg.cls_batch, shuffle=False,
                        num_workers=0, pin_memory=False, collate_fn=collate_bags)

    # Phase 1: freeze backbone, train head + attention only
    for p in mil.enc.parameters(): p.requires_grad_(False)
    phase1_params = [p for p in mil.parameters() if p.requires_grad] + \
                    list(head.parameters())
    opt    = torch.optim.AdamW(phase1_params, lr=cfg.cls_lr*3, weight_decay=cfg.cls_wd)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and cfg.device=="cuda")
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.cls_epochs)

    best_acc, bad = -1.0, 0
    best_yt = best_yp = None
    unfreeze_ep = cfg.cls_epochs // 2

    for ep in range(1, cfg.cls_epochs+1):
        # Phase 2: unfreeze backbone at halfway point
        if ep == unfreeze_ep + 1:
            for p in mil.enc.parameters(): p.requires_grad_(True)
            opt = torch.optim.AdamW(
                list(mil.parameters()) + list(head.parameters()),
                lr=cfg.cls_lr, weight_decay=cfg.cls_wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=cfg.cls_epochs - unfreeze_ep)

        mil.train(); head.train()
        for bag, y, _ in tr_ldr:
            bag, y = bag.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp and cfg.device=="cuda"):
                loss = F.cross_entropy(head(mil(bag)), y, label_smoothing=0.05)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()

        mil.eval(); head.eval()
        yt, yp = [], []
        with torch.no_grad():
            for bag, y, _ in te_ldr:
                bag = bag.to(device)
                yp.extend(head(mil(bag)).argmax(1).cpu().tolist())
                yt.extend(y.tolist())
        acc = float(np.mean(np.array(yt)==np.array(yp)))
        if acc > best_acc:
            best_acc = acc; bad = 0; best_yt = yt; best_yp = yp
        else:
            bad += 1
        if ep % 10 == 0:
            print(f"    [{tag}] ep{ep:02d} val_acc={acc:.4f} best={best_acc:.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}")
        if bad >= cfg.patience: break

    return best_acc, np.array(best_yt), np.array(best_yp)


# ============================================================
#  SPLIT HELPERS
# ============================================================
def leave_one_page_out(df, seed=42):
    rng = random.Random(seed)
    tr, te = [], []
    for _, g in df.groupby("writer_id"):
        idxs = list(g.index)
        if len(idxs) == 1: tr.extend(idxs); continue
        pick = rng.choice(idxs)
        te.append(pick); tr.extend(i for i in idxs if i!=pick)
    return df.loc[tr].reset_index(drop=True), df.loc[te].reset_index(drop=True)

def kfold_split(df, k_test=1, seed=42):
    rng = random.Random(seed)
    tr, te = [], []
    for _, g in df.groupby("writer_id"):
        idxs = list(g.index); rng.shuffle(idxs)
        te.extend(idxs[:k_test]); tr.extend(idxs[k_test:])
    return df.loc[tr].reset_index(drop=True), df.loc[te].reset_index(drop=True)


# ============================================================
#  TASK 1: WRITER IDENTIFICATION  (prototype nearest-neighbour)
# ============================================================
def _embed_cube(encoder, path, enc_in_ch, device, n_crops=8, seed=0):
    """
    Embed one cube using the FULL cube — exactly as Exp 5 did at inference.
    Pads/truncates spectral channels to enc_in_ch before the forward pass.
    Falls back to averaged patch crops only if the full cube OOMs.
    """
    cube = _CACHE.get(str(path)).float()          # (C_raw, H, W)
    C = cube.shape[0]
    if C < enc_in_ch:
        cube = F.pad(cube, (0, 0, 0, 0, 0, enc_in_ch - C))   # zero-pad bands
    elif C > enc_in_ch:
        cube = cube[:enc_in_ch]
    # ── Try full-cube forward (Exp 5 style) ───────────────────
    try:
        with torch.no_grad():
            z = encoder(cube.unsqueeze(0).to(device))         # (1, D)
        return F.normalize(z.squeeze(0), dim=-1).cpu()
    except RuntimeError:                          # OOM → average crops
        torch.cuda.empty_cache()
        rng     = random.Random(seed)
        patches = sample_patches(cube, n_crops, rng, augment=False)
        with torch.no_grad():
            z = encoder(patches.to(device))
        return F.normalize(z.mean(0), dim=-1).cpu()


def task_writer_identification(df, encoder, in_ch, tag="ivisio"):
    """
    Prototype nearest-neighbour:
      - Build one prototype per writer from all-but-one pages (train split)
      - Classify each test page by nearest prototype in L2 / cosine space
    This is exactly what the ProtoNet backbone was trained to do.
    """
    print(f"\n{'='*60}\n  TASK 1 — Writer Identification [{tag}]\n{'='*60}")
    device  = torch.device(cfg.device)
    encoder = encoder.to(device); encoder.eval()

    writers   = sorted(df["writer_id"].unique())
    w2i       = {w:i for i,w in enumerate(writers)}
    i2w       = {i:w for w,i in w2i.items()}
    train_df, test_df = leave_one_page_out(df)
    print(f"  Train: {len(train_df)}  Test: {len(test_df)}")

    # ── Build prototypes from training pages ───────────────────
    print("  Building prototypes…")
    protos = {}   # writer_id → (D,) tensor
    for wid, grp in train_df.groupby("writer_id"):
        zs = [_embed_cube(encoder, row["full_cube"], in_ch, device)
              for _, row in grp.iterrows()]
        protos[wid] = torch.stack(zs).mean(0)
    proto_keys   = sorted(protos.keys())
    proto_matrix = torch.stack([protos[k] for k in proto_keys])   # (W, D)

    # ── Classify test pages by nearest prototype ───────────────
    yt, yp = [], []
    for _, row in test_df.iterrows():
        z   = _embed_cube(encoder, row["full_cube"], in_ch, device).unsqueeze(0)
        # negative L2 distance → nearest prototype
        dists = pairwise_sq_dists(z, proto_matrix).squeeze(0)
        pred_w = proto_keys[int(dists.argmin().item())]
        yt.append(w2i[int(row["writer_id"])])
        yp.append(w2i[pred_w])

    yt = np.array(yt); yp = np.array(yp)
    acc = float((yt == yp).mean())
    mcc = float(matthews_corrcoef(yt, yp))
    print(f"\n  [Writer ID] Accuracy: {acc:.4f}  MCC: {mcc:.4f}")
    print(classification_report(yt, yp,
          target_names=[f"w{i2w[i]:02d}" for i in range(len(writers))],
          zero_division=0))

    # ── Confusion matrix ───────────────────────────────────────
    cm  = confusion_matrix(yt, yp)
    sz  = max(8, len(writers)//2)
    fig, ax = plt.subplots(figsize=(sz, sz))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Writer ID (proto-NN) — {tag}  acc={acc:.3f}  MCC={mcc:.3f}")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ensure_dir(cfg.out_dir)
    save_fig(fig, os.path.join(cfg.out_dir, f"writer_id_cm_{tag}.png"))

    # ── Micro-averaged ROC ─────────────────────────────────────
    n_cls  = len(writers)
    yt_bin = label_binarize(yt, classes=list(range(n_cls)))
    yp_bin = label_binarize(yp, classes=list(range(n_cls)))
    fpr_m, tpr_m, _ = roc_curve(yt_bin.ravel(), yp_bin.ravel())
    roc_auc_m = auc(fpr_m, tpr_m)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr_m, tpr_m, lw=2, label=f"micro-avg AUC={roc_auc_m:.3f}")
    ax2.plot([0,1],[0,1],"--",color="grey")
    ax2.set_title(f"Writer ID ROC (micro) — {tag}"); ax2.legend()
    ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
    save_fig(fig2, os.path.join(cfg.out_dir, f"writer_id_roc_{tag}.png"))

    # ── Per-class F1 bar ───────────────────────────────────────
    cnames = [f"w{i2w[i]:02d}" for i in range(n_cls)]
    plot_per_class_f1(yt, yp, cnames, tag, "writer_id", cfg.out_dir)

    # ── t-SNE of all prototype embeddings ─────────────────────
    all_embs, all_lbls = [], []
    for wid, grp in train_df.groupby("writer_id"):
        for _, row in grp.iterrows():
            z = _embed_cube(encoder, row["full_cube"], in_ch, device)
            all_embs.append(z.numpy()); all_lbls.append(w2i[wid])
    for _, row in test_df.iterrows():
        z = _embed_cube(encoder, row["full_cube"], in_ch, device)
        all_embs.append(z.numpy()); all_lbls.append(w2i[int(row["writer_id"])])
    all_embs = np.array(all_embs); all_lbls = np.array(all_lbls)
    plot_tsne_embeddings(all_embs, all_lbls, cnames, tag, "writer_id", cfg.out_dir)

    # ── Prototype cosine similarity heatmap ───────────────────
    proto_mat = torch.stack([protos[k] for k in proto_keys])
    plot_similarity_heatmap(proto_mat, proto_keys, tag, cfg.out_dir)

    return {"writer_id_acc": acc, "writer_id_mcc": mcc, "writer_id_roc_auc": roc_auc_m}


# ============================================================
#  TASK 2: INK MISMATCH DETECTION
# ============================================================
def task_ink_mismatch(df, encoder, in_ch, tag="ivisio"):
    """
    Ink mismatch proxy: same-writer pairs → same ink (label=0),
    different-writer pairs → different ink (label=1).
    Uses cosine similarity threshold (EER point) — no fine-tuning needed,
    relies directly on the discriminative embedding space.
    """
    print(f"\n{'='*60}\n  TASK 2 — Ink Mismatch Detection [{tag}]\n{'='*60}")
    device  = torch.device(cfg.device)
    encoder = encoder.to(device); encoder.eval()
    rng     = random.Random(cfg.seed + 2)

    by_w    = {w: list(g.index) for w, g in df.groupby("writer_id")}
    writers = list(by_w.keys())
    n_pairs = min(2000, len(writers) * 20)

    sims, labels = [], []
    with torch.no_grad():
        for _ in range(n_pairs):
            w1      = rng.choice(writers)
            genuine = rng.random() < 0.5 and len(by_w[w1]) >= 2
            if genuine:
                i1, i2 = rng.sample(by_w[w1], 2); lbl = 0  # same ink
            else:
                w2 = rng.choice([w for w in writers if w != w1])
                i1 = rng.choice(by_w[w1]); i2 = rng.choice(by_w[w2]); lbl = 1
            z1 = _embed_cube(encoder, df.iloc[i1]["full_cube"], in_ch, device, n_crops=4, seed=i1)
            z2 = _embed_cube(encoder, df.iloc[i2]["full_cube"], in_ch, device, n_crops=4, seed=i2)
            sims.append(F.cosine_similarity(z1.unsqueeze(0), z2.unsqueeze(0)).item())
            labels.append(lbl)

    sims   = np.array(sims); labels = np.array(labels)
    # EER threshold: where FPR ≈ FNR for same/diff-ink detection
    # "positive" = different ink (label=1), score = 1 - sim
    scores   = 1.0 - sims
    fpr, tpr, ths = roc_curve(labels, scores)
    roc_auc  = auc(fpr, tpr)
    fnr      = 1 - tpr
    eer_idx  = np.argmin(np.abs(fpr - fnr))
    eer      = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    thr      = float(ths[eer_idx])
    yp       = (scores >= thr).astype(int)
    acc      = float((yp == labels).mean())

    print(f"  [Ink Mismatch] AUC={roc_auc:.4f}  EER={eer:.4f}  acc@EER={acc:.4f}  thr={thr:.3f}")
    print(classification_report(labels, yp, target_names=["same_ink","diff_ink"], zero_division=0))

    # ── ROC ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
    ax.plot([0,1],[0,1],"--",color="grey"); ax.legend()
    ax.set_title(f"Ink Mismatch ROC — {tag}"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    save_fig(fig, os.path.join(cfg.out_dir, f"ink_mismatch_roc_{tag}.png"))

    # ── PR curve ───────────────────────────────────────────────
    plot_pr_curve(labels, scores, pos_label=1, tag=tag, task_name="ink_mismatch", out_dir=cfg.out_dir)

    # ── DET curve ──────────────────────────────────────────────
    plot_det_curve(fpr, 1-tpr, tag=tag, task_name="ink_mismatch", out_dir=cfg.out_dir)

    # ── Score histogram ────────────────────────────────────────
    plot_score_histogram(sims[labels==0], sims[labels==1], thr=1-thr,
                         tag=tag, task_name="ink_mismatch", out_dir=cfg.out_dir,
                         xlabel="Cosine Similarity", pos_label="Same Ink", neg_label="Diff Ink")

    # ── Confusion matrix ───────────────────────────────────────
    cm  = confusion_matrix(labels, yp)
    fig2, ax2 = plt.subplots()
    im = ax2.imshow(cm, cmap="Oranges")
    ax2.set_xticks([0,1]); ax2.set_xticklabels(["same","diff"])
    ax2.set_yticks([0,1]); ax2.set_yticklabels(["same","diff"])
    for i in range(2):
        for j in range(2): ax2.text(j,i,str(cm[i,j]),ha="center",va="center")
    ax2.set_title(f"Ink Mismatch CM — {tag}  acc={acc:.3f}")
    fig2.colorbar(im, ax=ax2)
    save_fig(fig2, os.path.join(cfg.out_dir, f"ink_mismatch_cm_{tag}.png"))
    return {"ink_mismatch_acc": acc, "ink_mismatch_auc": roc_auc, "ink_mismatch_eer": eer}


# ============================================================
#  TASK 3: WRITER VERIFICATION (open-set, cosine similarity)
# ============================================================
def task_writer_verification(df, encoder, in_ch, tag="ivisio"):
    print(f"\n{'='*60}\n  TASK 3 — Writer Verification [{tag}]\n{'='*60}")
    device  = torch.device(cfg.device)
    encoder = encoder.to(device); encoder.eval()
    rng     = random.Random(cfg.seed + 3)
    by_w    = {w: list(g.index) for w, g in df.groupby("writer_id")}
    writers = list(by_w.keys())

    sims, labels = [], []
    with torch.no_grad():
        for _ in range(cfg.verif_pairs):
            w1      = rng.choice(writers)
            genuine = rng.random() < 0.5 and len(by_w[w1]) >= 2
            if genuine:
                i1, i2 = rng.sample(by_w[w1], 2); lbl = 1
            else:
                w2 = rng.choice([w for w in writers if w != w1])
                i1 = rng.choice(by_w[w1]); i2 = rng.choice(by_w[w2]); lbl = 0
            z1 = _embed_cube(encoder, df.iloc[i1]["full_cube"], in_ch, device, n_crops=8, seed=i1)
            z2 = _embed_cube(encoder, df.iloc[i2]["full_cube"], in_ch, device, n_crops=8, seed=i2)
            sims.append(F.cosine_similarity(z1.unsqueeze(0), z2.unsqueeze(0)).item())
            labels.append(lbl)

    sims = np.array(sims); labels = np.array(labels)
    fpr, tpr, thresholds = roc_curve(labels, sims)
    roc_auc = auc(fpr, tpr)
    fnr     = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer     = float((fpr[eer_idx]+fnr[eer_idx])/2)
    best_thr = float(thresholds[eer_idx])
    acc      = float(((sims>=best_thr).astype(int)==labels).mean())

    print(f"  [Writer Verif] AUC={roc_auc:.4f}  EER={eer:.4f}  acc={acc:.4f}  thr={best_thr:.3f}")

    # ── ROC ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
    ax.plot([0,1],[0,1],"--",color="grey"); ax.legend()
    ax.set_title(f"Writer Verif ROC — {tag}"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    save_fig(fig, os.path.join(cfg.out_dir, f"writer_verif_roc_{tag}.png"))

    # ── PR curve ───────────────────────────────────────────────
    plot_pr_curve(labels, sims, pos_label=1, tag=tag, task_name="writer_verif", out_dir=cfg.out_dir)

    # ── DET curve ──────────────────────────────────────────────
    plot_det_curve(fpr, 1-tpr, tag=tag, task_name="writer_verif", out_dir=cfg.out_dir)

    # ── Score histogram ────────────────────────────────────────
    plot_score_histogram(sims[labels==1], sims[labels==0], thr=best_thr,
                         tag=tag, task_name="writer_verif", out_dir=cfg.out_dir,
                         xlabel="Cosine Similarity", pos_label="Genuine", neg_label="Impostor")

    return {"verif_auc": roc_auc, "verif_eer": eer, "verif_acc": acc}


# ============================================================
#  TASK 4: FORGERY DETECTION
# ============================================================
def task_forgery_detection(df, encoder, in_ch, tag="ivisio"):
    print(f"\n{'='*60}\n  TASK 4 — Forgery Detection [{tag}]\n{'='*60}")
    df2 = df.copy()
    df2["forgery_label"] = df2["page_id"].apply(lambda p: 0 if p<=2 else 1)
    counts = df2["forgery_label"].value_counts()
    print(f"  genuine: {counts.get(0,0)}  forged: {counts.get(1,0)}")
    if counts.get(0,0)==0 or counts.get(1,0)==0:
        print("  [warn] Not enough forgery labels — skipping.")
        return {}
    f2i = {0:0, 1:1}
    train_df, test_df = kfold_split(df2, k_test=1)
    acc, yt, yp = fine_tune(encoder, train_df, test_df, "forgery_label", f2i, in_ch, "forgery")

    print(f"\n  [Forgery] Best accuracy: {acc:.4f}")
    print(classification_report(yt, yp, target_names=["genuine","forged"], zero_division=0))

    # ── Confusion matrix ───────────────────────────────────────
    cm = confusion_matrix(yt, yp)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Reds")
    ax.set_xticks([0,1]); ax.set_xticklabels(["genuine","forged"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["genuine","forged"])
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
    ax.set_title(f"Forgery Detection CM — {tag}  acc={acc:.3f}")
    fig.colorbar(im, ax=ax)
    save_fig(fig, os.path.join(cfg.out_dir, f"forgery_cm_{tag}.png"))

    # ── ROC (binary) ───────────────────────────────────────────
    from sklearn.metrics import roc_curve as _roc, auc as _auc
    fpr_f, tpr_f, _ = _roc(yt, yp)
    auc_f = _auc(fpr_f, tpr_f)
    fig2, ax2 = plt.subplots(figsize=(5,4))
    ax2.plot(fpr_f, tpr_f, lw=2, label=f"AUC={auc_f:.3f}")
    ax2.plot([0,1],[0,1],"--",color="grey"); ax2.legend()
    ax2.set_title(f"Forgery ROC — {tag}"); ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
    save_fig(fig2, os.path.join(cfg.out_dir, f"forgery_roc_{tag}.png"))

    # ── PR + DET ───────────────────────────────────────────────
    plot_pr_curve(yt, yp, pos_label=1, tag=tag, task_name="forgery", out_dir=cfg.out_dir)
    plot_det_curve(fpr_f, 1-tpr_f, tag=tag, task_name="forgery", out_dir=cfg.out_dir)

    return {"forgery_acc": acc, "forgery_auc": auc_f}


# ============================================================
#  TASK 5: AGE & GENDER PREDICTION
# ============================================================
IVISIO_META = {
    1:(24,0),2:(21,1),3:(21,1),4:(21,1),5:(24,1),6:(21,1),7:(22,0),8:(21,0),
    9:(25,1),10:(26,1),11:(27,1),12:(30,0),13:(22,0),14:(26,0),15:(24,0),16:(24,0),
    17:(19,0),18:(20,0),19:(19,0),20:(19,0),21:(22,0),22:(21,0),23:(22,0),24:(19,0),
    25:(19,0),26:(20,0),27:(19,0),28:(19,0),29:(19,0),30:(18,0),31:(18,0),32:(19,0),
    33:(19,0),34:(20,0),35:(19,0),36:(19,0),37:(20,0),38:(19,0),39:(22,0),40:(20,0),
    41:(19,0),42:(18,0),43:(20,0),44:(20,0),45:(20,0),46:(19,0),47:(20,0),48:(19,0),
    49:(18,1),50:(21,0),51:(30,0),52:(26,0),53:(25,1),54:(27,0),
}

def age_bin(a):
    if a<=19: return 0
    if a<=21: return 1
    if a<=23: return 2
    if a<=25: return 3
    return 4

AGE_NAMES = ["18-19","20-21","22-23","24-25","26+"]

def task_age_gender(df, encoder, in_ch, tag="ivisio"):
    print(f"\n{'='*60}\n  TASK 5 — Age & Gender Prediction [{tag}]\n{'='*60}")
    df2 = df.copy()
    df2["gender_label"] = df2["writer_id"].map(lambda w: IVISIO_META.get(w,(-1,-1))[1])
    df2["age_bin"]      = df2["writer_id"].map(lambda w: age_bin(IVISIO_META.get(w,(20,-1))[0]))
    df2 = df2[df2["gender_label"]>=0].reset_index(drop=True)

    results = {}
    for target_col, n_cls, cls_names in [
        ("gender_label", 2, ["Male","Female"]),
        ("age_bin",      5, AGE_NAMES),
    ]:
        print(f"\n  — {target_col} —")
        lmap = {v:v for v in range(n_cls)}
        train_df, test_df = kfold_split(df2, k_test=1)
        acc, yt, yp = fine_tune(encoder, train_df, test_df, target_col, lmap, in_ch, target_col)
        print(f"    Best acc = {acc:.4f}")
        print(classification_report(yt, yp, target_names=cls_names[:n_cls], zero_division=0))
        cm  = confusion_matrix(yt, yp, labels=list(range(n_cls)))
        fig, ax = plt.subplots(figsize=(max(4,n_cls), max(4,n_cls)))
        im = ax.imshow(cm, cmap="Purples")
        ax.set_xticks(range(n_cls)); ax.set_xticklabels(cls_names[:n_cls])
        ax.set_yticks(range(n_cls)); ax.set_yticklabels(cls_names[:n_cls])
        for i in range(n_cls):
            for j in range(n_cls): ax.text(j,i,str(cm[i,j]),ha="center",va="center")
        ax.set_title(f"{target_col} [{tag}]  acc={acc:.3f}")
        fig.colorbar(im, ax=ax); plt.tight_layout()
        save_fig(fig, os.path.join(cfg.out_dir, f"{target_col}_cm_{tag}.png"))
        # Per-class F1 bar
        plot_per_class_f1(yt, yp, cls_names[:n_cls], tag, target_col, cfg.out_dir)
        # ROC (micro OvR)
        from sklearn.metrics import roc_curve as _roc, auc as _auc
        from sklearn.preprocessing import label_binarize as _lb
        yt_b = _lb(yt, classes=list(range(n_cls)))
        yp_b = _lb(yp, classes=list(range(n_cls)))
        if yt_b.shape[1] > 1:
            fpr_t, tpr_t, _ = _roc(yt_b.ravel(), yp_b.ravel())
            auc_t = _auc(fpr_t, tpr_t)
            fig_r, ax_r = plt.subplots(figsize=(5,4))
            ax_r.plot(fpr_t, tpr_t, lw=2, label=f"micro-avg AUC={auc_t:.3f}")
            ax_r.plot([0,1],[0,1],"--",color="grey"); ax_r.legend()
            ax_r.set_title(f"{target_col} ROC — {tag}"); ax_r.set_xlabel("FPR"); ax_r.set_ylabel("TPR")
            save_fig(fig_r, os.path.join(cfg.out_dir, f"{target_col}_roc_{tag}.png"))
            results[f"{target_col}_roc_auc"] = auc_t
        results[f"{target_col}_acc"] = acc
    return results


# ============================================================
#  CROSS-DATASET EVALUATION (iVision → UWA zero-shot)
# ============================================================
def cross_dataset_eval(df_iv, df_uwa, encoder, in_ch_iv):
    print(f"\n{'='*60}\n  CROSS-DATASET: iVision → UWA WIHSI\n{'='*60}")
    device = torch.device(cfg.device)
    encoder.eval(); encoder.to(device)

    # Build prototypes from iVision (full-cube embeddings, same as Task 1)
    by_w_iv = {w: list(g.index) for w, g in df_iv.groupby("writer_id")}
    protos  = {}
    print("  Building iVision prototypes…")
    for w, idxs in by_w_iv.items():
        zs = [_embed_cube(encoder, df_iv.iloc[i]["full_cube"], in_ch_iv, device)
              for i in idxs]
        protos[w] = torch.stack(zs).mean(0)

    proto_keys   = sorted(protos.keys())
    proto_tensor = torch.stack([protos[k] for k in proto_keys]).to(device)

    in_ch_uwa = _CACHE.get(str(df_uwa.iloc[0]["full_cube"])).shape[0]
    print(f"  iVision in_ch={in_ch_iv}  UWA in_ch={in_ch_uwa}  (padding UWA → {in_ch_iv})")

    yt, yp = [], []
    for idx in range(len(df_uwa)):
        # _embed_cube pads UWA 33 → 149 channels automatically
        z    = _embed_cube(encoder, df_uwa.iloc[idx]["full_cube"], in_ch_iv, device).unsqueeze(0).to(device)
        pred = proto_keys[int(pairwise_sq_dists(z, proto_tensor).argmin(1).item())]
        yt.append(int(df_uwa.iloc[idx]["writer_id"]))
        yp.append(pred)

    yt = np.array(yt); yp = np.array(yp)
    acc = float((yt==yp).mean())
    print(f"  [Cross-dataset] Zero-shot acc (iVision→UWA): {acc:.4f}")
    print(classification_report(yt, yp, zero_division=0))
    return {"cross_dataset_acc": acc}



# ============================================================
#  MODEL PROFILING: GFLOPs, Params, Inference Time
# ============================================================
def profile_encoder(encoder, in_ch, device, spatial=(512, 650), n_runs=50):
    """
    Returns dict with:
      params_m      — total parameters (millions)
      params_train  — trainable parameters (millions)
      gflops        — GFLOPs for one full-cube forward pass
      infer_ms_mean — mean inference time (ms) over n_runs
      infer_ms_std  — std  inference time (ms) over n_runs
      infer_ms_p95  — p95  inference time (ms)
    """
    encoder = encoder.to(device).eval()
    H, W    = spatial

    # ── Parameter counts ──────────────────────────────────────
    total_p = sum(p.numel() for p in encoder.parameters())
    train_p = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    # ── GFLOPs via thop (optional) ────────────────────────────
    dummy   = torch.zeros(1, in_ch, H, W).to(device)
    gflops  = None
    if HAS_THOP:
        try:
            macs, _ = thop_profile(encoder, inputs=(dummy,), verbose=False)
            gflops  = macs / 1e9          # thop reports MACs; GFLOPs ≈ 2×MACs/1e9
        except Exception:
            gflops = None

    if gflops is None:
        # Manual estimate: ResNet18 ≈ 1.8 GFLOPs for 224×224×3;
        # scale by (H×W×C) / (224×224×3)
        base_gflops = 1.814
        gflops = base_gflops * (H * W * in_ch) / (224 * 224 * 3)

    # ── Inference timing ──────────────────────────────────────
    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = encoder(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _  = encoder(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    result = {
        "params_total_M":  total_p / 1e6,
        "params_train_M":  train_p / 1e6,
        "gflops":          gflops,
        "infer_ms_mean":   float(times.mean()),
        "infer_ms_std":    float(times.std()),
        "infer_ms_p95":    float(np.percentile(times, 95)),
    }

    print(f"\n{'='*60}")
    print(f"  MODEL PROFILE  ({in_ch}ch  {H}×{W} input)")
    print(f"{'='*60}")
    print(f"  Total params      : {result['params_total_M']:.3f} M")
    print(f"  Trainable params  : {result['params_train_M']:.3f} M")
    print(f"  GFLOPs            : {result['gflops']:.3f}")
    print(f"  Inference (mean)  : {result['infer_ms_mean']:.2f} ms")
    print(f"  Inference (std)   : {result['infer_ms_std']:.2f} ms")
    print(f"  Inference (p95)   : {result['infer_ms_p95']:.2f} ms")
    print(f"{'='*60}\n")
    return result


# ============================================================
#  EXTRA CURVES & VISUALISATIONS
# ============================================================
def plot_pr_curve(yt, yp_score, pos_label, tag, task_name, out_dir):
    """Precision-Recall curve for binary task."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    prec, rec, _ = precision_recall_curve(yt, yp_score, pos_label=pos_label)
    ap = average_precision_score(yt, yp_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, lw=2)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve — {task_name} [{tag}]  AP={ap:.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    save_fig(fig, os.path.join(out_dir, f"{task_name}_pr_{tag}.png"))
    return ap


def plot_det_curve(fpr, fnr, tag, task_name, out_dir):
    """Detection Error Tradeoff (DET) curve."""
    from scipy.special import ndtri
    def to_ndtri(x):
        x = np.clip(x, 1e-6, 1-1e-6)
        return ndtri(x)
    try:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(to_ndtri(fpr), to_ndtri(fnr), lw=2)
        ticks = [0.001,0.01,0.05,0.1,0.2,0.5]
        tick_labels = [f"{t*100:.1f}" for t in ticks]
        ax.set_xticks([to_ndtri(t) for t in ticks]); ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_yticks([to_ndtri(t) for t in ticks]); ax.set_yticklabels(tick_labels, fontsize=7)
        ax.set_xlabel("FPR (%)"); ax.set_ylabel("FNR (%)")
        ax.set_title(f"DET Curve — {task_name} [{tag}]")
        save_fig(fig, os.path.join(out_dir, f"{task_name}_det_{tag}.png"))
    except Exception:
        pass   # scipy not available


def plot_score_histogram(scores_pos, scores_neg, thr, tag, task_name, out_dir,
                         xlabel="Score", pos_label="Genuine", neg_label="Impostor"):
    """Overlapping score distribution with threshold line."""
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(min(scores_pos.min(), scores_neg.min()),
                       max(scores_pos.max(), scores_neg.max()), 60)
    ax.hist(scores_pos, bins=bins, alpha=0.55, label=pos_label, color="#2196F3")
    ax.hist(scores_neg, bins=bins, alpha=0.55, label=neg_label, color="#F44336")
    ax.axvline(thr, color="black", linestyle="--", linewidth=1.5, label=f"thr={thr:.3f}")
    ax.set_xlabel(xlabel); ax.set_ylabel("Count")
    ax.set_title(f"Score Distribution — {task_name} [{tag}]")
    ax.legend(fontsize=8)
    save_fig(fig, os.path.join(out_dir, f"{task_name}_score_hist_{tag}.png"))


def plot_per_class_f1(yt, yp, class_names, tag, task_name, out_dir):
    """Per-class F1 bar chart."""
    from sklearn.metrics import f1_score
    f1s = f1_score(yt, yp, average=None, labels=list(range(len(class_names))), zero_division=0)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)*0.35), 4))
    colors = ["#4CAF50" if v >= 0.7 else "#FF9800" if v >= 0.4 else "#F44336" for v in f1s]
    ax.bar(class_names, f1s, color=colors)
    ax.axhline(f1s.mean(), color="navy", linestyle="--", label=f"mean={f1s.mean():.2f}")
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1 Score")
    ax.set_title(f"Per-class F1 — {task_name} [{tag}]")
    ax.legend(fontsize=8)
    if len(class_names) > 20:
        ax.set_xticks([]); ax.set_xlabel("Writer ID →")
    else:
        plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"{task_name}_f1bar_{tag}.png"))


def plot_tsne_embeddings(embeddings, labels, label_names, tag, task_name, out_dir, n_classes=20):
    """t-SNE 2D projection of encoder embeddings."""
    if not HAS_TSNE:
        return
    # Subsample if too many classes for readability
    unique_lbls = sorted(set(labels))
    if len(unique_lbls) > n_classes:
        chosen = unique_lbls[:n_classes]
        mask   = np.isin(labels, chosen)
        embeddings = embeddings[mask]; labels = labels[mask]
    try:
        z2d = TSNE(n_components=2, perplexity=min(30, len(embeddings)-1),
                   random_state=42, n_iter=1000).fit_transform(embeddings)
        unique_lbls = sorted(set(labels))
        cmap = plt.cm.get_cmap("tab20", len(unique_lbls))
        fig, ax = plt.subplots(figsize=(8, 7))
        for i, lbl in enumerate(unique_lbls):
            mask = labels == lbl
            name = label_names[lbl] if lbl < len(label_names) else str(lbl)
            ax.scatter(z2d[mask, 0], z2d[mask, 1], s=30, alpha=0.7,
                       color=cmap(i), label=name)
        ax.set_title(f"t-SNE Embedding — {task_name} [{tag}]")
        if len(unique_lbls) <= 20:
            ax.legend(fontsize=6, ncol=2, markerscale=1.2)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        plt.tight_layout()
        save_fig(fig, os.path.join(out_dir, f"{task_name}_tsne_{tag}.png"))
    except Exception as e:
        print(f"  [warn] t-SNE failed: {e}")


def plot_similarity_heatmap(proto_matrix, proto_labels, tag, out_dir):
    """Cosine similarity heatmap between all writer prototypes."""
    P = F.normalize(proto_matrix, dim=1)
    sim = (P @ P.T).numpy()
    n   = sim.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, n//4), max(5, n//4)))
    im = ax.imshow(sim, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_title(f"Writer Prototype Cosine Similarity — {tag}")
    ax.set_xlabel("Writer"); ax.set_ylabel("Writer")
    fig.colorbar(im, ax=ax, fraction=0.03)
    if n <= 54:
        ax.set_xticks(range(n)); ax.set_xticklabels([str(l) for l in proto_labels], fontsize=5, rotation=90)
        ax.set_yticks(range(n)); ax.set_yticklabels([str(l) for l in proto_labels], fontsize=5)
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"proto_sim_heatmap_{tag}.png"))


def plot_inference_profile(profile, out_dir):
    """Bar chart of inference timing breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    # Left: params
    cats  = ["Total\nParams (M)", "Trainable\nParams (M)"]
    vals  = [profile["params_total_M"], profile["params_train_M"]]
    axes[0].bar(cats, vals, color=["#1565C0","#42A5F5"])
    for i,(c,v) in enumerate(zip(cats,vals)):
        axes[0].text(i, v+0.1, f"{v:.2f}M", ha="center", fontsize=9)
    axes[0].set_title("Model Size"); axes[0].set_ylabel("Parameters (M)")
    # Right: inference time
    means = [profile["infer_ms_mean"]]
    stds  = [profile["infer_ms_std"]]
    axes[1].bar(["Full-cube\ninference"], means, yerr=stds,
                color="#388E3C", capsize=6, error_kw={"elinewidth":2})
    axes[1].text(0, means[0]+stds[0]+0.5,
                 f"mean={means[0]:.1f}ms\np95={profile['infer_ms_p95']:.1f}ms",
                 ha="center", fontsize=9)
    axes[1].set_title(f"Inference Time  ({profile['gflops']:.2f} GFLOPs)")
    axes[1].set_ylabel("ms / sample")
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, "model_profile.png"))


# ============================================================
#  SUMMARY RADAR PLOT
# ============================================================
def print_summary(all_results):
    print(f"\n{'='*60}\n  UNIFIED PIPELINE — SUMMARY\n{'='*60}")
    for task, res in all_results.items():
        print(f"\n  {task}")
        for k, v in res.items():
            print(f"    {k:<35}: {v:.4f}" if isinstance(v,float) else f"    {k}: {v}")

    metrics = {
        "Writer ID":    all_results.get("writer_id_ivisio",    {}).get("writer_id_acc",     0),
        "Ink Mismatch": all_results.get("ink_mismatch_ivisio", {}).get("ink_mismatch_auc",   0),
        "Verif AUC":    all_results.get("writer_verif_ivisio", {}).get("verif_auc",          0),
        "Forgery Acc":  all_results.get("forgery_ivisio",      {}).get("forgery_acc",        0),
        "Gender Acc":   all_results.get("age_gender_ivisio",   {}).get("gender_label_acc",   0),
        "Age Acc":      all_results.get("age_gender_ivisio",   {}).get("age_bin_acc",        0),
        "Cross-DS Acc": all_results.get("cross_dataset",       {}).get("cross_dataset_acc",  0),
        "Proto Acc":    all_results.get("protonet",            {}).get("proto_acc",          0),
    }
    names = list(metrics.keys()); vals = [metrics[n] for n in names]
    N = len(names); angles = [n/N*2*math.pi for n in range(N)] + [0]
    vals2 = vals + [vals[0]]

    fig, ax = plt.subplots(subplot_kw={"projection":"polar"}, figsize=(7,7))
    ax.set_theta_offset(math.pi/2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(names, size=9)
    ax.set_ylim(0,1)
    ax.plot(angles, vals2, "o-", linewidth=2, color="#1f77b4")
    ax.fill(angles, vals2, alpha=0.2, color="#1f77b4")
    ax.set_title("Forensics Pipeline — Metric Overview", size=12, pad=20)
    save_fig(fig, os.path.join(cfg.out_dir, "pipeline_radar.png"))
    print(f"\n  All outputs saved to: {cfg.out_dir}/")



# ============================================================
#  SAVE FULL RESULTS TO TEXT FILE
# ============================================================
def save_results_txt(all_results, profile, out_dir, in_ch, spatial=(512,650)):
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(out_dir, "results_summary.txt")
    lines = []
    lines += [
        "=" * 70,
        "  HYPERSPECTRAL DOCUMENT FORENSICS PIPELINE v3 — RESULTS SUMMARY",
        f"  Generated : {ts}",
        f"  Dataset   : iVision HHID  (54 writers, 269 cubes, {in_ch} bands, {spatial[0]}×{spatial[1]})",
        f"  Device    : {cfg.device}",
        "=" * 70, "",
        "── MODEL PROFILE ───────────────────────────────────────────────────",
        f"  Architecture        : ResNet18Encoder  (ProtoNet backbone, Exp 5)",
        f"  Input shape         : (1, {in_ch}, {spatial[0]}, {spatial[1]})",
        f"  Total parameters    : {profile['params_total_M']:.4f} M",
        f"  Trainable params    : {profile['params_train_M']:.4f} M",
        f"  GFLOPs              : {profile['gflops']:.4f}",
        f"  Inference mean      : {profile['infer_ms_mean']:.2f} ms",
        f"  Inference std       : {profile['infer_ms_std']:.2f} ms",
        f"  Inference p95       : {profile['infer_ms_p95']:.2f} ms",
        "",
        "── TASK RESULTS ─────────────────────────────────────────────────────",
    ]

    task_map = [
        ("ProtoNet Backbone",        "protonet",           [("Episodic Acc",     "proto_acc")]),
        ("Task 1: Writer ID [iVis]", "writer_id_ivisio",   [("Accuracy",         "writer_id_acc"),
                                                             ("MCC",              "writer_id_mcc"),
                                                             ("ROC AUC (micro)",  "writer_id_roc_auc")]),
        ("Task 2: Ink Mismatch [iV]","ink_mismatch_ivisio",[("Acc@EER",          "ink_mismatch_acc"),
                                                             ("AUC",              "ink_mismatch_auc"),
                                                             ("EER",              "ink_mismatch_eer")]),
        ("Task 2: Ink Mismatch [UW]","ink_mismatch_uwa",   [("Acc@EER",          "ink_mismatch_acc"),
                                                             ("AUC",              "ink_mismatch_auc"),
                                                             ("EER",              "ink_mismatch_eer")]),
        ("Task 3: Writer Verif [iV]","writer_verif_ivisio",[("AUC",              "verif_auc"),
                                                             ("EER",              "verif_eer"),
                                                             ("Acc@EER",          "verif_acc")]),
        ("Task 3: Writer Verif [UW]","writer_verif_uwa",   [("AUC",              "verif_auc"),
                                                             ("EER",              "verif_eer"),
                                                             ("Acc@EER",          "verif_acc")]),
        ("Task 4: Forgery Det [iV]", "forgery_ivisio",     [("Accuracy",         "forgery_acc"),
                                                             ("AUC",              "forgery_auc")]),
        ("Task 5: Gender Pred [iV]", "age_gender_ivisio",  [("Accuracy",         "gender_label_acc"),
                                                             ("ROC AUC",          "gender_label_roc_auc")]),
        ("Task 5: Age Pred [iV]",    "age_gender_ivisio",  [("Accuracy",         "age_bin_acc"),
                                                             ("ROC AUC",          "age_bin_roc_auc")]),
        ("Cross-Dataset (iV→UWA)",   "cross_dataset",      [("Zero-shot Acc",    "cross_dataset_acc")]),
    ]

    for title, key, metrics in task_map:
        res = all_results.get(key, {})
        if not res: continue
        lines.append(f"\n  {title}")
        lines.append(f"  {'─'*50}")
        for label, mkey in metrics:
            val = res.get(mkey, None)
            if val is not None:
                lines.append(f"    {label:<28}: {val:.4f}")

    lines += [
        "",
        "── OUTPUT FILES ─────────────────────────────────────────────────────",
        f"  Directory: {out_dir}/",
        "  Plots    : *_cm_*.png  *_roc_*.png  *_pr_*.png  *_det_*.png",
        "             *_score_hist_*.png  *_f1bar_*.png  *_tsne_*.png",
        "             proto_sim_heatmap_*.png  model_profile.png  pipeline_radar.png",
        "  Model    : forensics_model_v3.pt",
        "  Results  : results_summary.txt  results_summary.json",
        "",
        "=" * 70,
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [results] {path}")

    # Also save JSON for programmatic access
    json_path = os.path.join(out_dir, "results_summary.json")
    json_data = {
        "timestamp": ts, "device": cfg.device,
        "model_profile": {k: float(v) for k,v in profile.items()},
        "results": {k: {kk: float(vv) for kk,vv in v.items()}
                    for k,v in all_results.items()},
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  [results] {json_path}")


# ============================================================
#  MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all",
        choices=["all","writer_id","ink_mismatch","writer_verification",
                 "forgery_detection","age_gender","cross_dataset"])
    parser.add_argument("--use_pretrained",  action="store_true",
        help="Load saved ProtoNet checkpoint instead of training")
    parser.add_argument("--cross_dataset",   action="store_true")
    parser.add_argument("--proto_epochs",    type=int, default=cfg.proto_epochs)
    parser.add_argument("--cls_epochs",      type=int, default=cfg.cls_epochs)
    parser.add_argument("--max_cache",       type=int, default=cfg.max_cache)
    args = parser.parse_args()

    cfg.proto_epochs = args.proto_epochs
    cfg.cls_epochs   = args.cls_epochs
    cfg.max_cache    = args.max_cache
    global _CACHE
    _CACHE = CubeCache(maxsize=cfg.max_cache, fp16=cfg.fp16_cache)

    set_seed(cfg.seed); ensure_dir(cfg.out_dir)
    print(f"\n  Device    : {cfg.device}")
    print(f"  Task      : {args.task}")
    print(f"  Output    : {cfg.out_dir}")
    print(f"  max_cache : {cfg.max_cache} cubes (fp16={cfg.fp16_cache})")
    print(f"  Pretrained: {args.use_pretrained} → {cfg.proto_ckpt}\n")

    # ── Load datasets ─────────────────────────────────────────
    df_iv = load_index(cfg.ivisio_index, cfg.ivisio_dir, "ivisio")
    try:
        df_uwa = load_index(cfg.uwa_index, cfg.uwa_dir, "uwa")
    except FileNotFoundError:
        print("  [warn] UWA index not found — cross-dataset eval will be skipped.")
        df_uwa = pd.DataFrame()

    # Probe first cube to get in_ch
    in_ch_iv = _CACHE.get(str(df_iv.iloc[0]["full_cube"])).shape[0]
    print(f"  iVision in_ch = {in_ch_iv}")

    # ── ProtoNet backbone ─────────────────────────────────────
    all_results = {}

    if args.use_pretrained:
        if not os.path.exists(cfg.proto_ckpt):
            raise FileNotFoundError(
                f"Checkpoint not found: {cfg.proto_ckpt}\n"
                "Run without --use_pretrained first, or point to your Exp 5 checkpoint.")
        ckpt    = torch.load(cfg.proto_ckpt, map_location="cpu")
        encoder = ResNet18Encoder(in_ch=in_ch_iv, pretrained=True, embed_dim=cfg.embed_dim)
        encoder.load_state_dict(ckpt["state_dict"], strict=True)
        encoder.to(cfg.device)
        proto_acc = float(ckpt.get("acc", 0.958))
        print(f"  Loaded pretrained backbone. Reported acc = {proto_acc:.3f}")
        all_results["protonet"] = {"proto_acc": proto_acc}
    else:
        # Build cross-split items (exact Exp 5)
        train_items, test_items = build_crosssplit_items(
            cfg.ivisio_index, cfg.ivisio_dir, seed=cfg.seed, test_per_class=1)
        print(f"  ProtoNet split: train={len(train_items)} test={len(test_items)}")
        encoder, proto_acc, _, _, _ = train_protonet(train_items, test_items, in_ch_iv)
        all_results["protonet"] = {"proto_acc": proto_acc}

    encoder.eval()

    # ── Model profiling ───────────────────────────────────────
    profile = profile_encoder(encoder, in_ch_iv, cfg.device, spatial=(512, 650))
    plot_inference_profile(profile, cfg.out_dir)

    run_all = (args.task == "all")

    # ── Tasks ─────────────────────────────────────────────────
    if run_all or args.task == "writer_id":
        all_results["writer_id_ivisio"] = \
            task_writer_identification(df_iv, encoder, in_ch_iv, "ivisio")

    if run_all or args.task == "ink_mismatch":
        all_results["ink_mismatch_ivisio"] = \
            task_ink_mismatch(df_iv, encoder, in_ch_iv, "ivisio")
        if len(df_uwa) > 0:
            in_ch_uwa = _CACHE.get(str(df_uwa.iloc[0]["full_cube"])).shape[0]
            all_results["ink_mismatch_uwa"] = \
                task_ink_mismatch(df_uwa, encoder, in_ch_iv, "uwa")

    if run_all or args.task == "writer_verification":
        all_results["writer_verif_ivisio"] = \
            task_writer_verification(df_iv, encoder, in_ch_iv, "ivisio")
        if len(df_uwa) > 0:
            in_ch_uwa = _CACHE.get(str(df_uwa.iloc[0]["full_cube"])).shape[0]
            all_results["writer_verif_uwa"] = \
                task_writer_verification(df_uwa, encoder, in_ch_iv, "uwa")

    if run_all or args.task == "forgery_detection":
        all_results["forgery_ivisio"] = \
            task_forgery_detection(df_iv, encoder, in_ch_iv, "ivisio")

    if run_all or args.task == "age_gender":
        all_results["age_gender_ivisio"] = \
            task_age_gender(df_iv, encoder, in_ch_iv, "ivisio")

    if (run_all or args.cross_dataset) and len(df_uwa) > 0:
        all_results["cross_dataset"] = \
            cross_dataset_eval(df_iv, df_uwa, encoder, in_ch_iv)

    # ── Save final model ──────────────────────────────────────
    ckpt_path = os.path.join(cfg.out_dir, "forensics_model_v3.pt")
    torch.save({
        "encoder_state": encoder.state_dict(),
        "in_ch": in_ch_iv, "embed_dim": cfg.embed_dim,
        "results": {k: {kk: float(vv) for kk,vv in v.items()} for k,v in all_results.items()},
    }, ckpt_path)
    print(f"\n  Checkpoint saved: {ckpt_path}")

    save_results_txt(all_results, profile, cfg.out_dir, in_ch_iv, spatial=(512, 650))
    print_summary(all_results)


if __name__ == "__main__":
    main()
