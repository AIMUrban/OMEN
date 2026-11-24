import os
import math
import random
import logging
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import RidgeCV
from torch.utils.data import Dataset, DataLoader, Subset


USE_LOG_TRANSFORM = {"crime_rate"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str, tag: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{tag}_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger(tag)

@torch.no_grad()
def extract_features(model, rel_dict, device) -> np.ndarray:
    model.eval()
    out = model.encoder(rel_dict, indices=None)
    feat = torch.cat([out["h_local"], out["h_refined"]], dim=-1)
    return feat.detach().cpu().numpy()


def save_checkpoint(path, model, optimizer, scheduler, epoch, val_metric):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "val_metric": val_metric,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch"), ckpt.get("val_metric")


# ---------------------------------------------------------------------------
# Data preprocessing (Section 3.2: Anchor-Relative Modality Recoding)
# ---------------------------------------------------------------------------

def load_modality(path: str) -> dict:
    with open(path, "rb") as f:
        data = pickle.load(f)
    sample = next(iter(data.values()))
    if sample.ndim == 2:
        data = {k: v.mean(axis=0).astype(np.float32) for k, v in data.items()}
    elif sample.ndim == 1:
        data = {k: v.astype(np.float32) for k, v in data.items()}
    else:
        raise ValueError(f"Unexpected embedding shape: {sample.shape}")
    return data


def _farthest_point_sampling(embeddings: np.ndarray, n_anchors: int, random_state: int = 42) -> List[int]:
    np.random.seed(random_state)
    N = embeddings.shape[0]
    n_anchors = min(n_anchors, N)
    selected = [int(np.random.randint(N))]
    min_dists = np.full(N, np.inf)
    for _ in range(n_anchors - 1):
        last = embeddings[selected[-1]]
        new_dists = cdist(embeddings, last.reshape(1, -1), metric="cosine").squeeze()
        min_dists = np.minimum(min_dists, new_dists)
        selected.append(int(min_dists.argmax()))
    return selected

def _random_sampling(embeddings: np.ndarray, n_anchors: int, random_state: int = 42) -> List[int]:
    np.random.seed(random_state)
    N = embeddings.shape[0]
    N_array = [i for i in range(N)]
    n_anchors = min(n_anchors, N)
    selected = random.sample(N_array, n_anchors)

    return selected


def select_anchors_multimodal(
    modality_paths: Dict[str, str],
    n_anchors: int = 512,
    random_state: int = 42,
) -> Tuple[List[str], List[str]]:
    modality_data = {name: load_modality(path) for name, path in modality_paths.items()}
    region_ids = sorted(set.intersection(*[set(d.keys()) for d in modality_data.values()]))
    rows = []
    for rid in region_ids:
        vecs = []
        for name in modality_data:
            v = modality_data[name][rid]
            vecs.append(v / (np.linalg.norm(v) + 1e-8))
        rows.append(np.concatenate(vecs))
    joint = normalize(np.stack(rows).astype(np.float32), norm="l2")
    anchor_indices = _farthest_point_sampling(joint, n_anchors, random_state)
    anchors = list(dict.fromkeys([region_ids[i] for i in anchor_indices]))
    return anchors, region_ids


def compute_relative_representations(
    modality_paths: Dict[str, str],
    anchors: List[str],
    region_ids: List[str],
    tau: float = 1.0,
) -> Dict[str, np.ndarray]:
    relative_repr = {}
    for name, path in modality_paths.items():
        raw = load_modality(path)
        region_vecs = np.stack([raw[rid] for rid in region_ids])
        anchor_vecs = np.stack([raw[aid] for aid in anchors])
        region_norms = np.linalg.norm(region_vecs, axis=1, keepdims=True)
        anchor_norms = np.linalg.norm(anchor_vecs, axis=1, keepdims=True)
        region_normed = region_vecs / np.where(region_norms < 1e-8, 1.0, region_norms)
        anchor_normed = anchor_vecs / np.where(anchor_norms < 1e-8, 1.0, anchor_norms)
        dist_matrix = cdist(region_normed, anchor_normed, metric="euclidean")
        weights = np.exp(-dist_matrix / tau)
        weights /= weights.sum(axis=1, keepdims=True)
        relative_repr[name] = weights.astype(np.float32)
    return relative_repr


# ---------------------------------------------------------------------------
# Section 3.3.3 / Appendix A.3: Regional Context Refinement (region graph)
# ---------------------------------------------------------------------------

def construct_region_graph(
    s: torch.Tensor,
    k_mid: int = 16,
    k_min: int = 4,
    k_max: int = 64,
    calib_iters: int = 12,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = s.shape[0]
    sn = F.normalize(s, dim=-1)
    S = sn @ sn.t()
    eye = torch.eye(N, dtype=torch.bool, device=s.device)

    S_off = S.masked_fill(eye, float("-inf"))
    smax = S_off.max(dim=1).values
    mu = (S.sum(dim=1) - 1.0) / (N - 1)
    gap = (smax - mu).clamp_min(0.0)

    lo, hi = 0.0, 1.0
    for _ in range(calib_iters):
        beta = 0.5 * (lo + hi)
        cnt_med = (S_off >= (smax - beta * gap).unsqueeze(1)).sum(dim=1).float().median()
        if cnt_med < k_mid:
            lo = beta
        else:
            hi = beta
    beta = 0.5 * (lo + hi)
    thr = smax - beta * gap

    kth_lo = torch.topk(S_off, k=min(k_min, N - 1), dim=1).values[:, -1]
    thr = torch.minimum(thr, kth_lo)
    if k_max is not None and k_max < N - 1:
        kth_hi = torch.topk(S_off, k=k_max, dim=1).values[:, -1]
        thr = torch.maximum(thr, kth_hi)

    w = F.relu(S_off - thr.unsqueeze(1))
    keep = (S_off >= thr.unsqueeze(1)).float()
    w = w + eps * keep
    w = w / (w.sum(dim=1, keepdim=True) + eps)

    neighbor_counts = (keep > 0).sum(dim=1)
    return w, neighbor_counts


class RegionalContextRefinementLayer(nn.Module):
    """
    One gated message-passing layer over the region graph (Eq. 12, 26-27).

    The edge weights w still come from the region graph constructed by
    construct_region_graph(); this layer learns how much of the propagated
    neighbor evidence (the gate Gamma) should be injected into each region's
    representation, plus a position-wise FFN.
    """
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        hn = self.norm1(h)
        msg = torch.mm(w, self.value(hn))
        gate_val = self.gate(torch.cat([hn, msg], dim=-1))
        h = h + self.dropout(gate_val * msg)
        h = h + self.dropout(self.ffn(self.norm2(h)))
        return h


class ContentAdaptiveAggregator(nn.Module):
    """
    Content-adaptive profile aggregation (Eq. 7-11).

    Mean pooling is the uniform special case. This module learns modality-agnostic
    observation weights while keeping the readout as a normalized consensus:

        h_i = sum_m alpha_i^m q_i^m,   sum_m alpha_i^m = 1

    It is initialized to exactly uniform scores, so the model starts from mean.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        temperature: float = 1.0,
        init_uniform: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.scorer = nn.Sequential(
            nn.Linear(in_dim * 3 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        if init_uniform:
            with torch.no_grad():
                self.scorer[-1].weight.zero_()
                self.scorer[-1].bias.zero_()

    @staticmethod
    def entropy_from_centered(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        K = q.shape[-1]
        p = q / math.sqrt(K) + 1.0 / K
        p = p.clamp_min(eps)
        p = p / (p.sum(dim=-1, keepdim=True) + eps)
        return -(p * p.log()).sum(dim=-1, keepdim=True)

    @staticmethod
    def normalized_concentration(
            q: torch.Tensor,
            eps: float = 1e-9,
    ) -> torch.Tensor:
        """
        Compute the normalized information concentration of a probability
        distribution over anchors.

        Args:
            q: Raw softmax relative observations with shape [..., K].
               Each vector should be non-negative and sum to one.
            eps: Numerical stability constant.

        Returns:
            concentration: Tensor with shape [..., 1], bounded approximately
            in [0, 1].

            0 means that q is uniform and therefore uninformative.
            1 means that q is maximally concentrated on one anchor.
        """
        K = q.shape[-1]
        if K <= 1:
            return torch.zeros_like(q[..., :1])

        # q is already a probability distribution, but renormalize defensively
        # to avoid small numerical errors.
        p = q.clamp_min(eps)
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)

        entropy = -(p * p.log()).sum(dim=-1, keepdim=True)
        max_entropy = math.log(K)

        concentration = 1.0 - entropy / max_entropy
        return concentration.clamp(min=0.0, max=1.0)

    def forward(self, q_stack: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = q_stack.mean(dim=1, keepdim=True)
        concentration = self.normalized_concentration(q_stack)

        feat = torch.cat(
            [
                q_stack,
                q_stack - mu,
                q_stack * mu,
                concentration,
            ],
            dim=-1,
        )
        score = self.scorer(feat).squeeze(-1) / max(self.temperature, 1e-6)
        alpha = torch.softmax(score, dim=1)
        h = (alpha.unsqueeze(-1) * q_stack).sum(dim=1)
        return h, alpha


# ---------------------------------------------------------------------------
# OMEN encoder: f_theta (Section 3.3)
# ---------------------------------------------------------------------------

class OMENEncoder(nn.Module):
    """
    Anchor-relative profiles
      -> content-adaptive aggregation -> h_local  [N, K]           (Eq. 7-11)
      -> region graph construction + message passing -> h_refined [N, K]  (Eq. 12, 23-27)
      -> concat -> z [N, 2K]                                        (Eq. 13)

    Note: self.anchor_reweight below is a K -> K linear layer that is NOT
    part of the method described in the paper and is never applied in
    forward() (the call is commented out). It is kept only so that this
    module's parameter initialization order - and therefore the exact
    initial weights of every layer created after it - stays identical to
    the runs that produced the reported numbers. Do not remove it unless
    you intend to regenerate all reported results from scratch.

    anchor_reweight_init (inert, retained for the same reason):
        "identity" - W_a would be initialized to the identity matrix
        "xavier"   - W_a would use Xavier uniform initialization
    """

    def __init__(
        self,
        in_dim: int,
        dropout: float = 0.0,
        gnn_layers: int = 2,
        gnn_dropout: float = 0.1,
        graph_k_mid: int = 16,
        graph_k_min: int = 4,
        graph_k_max: int = 64,
        anchor_reweight_init: str = "identity",
        consensus_hidden: int = 128,
        consensus_temperature: float = 1.0,
    ):
        super().__init__()
        # assert anchor_reweight_init in ("identity", "xavier"), \
        #     f"anchor_reweight_init must be 'identity' or 'xavier', got {anchor_reweight_init}"
        self.in_dim = in_dim
        self.graph_k_mid = graph_k_mid
        self.graph_k_min = graph_k_min
        self.graph_k_max = graph_k_max

        self.aggregator = ContentAdaptiveAggregator(
            in_dim=in_dim,
            hidden_dim=consensus_hidden,
            dropout=dropout,
            temperature=consensus_temperature,
            init_uniform=True,
        )

        # Trainable projection applied before region graph construction (W_v).
        # It keeps the adaptive graph rule, but makes the similarity space trainable.
        self.graph_projection = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim, bias=False),
        )

        # Anchor reweighting layer: K -> K linear, no bias.
        # Unused in forward (see class docstring): kept only to preserve the
        # RNG-init order of all parameters created below.
        self.anchor_reweight = nn.Linear(in_dim, in_dim, bias=False)
        # self._init_anchor_reweight(anchor_reweight_init)

        self.norm = nn.LayerNorm(in_dim)
        self.context_layers = nn.ModuleList([
            RegionalContextRefinementLayer(in_dim, gnn_dropout)
            for _ in range(gnn_layers)
        ])
        self.z_norm = nn.LayerNorm(in_dim * 2)
        self.dropout = nn.Dropout(dropout)


    def forward(
        self,
        rel_dict: Dict[str, torch.Tensor],
        indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        modalities = list(rel_dict.keys())
        qs = []
        for m in modalities:
            x = rel_dict[m]                    # [N, K], softmax anchor-relative profile (Eq. 6)
            if indices is not None:
                x = x[indices]
            qs.append(x)  # [N, K]
        q_stack = torch.stack(qs, dim=1)       # [N, M, K]

        # Content-adaptive aggregation -> h_local (Eq. 7-11).
        # Uniform alpha recovers mean pooling.
        h, obs_alpha = self.aggregator(q_stack)  # [N, K], [N, M]
        h_local = self.dropout(self.norm(h))     # [N, K]

        # Region graph construction on a lightweight learnable projection (Eq. 23-25).
        graph_signal = self.graph_projection(h)

        w, neighbor_counts = construct_region_graph(
                graph_signal, k_mid=self.graph_k_mid, k_min=self.graph_k_min, k_max=self.graph_k_max,
            )

        # Regional context refinement: gated message passing (Eq. 12, 26-27).
        h_refined = h_local
        for layer in self.context_layers:
            h_refined = layer(h_refined, w)    # [N, K]

        z = self.z_norm(torch.cat([h_local, h_refined], dim=-1))   # Eq. 13, [N, 2K]

        return {
            "z": z,
            "h_local": h_local,
            "h_refined": h_refined,
            "q_stack": q_stack,
            "obs_alpha": obs_alpha,
            "modalities": modalities,
            "graph_w": w,
            "neighbor_counts": neighbor_counts,
        }


class OMENPretrainingWrapper(nn.Module):
    """Wraps OMENEncoder with the SSL projector used only during pretraining
    (Section 3.4.2, Eq. 15). The projector is discarded after pretraining."""

    def __init__(
        self,
        in_dim: int,
        proj_dim: int = 128,
        dropout: float = 0.0,
        gnn_layers: int = 2,
        gnn_dropout: float = 0.1,
        graph_k_mid: int = 16,
        graph_k_min: int = 4,
        graph_k_max: int = 64,
        anchor_reweight_init: str = "identity",
        consensus_hidden: int = 128,
        consensus_temperature: float = 1.0,
    ):
        super().__init__()
        self.encoder = OMENEncoder(
            in_dim, dropout,
            gnn_layers=gnn_layers, gnn_dropout=gnn_dropout,
            graph_k_mid=graph_k_mid, graph_k_min=graph_k_min, graph_k_max=graph_k_max,
            anchor_reweight_init=anchor_reweight_init,
            consensus_hidden=consensus_hidden,
            consensus_temperature=consensus_temperature,
        )
        self.projector = nn.Sequential(
            nn.Linear(in_dim * 2, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim, bias=False),
        )

    def encode_full(self, rel_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.encoder(rel_dict, indices=None)


# ---------------------------------------------------------------------------
# Section 3.4.1 / Appendix A.4: Modality-Composition Sampling
# (modality subset sampling only, no feature perturbation)
# ---------------------------------------------------------------------------

def _sample_subset(modalities: List[str], min_size: int = 1) -> List[str]:
    if len(modalities) <= min_size:
        return list(modalities)
    size = random.randint(min_size, len(modalities))
    return random.sample(modalities, size)


def sample_modality_composition_view(
    repr_tensors: Dict[str, torch.Tensor],
    modalities: List[str],
    min_view_modalities: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    Sample one modality-composition view via modality subset sampling only
    (Eq. 28-29). With probability 0.1, degrades to a single modality; with
    probability 0.1, uses the complete modality set; otherwise, sample a
    subset size uniformly from [min_view_modalities, |M|] and then sample
    that many modalities uniformly without replacement.
    """
    r = random.random()
    if r < 0.10:
        subset = random.sample(modalities, 1)
    elif r < 0.20:
        subset = list(modalities)
    else:
        subset = _sample_subset(modalities, min_view_modalities)

    return {m: repr_tensors[m] for m in subset}


# ---------------------------------------------------------------------------
# SSL losses
# ---------------------------------------------------------------------------

def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    sim_coeff: float = 1.0,
    var_coeff: float = 1.0,
    cov_coeff: float = 0.04,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    repr_loss = F.mse_loss(x, y)

    def std_loss(z):
        z = z - z.mean(dim=0)
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(F.relu(1.0 - std))

    def cov_loss(z):
        z = z - z.mean(dim=0)
        N = z.shape[0]
        if N <= 1:
            return torch.zeros((), device=z.device)
        cov = (z.T @ z) / (N - 1)
        return off_diagonal(cov).pow(2).sum() / z.shape[1]

    var_loss = 0.5 * (std_loss(x) + std_loss(y))
    cov      = 0.5 * (cov_loss(x) + cov_loss(y))
    total    = sim_coeff * repr_loss + var_coeff * var_loss + cov_coeff * cov
    return total, {
        "inv": float(repr_loss.detach()),
        "var": float(var_loss.detach()),
        "cov": float(cov.detach()),
    }


def retrieval_accuracy(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> float:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    acc1 = (logits.argmax(dim=1) == labels).float().mean()
    acc2 = (logits.t().argmax(dim=1) == labels).float().mean()
    return float((0.5 * (acc1 + acc2)).detach())


# ---------------------------------------------------------------------------
# Dataset / labels / metrics
# ---------------------------------------------------------------------------

class IndexDataset(Dataset):
    def __init__(self, n: int):
        self.n = n
    def __len__(self): return self.n
    def __getitem__(self, idx): return idx


class RegionLabelTable:
    def __init__(self, labels_csv: str, task_names: Sequence[str], region_ids: Sequence[str], label_key: str = "GEOID"):
        self.task_names = list(task_names)
        df = pd.read_csv(labels_csv, dtype={label_key: str})
        df = df.set_index(label_key).reindex(region_ids)
        targets, masks = [], []
        for rid in region_ids:
            row_vals, row_mask = [], []
            for task in task_names:
                try:
                    fv = float(df.loc[rid, task])
                except (ValueError, TypeError, KeyError):
                    fv = float("nan")
                if math.isnan(fv) or fv == 0.0:
                    row_vals.append(0.0)
                    row_mask.append(0.0)
                else:
                    row_vals.append(fv)
                    row_mask.append(1.0)
            targets.append(row_vals)
            masks.append(row_mask)
        self.targets    = np.asarray(targets, dtype=np.float32)
        self.masks      = np.asarray(masks,   dtype=np.float32)
        self.region_ids = list(region_ids)


def transform_y_train(y: np.ndarray, task: str) -> np.ndarray:
    return np.log1p(y) if task in USE_LOG_TRANSFORM else y


def inverse_transform_y(y: np.ndarray, task: str) -> np.ndarray:
    return np.expm1(y) if task in USE_LOG_TRANSFORM else y


def regression_metrics(pred: np.ndarray, tgt: np.ndarray) -> Dict[str, float]:
    if len(pred) < 2:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "count": len(pred)}
    return {
        "mae":   float(mean_absolute_error(tgt, pred)),
        "rmse":  float(np.sqrt(mean_squared_error(tgt, pred))),
        "r2":    float(r2_score(tgt, pred)),
        "count": int(len(pred)),
    }


# ---------------------------------------------------------------------------
# SSL training / validation
# ---------------------------------------------------------------------------

def _ssl_step_losses(
    model, repr_tensors, train_modalities, idx,
    min_view_modalities,
    inv_weight, var_weight, cov_weight,
):
    view1 = sample_modality_composition_view(repr_tensors, train_modalities, min_view_modalities)
    view2 = sample_modality_composition_view(repr_tensors, train_modalities, min_view_modalities)
    out1 = model.encode_full(view1)
    out2 = model.encode_full(view2)

    z1, z2 = out1["z"][idx], out2["z"][idx]
    p1, p2 = model.projector(z1), model.projector(z2)

    vic, vd = vicreg_loss(p1, p2, inv_weight, var_weight, cov_weight)

    nc = out1["neighbor_counts"].float()
    stats = {
        "vic": float(vic.detach()), "inv": vd["inv"], "var": vd["var"], "cov": vd["cov"],
        "nb_mean": float(nc.mean()), "nb_p10": float(torch.quantile(nc, 0.10)),
        "nb_p90": float(torch.quantile(nc, 0.90)),
        "retrieval": retrieval_accuracy(p1.detach(), p2.detach()),
    }
    return vic, stats


def train_ssl_epoch(
    model, loader, optimizer, repr_tensors, train_modalities, device,
    min_view_modalities=1,
    inv_weight=1.0, var_weight=1.0, cov_weight=0.04,
    gradient_clip=1.0,
) -> Dict[str, float]:
    model.train()
    keys = ["loss", "vic", "inv", "var", "cov", "nb_mean", "nb_p10", "nb_p90", "retrieval"]
    sums = {k: 0.0 for k in keys}
    sums["n"] = 0
    for batch in loader:
        idx = batch.to(device) if torch.is_tensor(batch) else torch.tensor(batch, device=device)
        loss, st = _ssl_step_losses(
            model, repr_tensors, train_modalities, idx,
            min_view_modalities,
            inv_weight, var_weight, cov_weight,
        )
        optimizer.zero_grad()
        loss.backward()
        if gradient_clip and gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        bsz = int(idx.numel())
        sums["loss"] += float(loss.detach()) * bsz
        for k in keys[1:]:
            sums[k] += st[k] * bsz
        sums["n"] += bsz
    n = max(sums.pop("n"), 1)
    return {k: v / n for k, v in sums.items()}


@torch.no_grad()
def build_fixed_val_views(
    repr_tensors: Dict[str, torch.Tensor],
    train_modalities: List[str],
    n_pairs: int,
    min_view_modalities: int,
    seed: int = 1234,
) -> List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
    py_state   = random.getstate()
    cpu_state  = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    random.seed(seed)
    torch.manual_seed(seed)
    pairs = []
    for _ in range(max(int(n_pairs), 1)):
        v1 = sample_modality_composition_view(repr_tensors, train_modalities, min_view_modalities)
        v2 = sample_modality_composition_view(repr_tensors, train_modalities, min_view_modalities)
        v1 = {k: v.detach().clone() for k, v in v1.items()}
        v2 = {k: v.detach().clone() for k, v in v2.items()}
        pairs.append((v1, v2))

    random.setstate(py_state)
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return pairs


@torch.no_grad()
def validate_ssl_fixed(
    model, val_view_pairs, val_idx, device,
    inv_weight=1.0, var_weight=1.0, cov_weight=0.04,
) -> Dict[str, float]:
    model.eval()
    idx = torch.as_tensor(np.asarray(val_idx), device=device, dtype=torch.long)
    keys = ["loss", "vic", "inv", "var", "cov", "nb_mean", "nb_p10", "nb_p90", "retrieval"]
    sums = {k: 0.0 for k in keys}
    for v1, v2 in val_view_pairs:
        out1 = model.encode_full(v1)
        out2 = model.encode_full(v2)
        z1, z2 = out1["z"][idx], out2["z"][idx]
        p1, p2 = model.projector(z1), model.projector(z2)
        vic, vd = vicreg_loss(p1, p2, inv_weight, var_weight, cov_weight)
        nc = out1["neighbor_counts"].float()
        sums["loss"]      += float(vic)
        sums["vic"]       += float(vic)
        sums["inv"]       += vd["inv"]
        sums["var"]       += vd["var"]
        sums["cov"]       += vd["cov"]
        sums["nb_mean"]   += float(nc.mean())
        sums["nb_p10"]    += float(torch.quantile(nc, 0.10))
        sums["nb_p90"]    += float(torch.quantile(nc, 0.90))
        sums["retrieval"] += retrieval_accuracy(p1, p2)
    npairs = max(len(val_view_pairs), 1)
    return {k: v / npairs for k, v in sums.items()}




# ---------------------------------------------------------------------------
# Downstream probe
# ---------------------------------------------------------------------------

def evaluate_downstream_ridge(
    z: np.ndarray,
    labels: RegionLabelTable,
    task_names: Sequence[str],
    kfold: int = 5,
    seed: int = 42,
    alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> Tuple[List[Dict], List[Dict]]:
    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
    fold_metrics, fold_alphas = [], []
    indices = np.arange(z.shape[0])
    for _, (train_idx, test_idx) in enumerate(kf.split(indices)):
        scaler   = StandardScaler()
        metrics  = {}
        alpha_used = {}
        for ti, task in enumerate(task_names):
            y  = labels.targets[:, ti]
            m  = labels.masks[:, ti].astype(bool)
            tr = train_idx[m[train_idx]]
            te = test_idx[m[test_idx]]
            if len(tr) < 5 or len(te) < 2:
                metrics[task]    = {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "count": len(te)}
                alpha_used[task] = float("nan")
                continue
            xtr = scaler.fit_transform(z[tr])
            xte = scaler.transform(z[te])
            y_train = transform_y_train(y[tr], task)
            probe   = RidgeCV(alphas=np.asarray(alphas, dtype=np.float64), fit_intercept=True)
            probe.fit(xtr, y_train)
            pred    = inverse_transform_y(probe.predict(xte), task)
            metrics[task]    = regression_metrics(pred, y[te])
            alpha_used[task] = float(probe.alpha_)
        fold_metrics.append(metrics)
        fold_alphas.append(alpha_used)
    return fold_metrics, fold_alphas


def summarize_probe_results(metrics3, metrics4, task_names: Sequence[str], logger, title: str):
    logger.info(f"\n{'='*110}\n{title}\n{'='*110}")
    logger.info(
        f"{'Task':<30} {'Initial R2':>10} {'Extended R2':>10} {'ΔR2':>8} | "
        f"{'Initial MAE':>10} {'Extended MAE':>10} {'ΔMAE':>8} | "
        f"{'Initial RMSE':>10} {'Extended RMSE':>10} {'ΔRMSE':>8}"
    )
    logger.info("─" * 110)

    deltas_r2, rel_dmae, rel_drmse = [], [], []
    m4_r2s, m4_maes, m4_rmses, m3_r2s, m3_maes, m3_rmses = [], [], [], [], [], []

    pos_r2 = 0

    for task in task_names:
        def _collect(metrics, key):
            vals = [fm[task][key] for fm in metrics
                    if task in fm and not math.isnan(fm[task].get(key, float("nan")))]
            return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))

        m3_r2,   s3_r2   = _collect(metrics3, "r2")
        m4_r2,   s4_r2   = _collect(metrics4, "r2")
        m3_mae,  s3_mae  = _collect(metrics3, "mae")
        m4_mae,  s4_mae  = _collect(metrics4, "mae")
        m3_rmse, s3_rmse = _collect(metrics3, "rmse")
        m4_rmse, s4_rmse = _collect(metrics4, "rmse")

        m4_r2s.append(m4_r2)
        m3_r2s.append(m3_r2)
        m4_maes.append(m4_mae)
        m3_maes.append(m3_mae)
        m4_rmses.append(m4_rmse)
        m3_rmses.append(m3_rmse)

        if math.isnan(m3_r2) or math.isnan(m4_r2):
            logger.info(f"{task:<30} {'N/A':>10} {'N/A':>10} {'N/A':>8} | {'N/A':>10} {'N/A':>10} {'N/A':>8} | {'N/A':>10} {'N/A':>10} {'N/A':>8}")
            continue

        dr2   = m4_r2   - m3_r2
        dmae  = m4_mae  - m3_mae
        drmse = m4_rmse - m3_rmse
        deltas_r2.append(dr2)
        if m3_mae > 1e-12:
            rel_dmae.append(100.0 * dmae / m3_mae)
        if m3_rmse > 1e-12:
            rel_drmse.append(100.0 * drmse / m3_rmse)
        pos_r2 += int(dr2 > 0)

        logger.info(
            f"{task:<30} "
            f"{m3_r2:>7.4f}±{s3_r2:<4.4f} {m4_r2:>7.4f}±{s4_r2:<4.4f} {dr2:>+8.4f} | "
            f"{m3_mae:>8.2f}±{s3_mae:<5.2f} {m4_mae:>8.2f}±{s4_mae:<5.2f} {dmae:>+8.2f} | "
            f"{m3_rmse:>8.2f}±{s3_rmse:<5.2f} {m4_rmse:>8.2f}±{s4_rmse:<5.2f} {drmse:>+8.2f}"
        )

    logger.info("─" * 110)
    logger.info(
        f"MEAN ΔR2={float(np.mean(deltas_r2)):+.4f} | "
        f"MEAN ΔMAE%={float(np.mean(rel_dmae)):+.2f}% | "
        f"MEAN ΔRMSE%={float(np.mean(rel_drmse)):+.2f}% | "
        f"positive_tasks(R2)={pos_r2}/{len(deltas_r2)}"
    )



