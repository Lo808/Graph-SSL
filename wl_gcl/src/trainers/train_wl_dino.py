from __future__ import annotations

import argparse
import copy
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from wl_gcl.configs.wl_dino import WLDinoConfig, make_wl_dino_cfg
from wl_gcl.src.augmentations.graph_augmentor import GraphAugmentor
from wl_gcl.src.contrastive.dual_view_miner import DualViewMiner
from wl_gcl.src.data_loader.dataset import load_dataset
from wl_gcl.src.models import get_model
from wl_gcl.src.trainers.eval import evaluate_linear_probe
from wl_gcl.src.utils.wl_core import WLHierarchyEngine


class ProjectionHead(nn.Module):
    """
    DINO-style projection MLP: d -> 2d -> d + normalization.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class PredictorHead(nn.Module):
    """
    BYOL-style predictor MLP: d -> 2d -> d.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PrototypeHead(nn.Module):
    """
    Shared global prototypes for DINO distillation.
    """

    def __init__(self, dim: int, num_prototypes: int, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        self.weight = nn.Parameter(torch.randn(num_prototypes, dim) * 0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if self.normalize:
            w = F.normalize(w, p=2, dim=-1)
        return torch.matmul(z, w.t())


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(requires_grad)


@torch.no_grad()
def _ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    for p_t, p_s in zip(teacher.parameters(), student.parameters()):
        p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)


def _build_wl_engine(
    data,
    wl_depth: int,
    force_convergence: bool = False,
    early_stop: bool = False,
) -> WLHierarchyEngine:
    nodes = list(range(data.num_nodes))
    edges = data.edge_index.t().tolist()

    engine = WLHierarchyEngine(nodes, edges)
    engine.build_wl_tree(
        max_iterations=wl_depth,
        force_convergence=force_convergence,
        early_stop=early_stop,
    )
    return engine


def _extract_level_ids(engine: WLHierarchyEngine, wl_depth: int) -> Tuple[List[torch.Tensor], List[Dict[int, List[int]]]]:
    n = len(engine.nodes)
    level_ids: List[torch.Tensor] = []
    members_by_level: List[Dict[int, List[int]]] = []

    for t in range(wl_depth + 1):
        cid2idx: Dict[str, int] = {}
        ids_t = torch.empty(n, dtype=torch.long)

        for i, node in enumerate(engine.nodes):
            cid = engine.get_cluster_id(node, t)
            if cid is None:
                # With early_stop=False and fixed depth this should not happen,
                # but keep a stable fallback for robustness.
                cid = f"missing_level_{t}"
            if cid not in cid2idx:
                cid2idx[cid] = len(cid2idx)
            ids_t[i] = cid2idx[cid]

        level_ids.append(ids_t)

        group: Dict[int, List[int]] = {}
        for i, c in enumerate(ids_t.tolist()):
            group.setdefault(c, []).append(i)
        members_by_level.append(group)

    return level_ids, members_by_level


def _pair_wl_distance(level_ids: Sequence[torch.Tensor], u: int, v: int, wl_depth: int) -> int:
    for t in range(wl_depth, -1, -1):
        if int(level_ids[t][u]) == int(level_ids[t][v]):
            return wl_depth - t
    return wl_depth


def _precompute_topk_wl_neighbors(
    level_ids: Sequence[torch.Tensor],
    members_by_level: Sequence[Dict[int, List[int]]],
    k_wl: int,
    wl_depth: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = level_ids[0].numel()
    k = max(0, min(k_wl, n - 1))

    if k == 0:
        return (
            torch.empty((n, 0), dtype=torch.long),
            torch.empty((n, 0), dtype=torch.float32),
        )

    neighbors = torch.empty((n, k), dtype=torch.long)
    distances = torch.empty((n, k), dtype=torch.float32)

    for i in range(n):
        selected: List[int] = []
        seen = {i}

        for t in range(wl_depth, -1, -1):
            cid = int(level_ids[t][i])
            for j in members_by_level[t].get(cid, []):
                if j in seen:
                    continue
                seen.add(j)
                selected.append(j)
                if len(selected) >= k:
                    break
            if len(selected) >= k:
                break

        if len(selected) < k:
            for j in range(n):
                if j in seen:
                    continue
                seen.add(j)
                selected.append(j)
                if len(selected) >= k:
                    break

        neighbors[i] = torch.tensor(selected, dtype=torch.long)
        distances[i] = torch.tensor(
            [_pair_wl_distance(level_ids, i, j, wl_depth) for j in selected],
            dtype=torch.float32,
        )

    return neighbors, distances


@torch.no_grad()
def _compute_topk_feature_neighbors(
    z: torch.Tensor,
    k_feat: int,
    chunk_size: int,
) -> torch.Tensor:
    """
    Chunked cosine top-k neighbors, avoiding full N x N materialization.
    """
    z = F.normalize(z, p=2, dim=-1)
    n = z.size(0)
    k = max(0, min(k_feat, n - 1))

    if k == 0:
        return torch.empty((n, 0), dtype=torch.long)

    out = torch.empty((n, k), dtype=torch.long, device="cpu")

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = z[start:end]  # [B, d]
        sim = torch.matmul(chunk, z.t())  # [B, N]

        row = torch.arange(end - start, device=z.device)
        col = torch.arange(start, end, device=z.device)
        sim[row, col] = -1e9

        idx = torch.topk(sim, k=k, dim=1).indices
        out[start:end] = idx.cpu()

    return out


def _build_candidate_matrix(
    wl_neighbors: torch.Tensor,
    feat_neighbors: torch.Tensor,
    max_candidates: int,
    num_random_neg: int,
    seed: int,
) -> torch.Tensor:
    n = wl_neighbors.size(0)
    target_k = max(0, min(max_candidates, n - 1))
    if target_k == 0:
        return torch.empty((n, 0), dtype=torch.long)

    rng = random.Random(seed)
    out = torch.empty((n, target_k), dtype=torch.long)

    for i in range(n):
        picked: List[int] = []
        seen = {i}

        for j in wl_neighbors[i].tolist():
            if j in seen:
                continue
            seen.add(j)
            picked.append(j)
            if len(picked) >= target_k:
                break

        if len(picked) < target_k:
            for j in feat_neighbors[i].tolist():
                if j in seen:
                    continue
                seen.add(j)
                picked.append(j)
                if len(picked) >= target_k:
                    break

        random_budget = max(0, num_random_neg)
        while len(picked) < target_k and random_budget > 0:
            j = rng.randrange(n)
            if j in seen:
                continue
            seen.add(j)
            picked.append(j)
            random_budget -= 1

        if len(picked) < target_k:
            for j in range(n):
                if j in seen:
                    continue
                seen.add(j)
                picked.append(j)
                if len(picked) >= target_k:
                    break

        out[i] = torch.tensor(picked, dtype=torch.long)

    return out


def _sample_dual_view_partners(
    positives: Sequence[Sequence[int]],
    seed: int,
) -> Tuple[torch.Tensor, float]:
    """
    Sample one mined positive per anchor.
    Fallback to self when no dual-view positive is available.
    """
    rng = random.Random(seed)
    n = len(positives)
    partners = torch.arange(n, dtype=torch.long)
    with_pos = 0

    for i, pos in enumerate(positives):
        if len(pos) == 0:
            continue
        with_pos += 1
        partners[i] = int(pos[rng.randrange(len(pos))])

    coverage = float(with_pos) / float(max(1, n))
    return partners, coverage


def _sample_wl_naive_partners(
    batch_nodes: torch.Tensor,
    level_ids_t: torch.Tensor,
    members_t: Dict[int, List[int]],
    seed: int,
    level_t: int,
    wl_depth: int,
    sample_mode: str = "uniform",
    anchor_repr: torch.Tensor | None = None,
    pair_temperature: float = 0.1,
    uniform_alpha: float = 1.0,
    wl_distance_beta: float = 1.0,
    level_ids_stack: torch.Tensor | None = None,
    distance_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] | None = None,
    debug: bool = False,
    epoch: int | None = None,
    batch_idx: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample one positive u for each anchor v uniformly from the same WL cluster
    at the selected level.

    Nodes with singleton clusters are skipped.
    Returns:
      - valid_anchor_idx: global node ids for anchors with a valid positive
      - positive_idx: sampled global node ids of their paired positives
    """
    rng = random.Random(seed)
    batch_list = batch_nodes.detach().cpu().tolist()
    valid_anchor: List[int] = []
    positives: List[int] = []

    for v in batch_list:
        cid = int(level_ids_t[v])
        group = members_t.get(cid, [v])
        assert len(group) > 0, f"Empty WL cluster for node {v} at cid={cid}"
        if len(group) <= 1:
            if debug:
                print(
                    f"[WL-NAIVE][Epoch {epoch:03d}][Batch {batch_idx:03d}] "
                    f"Skipped v: {v} cluster size: {len(group)}"
                )
            continue

        candidates = [u for u in group if u != v]
        if len(candidates) == 0:
            if debug:
                print(
                    f"[WL-NAIVE][Epoch {epoch:03d}][Batch {batch_idx:03d}] "
                    f"Skipped v: {v} cluster size: {len(group)}"
                )
            continue

        if sample_mode == "uniform":
            u = int(candidates[rng.randrange(len(candidates))])
        elif sample_mode in {"feature_softmax", "hybrid"}:
            if anchor_repr is None:
                raise ValueError("anchor_repr is required for feature-aware WL sampling")
            h_v = F.normalize(anchor_repr[v].detach(), dim=-1)
            idx = torch.tensor(candidates, device=anchor_repr.device, dtype=torch.long)
            h_u = F.normalize(anchor_repr.index_select(0, idx).detach(), dim=-1)
            sim = torch.matmul(h_u, h_v)  # [|C_t(v)|-1]
            tau = max(1e-6, float(pair_temperature))
            prob_soft = F.softmax(sim / tau, dim=0)
            if sample_mode == "hybrid":
                alpha = min(max(float(uniform_alpha), 0.0), 1.0)
                prob_uniform = torch.full_like(prob_soft, 1.0 / float(len(candidates)))
                prob = alpha * prob_uniform + (1.0 - alpha) * prob_soft
            else:
                prob = prob_soft
            weights = prob.detach().cpu().tolist()
            u = int(rng.choices(candidates, weights=weights, k=1)[0])
        elif sample_mode == "wl_distance":
            if level_ids_stack is None:
                raise ValueError("level_ids_stack is required for sample_mode='wl_distance'")

            cache_key = (int(level_t), int(v))
            cached = distance_cache.get(cache_key) if distance_cache is not None else None
            if cached is None:
                cand_tensor = torch.tensor(candidates, dtype=torch.long)
                # Vectorized d_t(u, v):
                # smallest k >= 0 such that c^{t+k}(u) != c^{t+k}(v),
                # else d_t = T - t.
                level_slice = level_ids_stack[level_t : wl_depth + 1]  # [L, N]
                anchor_path = level_slice[:, v].unsqueeze(1)  # [L, 1]
                cand_path = level_slice.index_select(1, cand_tensor)  # [L, C]
                neq = cand_path.ne(anchor_path)  # [L, C]
                has_neq = neq.any(dim=0)  # [C]
                first_neq = neq.to(torch.int64).argmax(dim=0)  # [C]
                fallback = torch.full_like(first_neq, fill_value=(wl_depth - level_t))
                dist_t = torch.where(has_neq, first_neq, fallback)  # [C]
                if distance_cache is not None:
                    distance_cache[cache_key] = (cand_tensor, dist_t)
            else:
                cand_tensor, dist_t = cached

            logits = float(wl_distance_beta) * dist_t.to(torch.float32)
            prob = F.softmax(logits, dim=0)
            pick = int(rng.choices(range(cand_tensor.numel()), weights=prob.tolist(), k=1)[0])
            u = int(cand_tensor[pick].item())
        else:
            raise ValueError(f"Unknown wl naive sampling mode: {sample_mode}")

        if debug:
            print(
                f"[WL-NAIVE][Epoch {epoch:03d}][Batch {batch_idx:03d}] "
                f"Sampled pair: {v} {u}"
            )
        valid_anchor.append(v)
        positives.append(u)

    return (
        torch.tensor(valid_anchor, dtype=torch.long, device=batch_nodes.device),
        torch.tensor(positives, dtype=torch.long, device=batch_nodes.device),
    )


def _wl_distances_for_candidates(
    level_ids_device: Sequence[torch.Tensor],
    anchors: torch.Tensor,
    candidates: torch.Tensor,
    wl_depth: int,
) -> torch.Tensor:
    """
    Compute d_WL(u, v) for each row u in anchors and row-wise candidates v.
    """
    bsz, k = candidates.shape
    shared = torch.full((bsz, k), -1, dtype=torch.long, device=anchors.device)

    flat_candidates = candidates.reshape(-1)
    for t in range(wl_depth, -1, -1):
        a = level_ids_device[t].index_select(0, anchors).unsqueeze(1)  # [B, 1]
        c = level_ids_device[t].index_select(0, flat_candidates).view(bsz, k)  # [B, K]
        eq = a.eq(c)
        write_mask = (shared < 0) & eq
        shared = torch.where(write_mask, torch.full_like(shared, t), shared)

    dist = (wl_depth - shared.clamp(min=0)).to(torch.float32)
    return dist


def _combine_losses(
    objective: str,
    loss_distill: torch.Tensor,
    loss_wl: torch.Tensor,
    lambda_wl: float,
) -> torch.Tensor:
    mode = objective.strip().lower()
    if mode in {"dino", "byol", "bgrl", "bgrl_wl_naive"}:
        return loss_distill
    if mode in {"bgrl_wl_cls", "bgrl_wl_naive_cls"}:
        return loss_distill + lambda_wl * loss_wl
    if mode == "wl":
        return loss_wl
    if mode == "full":
        return loss_distill + lambda_wl * loss_wl
    raise ValueError(
        "objective must be one of: dino, byol, bgrl, bgrl_wl_naive, bgrl_wl_cls, bgrl_wl_naive_cls, wl, full"
    )


def _adaptive_wl_scale(
    *,
    loss_distill: torch.Tensor,
    loss_wl: torch.Tensor,
    distill_ema: float | None,
    wl_ema: float | None,
    momentum: float,
    eps: float,
    min_scale: float,
    max_scale: float,
) -> tuple[float, float, float]:
    """
    Compute adaptive WL scaling from EMA(distill) / EMA(wl).
    Returns: (scale, updated_distill_ema, updated_wl_ema)
    """
    distill_value = float(loss_distill.detach().item())
    wl_value = float(loss_wl.detach().item())

    if distill_ema is None or wl_ema is None:
        distill_ema = distill_value
        wl_ema = wl_value
    else:
        distill_ema = momentum * distill_ema + (1.0 - momentum) * distill_value
        wl_ema = momentum * wl_ema + (1.0 - momentum) * wl_value

    raw_scale = distill_ema / max(wl_ema, eps)
    scale = float(min(max(raw_scale, min_scale), max_scale))
    return scale, distill_ema, wl_ema


def _byol_regression_loss(p: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """
    BYOL regression loss:
        2 - 2 * cos_sim(p, z_target)
    """
    p = F.normalize(p, dim=-1)
    z_target = F.normalize(z_target.detach(), dim=-1)
    return 2.0 - 2.0 * (p * z_target).sum(dim=-1).mean()


def _wl_alignment_loss(
    anchor: torch.Tensor,
    neighbors: torch.Tensor,
    wl_dist: torch.Tensor,
    tau_wl: float,
    use_repulsion: bool,
    repulsion_beta: float,
    repulsion_threshold: float,
) -> torch.Tensor:
    """
    WL-weighted geometric alignment:
        sum_u w(u,v) * ||z_v - z_u||^2
    with optional weak repulsion for far WL nodes.
    """
    # [B, K]
    dist2 = ((neighbors - anchor.unsqueeze(1)) ** 2).sum(dim=-1)
    weights = torch.exp(-wl_dist / tau_wl)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    loss_align = (weights * dist2).sum(dim=1).mean()

    if not use_repulsion:
        return loss_align

    sim = torch.einsum("bd,bkd->bk", anchor, neighbors)
    neg_mask = wl_dist > repulsion_threshold

    if not torch.any(neg_mask):
        return loss_align

    repulse = torch.clamp(sim, min=0.0)
    repulse = (repulse * neg_mask.float()).sum() / neg_mask.float().sum().clamp_min(1.0)
    return loss_align + repulsion_beta * repulse


def _teacher_temperature(epoch: int, cfg: WLDinoConfig) -> float:
    """
    DINO-style teacher temperature schedule (linear warmup then constant).
    """
    warmup_epochs = max(0, int(cfg.tau_t_warmup_epochs))
    if warmup_epochs == 0:
        return float(cfg.tau_t)

    start = float(cfg.tau_t_warmup_start)
    target = float(cfg.tau_t)

    if epoch >= warmup_epochs:
        return target

    alpha = float(epoch) / float(warmup_epochs)
    return start + (target - start) * alpha


def _wl_naive_curriculum_level(epoch: int, total_epochs: int, wl_depth: int) -> int:
    """
    Uniform curriculum over WL depths:
      levels 1..T are visited uniformly across epochs.
    """
    if wl_depth <= 0:
        return 0
    e = max(1, int(epoch))
    n = max(1, int(total_epochs))
    return min(wl_depth, ((e - 1) * wl_depth) // n + 1)


def _wl_naive_pair_temperature(
    level: int,
    wl_depth: int,
    tau_start: float,
    tau_end: float,
) -> float:
    """
    Pair-sampling temperature schedule aligned with WL depth curriculum.
    level=1 -> tau_start, level=wl_depth -> tau_end.
    """
    if wl_depth <= 1:
        return float(tau_end)
    lvl = max(1, min(int(level), int(wl_depth)))
    alpha = float(lvl - 1) / float(max(1, wl_depth - 1))
    return float(tau_start) + (float(tau_end) - float(tau_start)) * alpha


def _wl_naive_uniform_alpha(
    epoch: int,
    total_epochs: int,
    start_frac: float,
    end_alpha: float,
) -> float:
    """
    Epoch schedule for hybrid WL pair sampling:
      - alpha=1.0 until start_frac of training.
      - then linear decay to end_alpha by final epoch.
    """
    n = max(1, int(total_epochs))
    if n <= 1:
        return float(end_alpha)

    e = max(1, int(epoch))
    progress = float(e - 1) / float(n - 1)
    s = min(max(float(start_frac), 0.0), 0.99)
    end = min(max(float(end_alpha), 0.0), 1.0)

    if progress <= s:
        return 1.0

    beta = (progress - s) / max(1e-12, (1.0 - s))
    return 1.0 + beta * (end - 1.0)


def _parse_wl_cls_levels(level_spec: str, wl_depth: int) -> List[int]:
    """
    Parse WL classification levels from config.
    Accepted forms:
      - "all" -> [1, 2, ..., wl_depth]
      - comma-separated integers, e.g. "1,2,4"
    """
    if wl_depth <= 0:
        return []
    raw = (level_spec or "all").strip().lower()
    if raw == "all":
        return list(range(1, wl_depth + 1))

    out: List[int] = []
    seen = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            t = int(tok)
        except ValueError:
            continue
        if 1 <= t <= wl_depth and t not in seen:
            seen.add(t)
            out.append(t)
    out.sort()
    return out


def _wl_cls_alpha_weights(levels: Sequence[int], scheme: str) -> Dict[int, float]:
    """
    Build normalized alpha_t weights for WL classification levels.
    """
    lv = list(levels)
    if len(lv) == 0:
        return {}

    mode = scheme.strip().lower()
    if mode == "uniform":
        raw = {t: 1.0 for t in lv}
    elif mode == "deeper_more":
        raw = {t: float(t) for t in lv}
    elif mode == "shallower_more":
        max_t = max(lv)
        raw = {t: float(max_t - t + 1) for t in lv}
    else:
        raise ValueError("wl_cls_alpha_scheme must be one of: uniform, deeper_more, shallower_more")

    z = sum(raw.values())
    if z <= 0:
        return {t: 1.0 / float(len(lv)) for t in lv}
    return {t: raw[t] / z for t in lv}


def train_wl_dino(cfg: WLDinoConfig) -> Dict[str, float]:
    device = torch.device(cfg.device)

    dataset = load_dataset(cfg.dataset)
    data = dataset.data.to(device)
    num_nodes = int(data.num_nodes)
    objective_mode = cfg.objective.strip().lower()
    if objective_mode not in {
        "dino",
        "byol",
        "bgrl",
        "bgrl_wl_naive",
        "bgrl_wl_cls",
        "bgrl_wl_naive_cls",
        "wl",
        "full",
    }:
        raise ValueError(
            "objective must be one of: dino, byol, bgrl, bgrl_wl_naive, bgrl_wl_cls, bgrl_wl_naive_cls, wl, full"
        )
    need_byol = objective_mode == "byol"
    need_bgrl = objective_mode == "bgrl"
    need_bgrl_wl_naive = objective_mode == "bgrl_wl_naive"
    need_bgrl_wl_cls = objective_mode == "bgrl_wl_cls"
    need_bgrl_wl_naive_cls = objective_mode == "bgrl_wl_naive_cls"
    need_wl_pairs = need_bgrl_wl_naive or need_bgrl_wl_naive_cls
    need_wl_cls_obj = need_bgrl_wl_cls or need_bgrl_wl_naive_cls
    # bgrl_wl_naive follows the same student/teacher bootstrap path as BGRL.
    need_bootstrap = need_byol or need_bgrl or need_wl_pairs or need_wl_cls_obj
    wl_naive_pair_sampling = cfg.wl_naive_pair_sampling.strip().lower()
    if wl_naive_pair_sampling not in {"uniform", "feature_softmax", "hybrid", "wl_distance"}:
        raise ValueError(
            "wl_naive_pair_sampling must be one of: uniform, feature_softmax, hybrid, wl_distance"
        )

    wl_depth_eff = int(cfg.wl_depth)
    if cfg.use_max_wl_depth:
        probe_engine = _build_wl_engine(
            data,
            wl_depth=1,
            force_convergence=True,
            early_stop=True,
        )
        wl_depth_eff = max(1, int(probe_engine.max_depth))
        print(
            f"[WL-DINO] use_max_wl_depth=True -> "
            f"resolved_wl_depth={wl_depth_eff} (cfg.wl_depth={cfg.wl_depth})"
        )

    wl_engine = _build_wl_engine(
        data,
        wl_depth=wl_depth_eff,
        force_convergence=False,
        early_stop=False,
    )
    level_ids, members_by_level = _extract_level_ids(wl_engine, wl_depth=wl_depth_eff)
    wl_neighbors, _wl_neighbor_distances = _precompute_topk_wl_neighbors(
        level_ids=level_ids,
        members_by_level=members_by_level,
        k_wl=cfg.k_wl,
        wl_depth=wl_depth_eff,
    )

    level_ids_device = [ids.to(device) for ids in level_ids]
    level_ids_stack_cpu = torch.stack(level_ids, dim=0).to(torch.long)
    wl_naive_distance_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    student = get_model(
        name=cfg.model,
        input_dim=dataset.num_features,
        hidden_dim=cfg.hidden_dim,
        out_dim=cfg.out_dim,
        dropout=cfg.dropout,
        tau=cfg.tau,
        num_layers=cfg.num_layers,
        heads=cfg.heads,
    ).to(device)

    teacher = copy.deepcopy(student).to(device)
    _set_requires_grad(teacher, False)
    teacher.eval()

    proj_student = ProjectionHead(cfg.out_dim).to(device)
    proj_teacher = copy.deepcopy(proj_student).to(device)
    _set_requires_grad(proj_teacher, False)
    proj_teacher.eval()

    distill_space = cfg.distill_space.strip().lower()
    if distill_space not in {"candidate", "prototype"}:
        raise ValueError("distill_space must be one of: candidate, prototype")
    if cfg.use_dual_view_miner_pairs and (not need_bootstrap) and distill_space != "prototype":
        raise ValueError("Dual-view mined pairs currently support distill_space='prototype' only.")

    miner = None
    if cfg.use_dual_view_miner_pairs and (not need_wl_pairs) and (not need_wl_cls_obj):
        miner = DualViewMiner(
            wl_engine=wl_engine,
            nodes_list=list(range(num_nodes)),
            theta=cfg.miner_theta,
            delta=cfg.miner_delta,
        )

    pred_student = None
    if need_bootstrap:
        pred_student = PredictorHead(cfg.out_dim).to(device)

    proto_student = None
    proto_teacher = None
    if (not need_bootstrap) and distill_space == "prototype":
        proto_student = PrototypeHead(
            dim=cfg.out_dim,
            num_prototypes=cfg.num_prototypes,
            normalize=True,
        ).to(device)
        proto_teacher = copy.deepcopy(proto_student).to(device)
        _set_requires_grad(proto_teacher, False)
        proto_teacher.eval()

    sup_head = None
    if cfg.lambda_sup > 0.0:
        sup_head = nn.Linear(cfg.out_dim, dataset.num_classes).to(device)

    wl_cls_levels_active: List[int] = []
    wl_cls_alpha: Dict[int, float] = {}
    wl_cls_labels: Dict[int, torch.Tensor] = {}
    wl_cls_heads: nn.ModuleDict | None = None
    if need_wl_cls_obj:
        requested_levels = _parse_wl_cls_levels(cfg.wl_cls_levels, wl_depth=wl_depth_eff)
        wl_cls_heads = nn.ModuleDict()
        for t in requested_levels:
            num_classes_t = int(level_ids[t].max().item()) + 1
            if num_classes_t <= 1:
                continue
            wl_cls_heads[str(t)] = nn.Linear(cfg.out_dim, num_classes_t).to(device)
            wl_cls_levels_active.append(t)
            wl_cls_labels[t] = level_ids_device[t]

        wl_cls_alpha = _wl_cls_alpha_weights(
            wl_cls_levels_active,
            scheme=cfg.wl_cls_alpha_scheme,
        )
        # WL is used here as hierarchical pseudo-label supervision, not as positive-pair sampling.
        if len(wl_cls_levels_active) == 0:
            print("[WL-DINO] bgrl_wl_cls: no valid WL levels (>1 class). WL classification term will be zero.")

    augmentor = GraphAugmentor(
        edge_drop_prob=cfg.drop_edge_prob,
        feature_mask_prob=cfg.feature_mask_prob,
    )

    optim_params = list(student.parameters()) + list(proj_student.parameters())
    if pred_student is not None:
        optim_params += list(pred_student.parameters())
    if proto_student is not None:
        optim_params += list(proto_student.parameters())
    if sup_head is not None:
        optim_params += list(sup_head.parameters())
    if wl_cls_heads is not None:
        optim_params += list(wl_cls_heads.parameters())

    optimizer = Adam(
        optim_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs) if cfg.scheduler else None

    proto_info = (
        f" | num_prototypes={cfg.num_prototypes}"
        if ((not need_bootstrap) and distill_space == "prototype")
        else ""
    )
    wl_loss_type = cfg.wl_loss_type.strip().lower()
    if wl_loss_type not in {"align", "kl"}:
        raise ValueError("wl_loss_type must be one of: align, kl")
    print(
        f"\n[WL-DINO] Dataset={cfg.dataset.upper()} | Model={cfg.model.upper()} | "
        f"objective={cfg.objective} | distill_space={distill_space}{proto_info} | "
        f"dual_view_pairs={cfg.use_dual_view_miner_pairs} | "
        f"wl_loss_type={wl_loss_type} | "
        f"wl_naive_pair_sampling={wl_naive_pair_sampling} | "
        f"wl_naive_distance_beta={cfg.wl_naive_distance_beta:.3f} | "
        f"adaptive_wl_balance={cfg.use_adaptive_wl_balance} | "
        f"lambda_sup={cfg.lambda_sup} | "
        f"epochs={cfg.epochs} | device={cfg.device}\n"
    )
    if need_wl_cls_obj:
        level_text = ",".join(str(t) for t in wl_cls_levels_active) if wl_cls_levels_active else "none"
        print(
            "[WL-DINO] WL-cls settings | "
            f"levels={level_text} | "
            f"alpha_scheme={cfg.wl_cls_alpha_scheme} | "
            f"lambda_wl={cfg.lambda_wl}"
        )

    best_acc = 0.0
    best_state = None
    best_head_state = None
    best_pred_state = None
    best_proto_state = None
    best_sup_state = None
    best_wl_cls_state = None
    teacher_center: torch.Tensor | None = None
    wl_balance_distill_ema: float | None = None
    wl_balance_ema: float | None = None
    mined_partner_idx = torch.arange(num_nodes, device=device)
    mined_pair_coverage = 0.0

    for epoch in range(1, cfg.epochs + 1):
        student.train()
        proj_student.train()
        if pred_student is not None:
            pred_student.train()
        if sup_head is not None:
            sup_head.train()
        if wl_cls_heads is not None:
            wl_cls_heads.train()
        optimizer.zero_grad()
        tau_t_curr = _teacher_temperature(epoch=epoch, cfg=cfg)
        wl_naive_level = _wl_naive_curriculum_level(
            epoch=epoch,
            total_epochs=cfg.epochs,
            wl_depth=wl_depth_eff,
        )
        wl_naive_pair_tau = _wl_naive_pair_temperature(
            level=wl_naive_level,
            wl_depth=wl_depth_eff,
            tau_start=cfg.wl_naive_pair_temp_start,
            tau_end=cfg.wl_naive_pair_temp_end,
        )
        wl_naive_alpha = _wl_naive_uniform_alpha(
            epoch=epoch,
            total_epochs=cfg.epochs,
            start_frac=cfg.wl_naive_mix_start_frac,
            end_alpha=cfg.wl_naive_mix_end_alpha,
        )
        if need_wl_pairs and cfg.wl_naive_debug:
            members_t = members_by_level[wl_naive_level]
            unique_clusters = len(members_t)
            clusters_gt1 = sum(1 for nodes in members_t.values() if len(nodes) > 1)
            print(f"[WL-NAIVE][Epoch {epoch:03d}] WL level: {wl_naive_level}")
            print(f"[WL-NAIVE][Epoch {epoch:03d}] Pair tau: {wl_naive_pair_tau:.4f}")
            print(f"[WL-NAIVE][Epoch {epoch:03d}] Uniform alpha: {wl_naive_alpha:.4f}")
            print(f"[WL-NAIVE][Epoch {epoch:03d}] Pair mode: {wl_naive_pair_sampling}")
            print(f"[WL-NAIVE][Epoch {epoch:03d}] Distance beta: {cfg.wl_naive_distance_beta:.4f}")
            print(f"[WL-NAIVE][Epoch {epoch:03d}] Num clusters at level t: {unique_clusters}")
            print(f"[WL-NAIVE][Epoch {epoch:03d}] Clusters with size >1: {clusters_gt1}")

        if cfg.use_augmentations:
            x_s, ei_s = augmentor.augment(data.x, data.edge_index)
            x_t, ei_t = augmentor.augment(data.x, data.edge_index)
        else:
            x_s, ei_s = data.x, data.edge_index
            x_t, ei_t = data.x, data.edge_index

        if need_bootstrap:
            h_student_1 = student(x_s, ei_s)
            h_student_2 = student(x_t, ei_t)
            if pred_student is None:
                raise RuntimeError("Bootstrap objective requires predictor head.")

            if need_byol:
                z_student_1 = proj_student(h_student_1)
                z_student_2 = proj_student(h_student_2)
            else:
                # BGRL uses encoder representations directly.
                z_student_1 = h_student_1
                z_student_2 = h_student_2

            p_student_1 = pred_student(z_student_1)
            p_student_2 = pred_student(z_student_2)

            if teacher is None:
                raise RuntimeError("Bootstrap objectives require teacher encoder.")
            with torch.no_grad():
                h_teacher_1 = teacher(x_s, ei_s)
                h_teacher_2 = teacher(x_t, ei_t)
                if need_byol:
                    if proj_teacher is None:
                        raise RuntimeError("BYOL objective requires teacher projection head.")
                    z_teacher_1 = proj_teacher(h_teacher_1)
                    z_teacher_2 = proj_teacher(h_teacher_2)
                else:
                    z_teacher_1 = h_teacher_1
                    z_teacher_2 = h_teacher_2

            # Keep a canonical view for optional supervised anchor.
            h_student = h_student_1
            z_student = z_student_1
            z_teacher = z_teacher_1
        else:
            h_student = student(x_s, ei_s)
            z_student = proj_student(h_student)

            if teacher is None or proj_teacher is None:
                raise RuntimeError("DINO objectives require teacher encoder and projection head.")
            with torch.no_grad():
                h_teacher = teacher(x_t, ei_t)
                z_teacher = proj_teacher(h_teacher)

        need_distill = objective_mode in {"dino", "full"}
        need_wl = objective_mode in {"wl", "full"}
        need_wl_cls = need_wl_cls_obj
        need_candidate_distill = need_distill and distill_space == "candidate"
        need_candidate = need_candidate_distill or need_wl

        candidate_matrix = None
        if need_candidate:
            feat_neighbors = _compute_topk_feature_neighbors(
                z_teacher.detach(),
                k_feat=cfg.k_feat,
                chunk_size=cfg.feat_knn_chunk_size,
            )
            candidate_matrix = _build_candidate_matrix(
                wl_neighbors=wl_neighbors,
                feat_neighbors=feat_neighbors,
                max_candidates=cfg.max_candidates,
                num_random_neg=cfg.num_random_neg,
                seed=epoch + 104729,
            )

        if (need_distill or need_bootstrap) and miner is not None:
            refresh = max(1, int(cfg.miner_refresh_epochs))
            if epoch == 1 or ((epoch - 1) % refresh == 0):
                miner_view = z_teacher_2 if need_bootstrap else z_teacher
                with torch.no_grad():
                    mined_pos, _ = miner.mine_candidates(miner_view.detach())
                mined_partner_cpu, mined_pair_coverage = _sample_dual_view_partners(
                    positives=mined_pos,
                    seed=epoch + 1337,
                )
                mined_partner_idx = mined_partner_cpu.to(device)

        if need_distill:
            if distill_space == "prototype":
                center_dim = int(cfg.num_prototypes)
            else:
                if candidate_matrix is None:
                    raise RuntimeError("candidate_matrix is required for candidate distillation")
                center_dim = int(candidate_matrix.size(1))

            if teacher_center is None or teacher_center.size(1) != center_dim:
                teacher_center = torch.zeros((1, center_dim), device=device)

        perm = torch.randperm(num_nodes, device=device)
        num_batches = int(math.ceil(num_nodes / cfg.batch_size))

        epoch_loss = 0.0
        epoch_distill = 0.0
        epoch_wl = 0.0
        epoch_wl_term = 0.0
        epoch_wl_scale_sum = 0.0
        epoch_wl_scale_weight = 0.0
        epoch_sup = 0.0
        epoch_pair_coverage_sum = 0.0
        epoch_pair_coverage_weight = 0.0
        epoch_wl_cls_by_level = {t: 0.0 for t in wl_cls_levels_active}

        if sup_head is not None:
            if getattr(data, "train_mask", None) is not None and int(data.train_mask.sum().item()) > 0:
                sup_logits = sup_head(h_student[data.train_mask])
                loss_sup = F.cross_entropy(sup_logits, data.y[data.train_mask])
                (cfg.lambda_sup * loss_sup).backward(retain_graph=True)
                epoch_sup = float(loss_sup.item())
            else:
                loss_sup = torch.tensor(0.0, device=device)
        else:
            loss_sup = torch.tensor(0.0, device=device)

        for bi in range(num_batches):
            start = bi * cfg.batch_size
            end = min((bi + 1) * cfg.batch_size, num_nodes)
            batch_nodes = perm[start:end]
            bsz = int(batch_nodes.numel())
            batch_weight = float(bsz) / float(num_nodes)

            teacher_nodes = batch_nodes
            if (need_distill or need_bootstrap) and miner is not None:
                teacher_nodes = mined_partner_idx.index_select(0, batch_nodes)
            if need_bootstrap:
                if need_wl_pairs:
                    members_t = members_by_level[wl_naive_level]
                    batch_nodes_cpu = batch_nodes.detach().cpu().tolist()
                    sizes: List[int] = []
                    for v in batch_nodes_cpu:
                        cid = int(level_ids[wl_naive_level][v])
                        group = members_t.get(cid, [v])
                        sizes.append(len(group))

                    if sizes and cfg.wl_naive_debug:
                        avg_size = float(sum(sizes)) / float(len(sizes))
                        min_size = int(min(sizes))
                        max_size = int(max(sizes))
                        valid_ratio = float(sum(1 for s in sizes if s > 1)) / float(len(sizes))
                        print(
                            f"[WL-NAIVE][Epoch {epoch:03d}][Batch {bi:03d}] "
                            f"Cluster sizes avg={avg_size:.2f} min={min_size} max={max_size}"
                        )
                        print(
                            f"[WL-NAIVE][Epoch {epoch:03d}][Batch {bi:03d}] "
                            f"Valid WL pairs ratio: {valid_ratio:.4f}"
                        )

                    if cfg.wl_naive_debug:
                        for v in batch_nodes_cpu[:5]:
                            cid = int(level_ids[wl_naive_level][v])
                            group = members_t.get(cid, [v])
                            print(
                                f"[WL-NAIVE][Epoch {epoch:03d}][Batch {bi:03d}] "
                                f"v: {v} cluster: {group[:10]}"
                            )

                    valid_anchor_idx, pos_idx = _sample_wl_naive_partners(
                        batch_nodes=batch_nodes,
                        level_ids_t=level_ids[wl_naive_level],
                        members_t=members_t,
                        seed=(epoch * 1_000_003) + bi,
                        level_t=wl_naive_level,
                        wl_depth=wl_depth_eff,
                        sample_mode=wl_naive_pair_sampling,
                        anchor_repr=h_student_1,
                        pair_temperature=wl_naive_pair_tau,
                        uniform_alpha=wl_naive_alpha,
                        wl_distance_beta=cfg.wl_naive_distance_beta,
                        level_ids_stack=level_ids_stack_cpu,
                        distance_cache=wl_naive_distance_cache,
                        debug=cfg.wl_naive_debug,
                        epoch=epoch,
                        batch_idx=bi,
                    )
                    batch_pair_cov = float(valid_anchor_idx.numel()) / float(max(1, bsz))
                    epoch_pair_coverage_sum += batch_pair_cov * batch_weight
                    epoch_pair_coverage_weight += batch_weight

                    if valid_anchor_idx.numel() == 0:
                        # Keep graph connected for backward even when all anchors are skipped.
                        loss_distill = p_student_1.sum() * 0.0
                    else:
                        p_online_1 = p_student_1.index_select(0, valid_anchor_idx)
                        p_online_2 = p_student_2.index_select(0, valid_anchor_idx)
                        z_target_1 = z_teacher_1.index_select(0, pos_idx)
                        z_target_2 = z_teacher_2.index_select(0, pos_idx)
                        loss_distill = 0.5 * (
                            _byol_regression_loss(p_online_1, z_target_2)
                            + _byol_regression_loss(p_online_2, z_target_1)
                        )
                else:
                    p_online_1 = p_student_1.index_select(0, batch_nodes)
                    p_online_2 = p_student_2.index_select(0, batch_nodes)
                    z_target_1 = z_teacher_1.index_select(0, teacher_nodes)
                    z_target_2 = z_teacher_2.index_select(0, teacher_nodes)
                    loss_distill = 0.5 * (
                        _byol_regression_loss(p_online_1, z_target_2)
                        + _byol_regression_loss(p_online_2, z_target_1)
                    )
                if need_wl_cls:
                    if wl_cls_heads is None or len(wl_cls_levels_active) == 0:
                        loss_wl = h_student_1.sum() * 0.0
                    else:
                        h_anchor = h_student_1.index_select(0, batch_nodes)
                        loss_wl = torch.tensor(0.0, device=device)
                        for t in wl_cls_levels_active:
                            logits_t = wl_cls_heads[str(t)](h_anchor)
                            labels_t = wl_cls_labels[t].index_select(0, batch_nodes)
                            loss_wl_t = F.cross_entropy(logits_t, labels_t)
                            loss_wl = loss_wl + float(wl_cls_alpha[t]) * loss_wl_t
                            epoch_wl_cls_by_level[t] += float(loss_wl_t.item()) * batch_weight
                else:
                    loss_wl = torch.tensor(0.0, device=device)
            else:
                anchor_s = z_student.index_select(0, batch_nodes)  # [B, d]
                anchor_t = z_teacher.index_select(0, teacher_nodes)  # [B, d]

                cand_idx = None
                log_q_s_candidate = None
                if need_candidate:
                    if candidate_matrix is None:
                        raise RuntimeError("candidate_matrix missing for candidate-based computations")
                    cand_idx = candidate_matrix.index_select(0, batch_nodes.cpu()).to(device)  # [B, K]
                    flat = cand_idx.reshape(-1)
                    cand_s = z_student.index_select(0, flat).view(bsz, -1, z_student.size(1))  # [B, K, d]
                    cand_t = z_teacher.index_select(0, flat).view(bsz, -1, z_teacher.size(1))  # [B, K, d]

                if need_distill:
                    if teacher_center is None:
                        raise RuntimeError("teacher_center was not initialized")
                    if distill_space == "prototype":
                        if proto_student is None or proto_teacher is None:
                            raise RuntimeError("prototype heads are required for prototype distillation")
                        logits_s_distill = proto_student(anchor_s) / cfg.tau_s
                        logits_t_raw = proto_teacher(anchor_t)
                    else:
                        if not need_candidate or cand_idx is None:
                            raise RuntimeError("candidate distillation requires candidates")
                        logits_s_distill = torch.einsum("bd,bkd->bk", anchor_s, cand_s) / cfg.tau_s
                        logits_t_raw = torch.einsum("bd,bkd->bk", anchor_t, cand_t)
                        log_q_s_candidate = F.log_softmax(logits_s_distill, dim=-1)

                    logits_t = (logits_t_raw - teacher_center) / tau_t_curr
                    q_t = F.softmax(logits_t, dim=-1).detach()
                    log_q_s_distill = F.log_softmax(logits_s_distill, dim=-1)
                    loss_distill = -(q_t * log_q_s_distill).sum(dim=-1).mean()

                    with torch.no_grad():
                        batch_center = logits_t_raw.detach().mean(dim=0, keepdim=True)
                        teacher_center = (
                            teacher_center * cfg.center_momentum
                            + batch_center * (1.0 - cfg.center_momentum)
                        )
                else:
                    loss_distill = torch.tensor(0.0, device=device)

                if need_wl:
                    if not need_candidate or cand_idx is None:
                        raise RuntimeError("WL loss requires candidate neighbors")
                    wl_dist = _wl_distances_for_candidates(
                        level_ids_device=level_ids_device,
                        anchors=batch_nodes,
                        candidates=cand_idx,
                        wl_depth=wl_depth_eff,
                    )

                    if wl_loss_type == "align":
                        loss_wl = _wl_alignment_loss(
                            anchor=anchor_s,
                            neighbors=cand_s,
                            wl_dist=wl_dist,
                            tau_wl=cfg.tau_wl,
                            use_repulsion=cfg.use_wl_repulsion,
                            repulsion_beta=cfg.wl_repulsion_beta,
                            repulsion_threshold=cfg.wl_repulsion_threshold,
                        )
                    else:
                        if log_q_s_candidate is None:
                            logits_s_candidate = torch.einsum("bd,bkd->bk", anchor_s, cand_s) / cfg.tau_s
                            log_q_s_candidate = F.log_softmax(logits_s_candidate, dim=-1)
                        p_wl = F.softmax(-wl_dist / cfg.tau_wl, dim=-1)
                        loss_wl = F.kl_div(log_q_s_candidate, p_wl, reduction="batchmean")
                else:
                    loss_wl = torch.tensor(0.0, device=device)

            wl_scale = 1.0
            lambda_wl_eff = float(cfg.lambda_wl)
            if cfg.use_adaptive_wl_balance and need_distill and need_wl:
                wl_scale, wl_balance_distill_ema, wl_balance_ema = _adaptive_wl_scale(
                    loss_distill=loss_distill,
                    loss_wl=loss_wl,
                    distill_ema=wl_balance_distill_ema,
                    wl_ema=wl_balance_ema,
                    momentum=cfg.wl_balance_ema_momentum,
                    eps=cfg.wl_balance_eps,
                    min_scale=cfg.wl_balance_min_scale,
                    max_scale=cfg.wl_balance_max_scale,
                )
                lambda_wl_eff = float(cfg.lambda_wl) * wl_scale

            loss = _combine_losses(cfg.objective, loss_distill, loss_wl, lambda_wl_eff)
            retain_graph = bi < (num_batches - 1)
            (loss * batch_weight).backward(retain_graph=retain_graph)

            epoch_loss += float(loss.item()) * batch_weight
            epoch_distill += float(loss_distill.item()) * batch_weight
            epoch_wl += float(loss_wl.item()) * batch_weight
            epoch_wl_term += float((lambda_wl_eff * loss_wl).item()) * batch_weight
            if need_wl:
                epoch_wl_scale_sum += wl_scale * batch_weight
                epoch_wl_scale_weight += batch_weight

        epoch_loss += cfg.lambda_sup * epoch_sup
        if need_wl_pairs and epoch_pair_coverage_weight > 0.0:
            mined_pair_coverage = epoch_pair_coverage_sum / epoch_pair_coverage_weight

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if teacher is not None:
            _ema_update(teacher, student, cfg.m)
        if proj_teacher is not None:
            _ema_update(proj_teacher, proj_student, cfg.m)
        if proto_teacher is not None and proto_student is not None:
            _ema_update(proto_teacher, proto_student, cfg.m)

        if epoch % cfg.log_interval == 0 or epoch == cfg.epochs:
            acc = evaluate_linear_probe(
                student,
                data,
                dataset.num_classes,
                device,
            )

            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(student.state_dict())
                best_head_state = copy.deepcopy(proj_student.state_dict())
                if pred_student is not None:
                    best_pred_state = copy.deepcopy(pred_student.state_dict())
                if proto_student is not None:
                    best_proto_state = copy.deepcopy(proto_student.state_dict())
                if sup_head is not None:
                    best_sup_state = copy.deepcopy(sup_head.state_dict())
                if wl_cls_heads is not None:
                    best_wl_cls_state = copy.deepcopy(wl_cls_heads.state_dict())

            wl_scale_avg = (
                epoch_wl_scale_sum / max(epoch_wl_scale_weight, 1e-12)
                if epoch_wl_scale_weight > 0.0
                else 1.0
            )
            wl_metric_name = "WL_cls_sum" if need_wl_cls else f"WL_{wl_loss_type}"
            wl_cls_detail = ""
            if need_wl_cls and len(epoch_wl_cls_by_level) > 0:
                wl_cls_detail = " | " + ", ".join(
                    f"L_wl_t{t}: {epoch_wl_cls_by_level[t]:.4f}" for t in wl_cls_levels_active
                )

            print(
                f"[WL-DINO | {cfg.dataset:<12}] "
                f"Epoch {epoch:03d}/{cfg.epochs} "
                f"Loss: {epoch_loss:.4f} "
                f"Distill: {epoch_distill:.4f} "
                f"{wl_metric_name}: {epoch_wl:.4f} "
                f"WL_term: {epoch_wl_term:.4f} "
                f"WL_scale: {wl_scale_avg:.2f} "
                f"PairCov: {mined_pair_coverage:.3f} "
                f"Sup: {epoch_sup:.4f} "
                f"Tt: {tau_t_curr:.4f} "
                f"Acc: {acc:.4f}"
                f"{wl_cls_detail}"
            )

    print(f"[WL-DINO | {cfg.dataset.upper():<12}] Best Acc: {best_acc:.4f}")

    best_ckpt_path = None
    if cfg.save_best and best_state is not None and best_head_state is not None:
        out_dir = Path(cfg.output_dir) / cfg.dataset / cfg.model / cfg.objective
        out_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = out_dir / "best_encoder.pt"
        torch.save(
            {
                "encoder_state_dict": best_state,
                "projection_head_state_dict": best_head_state,
                "predictor_head_state_dict": best_pred_state,
                "prototype_head_state_dict": best_proto_state,
                "supervised_head_state_dict": best_sup_state,
                "wl_cls_heads_state_dict": best_wl_cls_state,
                "best_accuracy": best_acc,
                "cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else None,
            },
            best_ckpt_path,
        )

    return {
        "dataset": str(cfg.dataset),
        "model": str(cfg.model),
        "objective": str(cfg.objective),
        "distill_space": str(cfg.distill_space),
        "wl_loss_type": str(cfg.wl_loss_type),
        "best_accuracy": float(best_acc),
        "epochs": int(cfg.epochs),
        "best_ckpt_path": str(best_ckpt_path) if best_ckpt_path is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WL-guided DINO training.")

    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--model", type=str, default="gin")
    parser.add_argument(
        "--objective",
        choices=["dino", "byol", "bgrl", "bgrl_wl_naive", "bgrl_wl_cls", "bgrl_wl_naive_cls", "wl", "full"],
        default="full",
    )
    parser.add_argument("--distill_space", choices=["candidate", "prototype"], default=None)
    parser.add_argument("--num_prototypes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--out_dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--lambda_wl", type=float, default=None)
    parser.add_argument("--lambda_sup", type=float, default=None)
    parser.add_argument("--use_dual_view_miner_pairs", action="store_true")
    parser.add_argument("--miner_theta", type=float, default=None)
    parser.add_argument("--miner_delta", type=int, default=None)
    parser.add_argument("--miner_refresh_epochs", type=int, default=None)
    parser.add_argument("--wl_loss_type", choices=["align", "kl"], default=None)
    parser.add_argument("--wl_depth", type=int, default=None)
    parser.add_argument("--use_max_wl_depth", action="store_true")
    parser.add_argument("--use_adaptive_wl_balance", action="store_true")
    parser.add_argument("--wl_balance_ema_momentum", type=float, default=None)
    parser.add_argument("--wl_balance_eps", type=float, default=None)
    parser.add_argument("--wl_balance_min_scale", type=float, default=None)
    parser.add_argument("--wl_balance_max_scale", type=float, default=None)
    parser.add_argument("--use_wl_repulsion", action="store_true")
    parser.add_argument("--wl_repulsion_beta", type=float, default=None)
    parser.add_argument("--wl_repulsion_threshold", type=float, default=None)
    parser.add_argument("--wl_naive_step_size", type=int, default=None)
    parser.add_argument(
        "--wl_naive_pair_sampling",
        choices=["uniform", "feature_softmax", "hybrid", "wl_distance"],
        default=None,
    )
    parser.add_argument("--wl_naive_pair_temp_start", type=float, default=None)
    parser.add_argument("--wl_naive_pair_temp_end", type=float, default=None)
    parser.add_argument("--wl_naive_mix_start_frac", type=float, default=None)
    parser.add_argument("--wl_naive_mix_end_alpha", type=float, default=None)
    parser.add_argument("--wl_naive_distance_beta", type=float, default=None)
    parser.add_argument("--wl_cls_levels", type=str, default=None)
    parser.add_argument(
        "--wl_cls_alpha_scheme",
        choices=["uniform", "deeper_more", "shallower_more"],
        default=None,
    )
    parser.add_argument("--wl_naive_debug", action="store_true")
    parser.add_argument("--tau_t_warmup_start", type=float, default=None)
    parser.add_argument("--tau_t_warmup_epochs", type=int, default=None)
    parser.add_argument("--center_momentum", type=float, default=None)
    parser.add_argument("--use_augmentations", action="store_true")
    args = parser.parse_args()

    cfg = make_wl_dino_cfg(args.dataset)
    cfg = replace(
        cfg,
        model=args.model,
        objective=args.objective,
        use_augmentations=args.use_augmentations,
        use_dual_view_miner_pairs=args.use_dual_view_miner_pairs,
        use_adaptive_wl_balance=args.use_adaptive_wl_balance,
        wl_naive_debug=args.wl_naive_debug,
        use_max_wl_depth=args.use_max_wl_depth,
    )

    for k, v in vars(args).items():
        if k in {
            "dataset",
            "model",
            "objective",
            "use_augmentations",
            "use_dual_view_miner_pairs",
            "use_adaptive_wl_balance",
            "wl_naive_debug",
            "use_max_wl_depth",
        }:
            continue
        if v is not None and hasattr(cfg, k):
            cfg = replace(cfg, **{k: v})

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        cfg = replace(cfg, device="cpu")
        print("[WARN] CUDA requested but not available. Falling back to CPU.")

    train_wl_dino(cfg)


if __name__ == "__main__":
    main()
