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


def _build_wl_engine(data, wl_depth: int) -> WLHierarchyEngine:
    nodes = list(range(data.num_nodes))
    edges = data.edge_index.t().tolist()

    engine = WLHierarchyEngine(nodes, edges)
    engine.build_wl_tree(
        max_iterations=wl_depth,
        force_convergence=False,
        early_stop=False,
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
    if mode in {"dino", "byol", "bgrl"}:
        return loss_distill
    if mode == "wl":
        return loss_wl
    if mode == "full":
        return loss_distill + lambda_wl * loss_wl
    raise ValueError("objective must be one of: dino, byol, bgrl, wl, full")


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


def train_wl_dino(cfg: WLDinoConfig) -> Dict[str, float]:
    device = torch.device(cfg.device)

    dataset = load_dataset(cfg.dataset)
    data = dataset.data.to(device)
    num_nodes = int(data.num_nodes)
    objective_mode = cfg.objective.strip().lower()
    if objective_mode not in {"dino", "byol", "bgrl", "wl", "full"}:
        raise ValueError("objective must be one of: dino, byol, bgrl, wl, full")
    need_byol = objective_mode == "byol"
    need_bgrl = objective_mode == "bgrl"
    need_bootstrap = need_byol or need_bgrl

    wl_engine = _build_wl_engine(data, wl_depth=cfg.wl_depth)
    level_ids, members_by_level = _extract_level_ids(wl_engine, wl_depth=cfg.wl_depth)
    wl_neighbors, _wl_neighbor_distances = _precompute_topk_wl_neighbors(
        level_ids=level_ids,
        members_by_level=members_by_level,
        k_wl=cfg.k_wl,
        wl_depth=cfg.wl_depth,
    )

    level_ids_device = [ids.to(device) for ids in level_ids]

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
    if cfg.use_dual_view_miner_pairs:
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

    optimizer = Adam(
        optim_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs) if cfg.scheduler else None

    proto_info = f" | num_prototypes={cfg.num_prototypes}" if ((not need_bootstrap) and distill_space == "prototype") else ""
    wl_loss_type = cfg.wl_loss_type.strip().lower()
    if wl_loss_type not in {"align", "kl"}:
        raise ValueError("wl_loss_type must be one of: align, kl")
    print(
        f"\n[WL-DINO] Dataset={cfg.dataset.upper()} | Model={cfg.model.upper()} | "
        f"objective={cfg.objective} | distill_space={distill_space}{proto_info} | "
        f"dual_view_pairs={cfg.use_dual_view_miner_pairs} | "
        f"wl_loss_type={wl_loss_type} | "
        f"adaptive_wl_balance={cfg.use_adaptive_wl_balance} | "
        f"lambda_sup={cfg.lambda_sup} | "
        f"epochs={cfg.epochs} | device={cfg.device}\n"
    )

    best_acc = 0.0
    best_state = None
    best_head_state = None
    best_pred_state = None
    best_proto_state = None
    best_sup_state = None
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
        optimizer.zero_grad()
        tau_t_curr = _teacher_temperature(epoch=epoch, cfg=cfg)

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

            with torch.no_grad():
                h_teacher_1 = teacher(x_s, ei_s)
                h_teacher_2 = teacher(x_t, ei_t)
                if need_byol:
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

            with torch.no_grad():
                h_teacher = teacher(x_t, ei_t)
                z_teacher = proj_teacher(h_teacher)

        need_distill = objective_mode in {"dino", "full"}
        need_wl = objective_mode in {"wl", "full"}
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

            teacher_nodes = batch_nodes
            if (need_distill or need_bootstrap) and miner is not None:
                teacher_nodes = mined_partner_idx.index_select(0, batch_nodes)
            if need_bootstrap:
                p_online_1 = p_student_1.index_select(0, batch_nodes)
                p_online_2 = p_student_2.index_select(0, batch_nodes)
                z_target_1 = z_teacher_1.index_select(0, teacher_nodes)
                z_target_2 = z_teacher_2.index_select(0, teacher_nodes)
                loss_distill = 0.5 * (
                    _byol_regression_loss(p_online_1, z_target_2)
                    + _byol_regression_loss(p_online_2, z_target_1)
                )
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
                        wl_depth=cfg.wl_depth,
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
            batch_weight = float(bsz) / float(num_nodes)
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

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        _ema_update(teacher, student, cfg.m)
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

            wl_scale_avg = (
                epoch_wl_scale_sum / max(epoch_wl_scale_weight, 1e-12)
                if epoch_wl_scale_weight > 0.0
                else 1.0
            )

            print(
                f"[WL-DINO | {cfg.dataset:<12}] "
                f"Epoch {epoch:03d}/{cfg.epochs} "
                f"Loss: {epoch_loss:.4f} "
                f"Distill: {epoch_distill:.4f} "
                f"WL_{wl_loss_type}: {epoch_wl:.4f} "
                f"WL_term: {epoch_wl_term:.4f} "
                f"WL_scale: {wl_scale_avg:.2f} "
                f"PairCov: {mined_pair_coverage:.3f} "
                f"Sup: {epoch_sup:.4f} "
                f"Tt: {tau_t_curr:.4f} "
                f"Acc: {acc:.4f}"
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
    parser.add_argument("--objective", choices=["dino", "byol", "bgrl", "wl", "full"], default="full")
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
    parser.add_argument("--use_adaptive_wl_balance", action="store_true")
    parser.add_argument("--wl_balance_ema_momentum", type=float, default=None)
    parser.add_argument("--wl_balance_eps", type=float, default=None)
    parser.add_argument("--wl_balance_min_scale", type=float, default=None)
    parser.add_argument("--wl_balance_max_scale", type=float, default=None)
    parser.add_argument("--use_wl_repulsion", action="store_true")
    parser.add_argument("--wl_repulsion_beta", type=float, default=None)
    parser.add_argument("--wl_repulsion_threshold", type=float, default=None)
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
    )

    for k, v in vars(args).items():
        if k in {
            "dataset",
            "model",
            "objective",
            "use_augmentations",
            "use_dual_view_miner_pairs",
            "use_adaptive_wl_balance",
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
