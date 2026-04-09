from __future__ import annotations

from typing import Dict, List, Any

import torch
import torch.nn.functional as F

from wl_gcl.src.losses.wl_losses import wl_contrastive_loss as wl_contrastive_loss_legacy


def branching_fraction_for_level(engine, level: int) -> float:
    """
    Fraction of graph nodes whose parent at level-1 branches into >=2 children at level.
    """
    parents = ["root"] if level == 1 else list(engine.level_nodes.get(level - 1, []))
    if not parents:
        return 0.0

    branching_nodes = 0
    total_nodes = 0

    for p_cid in parents:
        members = engine.tree_members.get(p_cid, [])
        if not members:
            continue

        k_children = 0
        for child_cid in engine.level_nodes.get(level, []):
            if engine.parent.get(child_cid) == p_cid:
                k_children += 1

        total_nodes += len(members)
        if k_children >= 2:
            branching_nodes += len(members)

    return float(branching_nodes / total_nodes) if total_nodes > 0 else 0.0


def tree_nll_loss_branch_only(
    engine,
    logits_by_level: Dict[int, torch.Tensor],
    active_levels: List[int],
    branch_only: bool = True,
    min_children: int = 2,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """
    Tree Path NLL with optional branch-only filtering.

    branch_only=True ignores deterministic parent->child transitions (K < min_children),
    preventing denominator dilution from zero-information samples.
    """
    device = next(iter(logits_by_level.values())).device
    total_loss = torch.tensor(0.0, device=device)

    levels = sorted([t for t in active_levels if t in logits_by_level and t >= 1])
    if not levels:
        return total_loss, {
            "used_levels": 0,
            "used_nodes": 0,
            "skipped_singleton_parents": 0,
            "total_parents": 0,
        }

    used_levels = 0
    used_nodes = 0
    skipped_singleton_parents = 0
    total_parents = 0

    for t in levels:
        logits_t = logits_by_level[t]  # (N, C_t)
        y_t, cid2idx_t, c_t = engine.get_level_targets(t)
        if y_t is None or cid2idx_t is None or c_t is None:
            continue
        y_t = y_t.to(device)

        children_idx_by_parent: Dict[str, List[int]] = {}
        for child_cid in engine.level_nodes.get(t, []):
            p_cid = engine.parent.get(child_cid)
            if p_cid is None or child_cid not in cid2idx_t:
                continue
            children_idx_by_parent.setdefault(p_cid, []).append(cid2idx_t[child_cid])

        parents = ["root"] if t == 1 else list(engine.level_nodes.get(t - 1, []))

        level_loss_sum = torch.tensor(0.0, device=device)
        level_count = 0

        for p_cid in parents:
            total_parents += 1
            child_gidx = children_idx_by_parent.get(p_cid, [])
            k_children = len(child_gidx)

            if k_children == 0:
                continue

            if branch_only and k_children < min_children:
                skipped_singleton_parents += 1
                continue

            members = engine.tree_members.get(p_cid, [])
            if not members:
                continue

            node_idx = torch.tensor([engine.node2idx[v] for v in members], device=device, dtype=torch.long)
            child_gidx_t = torch.tensor(child_gidx, device=device, dtype=torch.long)

            logits_sub = logits_t.index_select(0, node_idx).index_select(1, child_gidx_t)
            y_sub_global = y_t.index_select(0, node_idx)

            lut = torch.full((c_t,), -1, device=device, dtype=torch.long)
            lut[child_gidx_t] = torch.arange(child_gidx_t.numel(), device=device, dtype=torch.long)
            y_sub_local = lut[y_sub_global]

            valid = y_sub_local >= 0
            if not torch.all(valid):
                logits_sub = logits_sub[valid]
                y_sub_local = y_sub_local[valid]
                if y_sub_local.numel() == 0:
                    continue

            level_loss_sum = level_loss_sum + F.cross_entropy(logits_sub, y_sub_local, reduction="sum")
            level_count += int(y_sub_local.numel())

        if level_count > 0:
            total_loss = total_loss + (level_loss_sum / level_count)
            used_levels += 1
            used_nodes += level_count

    return total_loss, {
        "used_levels": used_levels,
        "used_nodes": used_nodes,
        "skipped_singleton_parents": skipped_singleton_parents,
        "total_parents": total_parents,
    }


def wl_contrastive_loss_fixed(
    engine,
    z: torch.Tensor,
    level: int,
    temperature: float = 0.5,
    ignore_self_in_denominator: bool = True,
    skip_rows_without_positive: bool = True,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """
    Mask-safe WL InfoNCE.

    Fixes two issues from the legacy implementation:
    - self-pairs are removed from denominator
    - rows with no positives can be skipped instead of forced huge penalties
    """
    device = z.device
    n = z.size(0)

    logits = torch.mm(z, z.t()) / temperature

    mask_pos = torch.zeros((n, n), dtype=torch.bool, device=device)

    cluster_to_idx: Dict[str, List[int]] = {}
    for i, node in enumerate(engine.nodes):
        cid = engine.get_cluster_id(node, level)
        if cid is None:
            continue
        cluster_to_idx.setdefault(cid, []).append(i)

    for idxs in cluster_to_idx.values():
        idx = torch.tensor(idxs, device=device, dtype=torch.long)
        mask_pos[idx.unsqueeze(1), idx.unsqueeze(0)] = True

    mask_pos.fill_diagonal_(False)

    mask_denom = torch.ones((n, n), dtype=torch.bool, device=device)
    if ignore_self_in_denominator:
        mask_denom.fill_diagonal_(False)

    neg_inf = torch.tensor(float("-inf"), device=device)
    pos_logits = torch.where(mask_pos, logits, neg_inf)
    den_logits = torch.where(mask_denom, logits, neg_inf)

    log_pos = torch.logsumexp(pos_logits, dim=1)
    log_den = torch.logsumexp(den_logits, dim=1)

    row_has_pos = mask_pos.any(dim=1)
    row_has_den = mask_denom.any(dim=1)
    valid_rows = row_has_den
    if skip_rows_without_positive:
        valid_rows = valid_rows & row_has_pos

    if not torch.any(valid_rows):
        zero = torch.tensor(0.0, device=device)
        return zero, {
            "valid_rows": 0,
            "rows_with_positive": int(row_has_pos.sum().item()),
            "total_rows": int(n),
        }

    loss_rows = -(log_pos - log_den)
    loss_rows = torch.where(torch.isfinite(loss_rows), loss_rows, torch.tensor(0.0, device=device))
    loss = loss_rows[valid_rows].mean()

    return loss, {
        "valid_rows": int(valid_rows.sum().item()),
        "rows_with_positive": int(row_has_pos.sum().item()),
        "total_rows": int(n),
    }


def compute_wl_contrastive(
    engine,
    z: torch.Tensor,
    level: int,
    temperature: float,
    mode: str,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """
    Dispatch helper:
      - mode='legacy' -> original project loss
      - mode='fixed'  -> mask-safe variant
    """
    m = mode.lower()
    if m == "legacy":
        loss = wl_contrastive_loss_legacy(engine, z, level=level, temperature=temperature)
        return loss, {"mode": "legacy"}
    if m == "fixed":
        loss, stats = wl_contrastive_loss_fixed(engine, z, level=level, temperature=temperature)
        stats["mode"] = "fixed"
        return loss, stats
    raise ValueError(f"Unknown contrastive mode: {mode}")
