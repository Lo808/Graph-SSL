import torch
import torch.nn.functional as F
import random
from typing import Dict, List



def wl_classification_loss(logits, level_targets, active_levels):
    """
    Cross-entropy over selected WL levels.
    """
    loss = 0.0
    for t in active_levels:
        y_t = level_targets[t]["y"]
        loss = loss + torch.nn.functional.cross_entropy(logits[t], y_t)
    return loss


def hierarchy_regularization(engine, z, levels, nodes=None):
    """
    WL centroid consistency regularization.
    """
    if nodes is None:
        nodes = range(z.size(0))

    reg = torch.tensor(0.0, device=z.device)

    centroid_by_level = {
        t: engine.compute_centroids(z, t) for t in levels
    }

    for t in levels[1:]:
        t_prev = t - 1
        if t_prev not in centroid_by_level:
            continue

        cent_t = centroid_by_level[t]
        cent_prev = centroid_by_level[t_prev]

        for v in nodes:
            cid_t = engine.get_cluster_id(v, t)
            cid_prev = engine.get_cluster_id(v, t_prev)

            if cid_t in cent_t and cid_prev in cent_prev:
                reg = reg + (cent_t[cid_t] - cent_prev[cid_prev]).pow(2).sum()

    return reg / len(list(nodes))

def wl_contrastive_loss(engine, z, level, temperature=0.5):
    """
    Fully vectorized WL-InfoNCE.

    z: (N, d) normalized embeddings
    """

    device = z.device
    N = z.size(0)

    # ------------------------------------------------------------
    # 1) Similarity matrix (N x N)
    # ------------------------------------------------------------
    S = torch.mm(z, z.t()) / temperature
    expS = torch.exp(S)

    # ------------------------------------------------------------
    # 2) Build WL positive mask
    # mask_pos[u,v] = 1 if v in same WL cluster at 'level'
    # ------------------------------------------------------------
    mask_pos = torch.zeros((N, N), device=device, dtype=torch.float32)

    for u in range(N):
        cluster = engine.get_cluster_at_level(u, level)
        if cluster is None:
            continue
        mask_pos[u, cluster] = 1.0

    # remove self-similarity from positives
    mask_pos.fill_diagonal_(0.0)

    # ------------------------------------------------------------
    # 3) Negative mask = everything else
    # ------------------------------------------------------------
    mask_neg = 1.0 - mask_pos

    # ------------------------------------------------------------
    # 4) InfoNCE
    # ------------------------------------------------------------
    pos_sum = (expS * mask_pos).sum(dim=1)
    neg_sum = (expS * mask_neg).sum(dim=1)

    # avoid division by zero
    eps = 1e-9
    loss = -torch.log((pos_sum + eps) / (pos_sum + neg_sum + eps))

    return loss.mean()

def wl_tree_nll_loss(engine, logits_by_level: Dict[int, torch.Tensor], active_levels: List[int]) -> torch.Tensor:
    """
    Tree Path Negative Log-Likelihood (Tree-NLL).

    For each level t, compute:
        -log p(C^(t)(v) | C^(t-1)(v))
    by restricting the softmax at level t to children of the node's parent cluster at level t-1.

    logits_by_level[t]: (N, C_t) where C_t = nb clusters at level t (global head for that level)
    active_levels: list of levels to include (e.g. [1,2,3])

    Returns:
        scalar loss (mean over nodes, summed over levels)
    """
    device = next(iter(logits_by_level.values())).device
    total = torch.tensor(0.0, device=device)

    # We assume the model/engine are built on all nodes => N is the same for all levels.
    # We also assume level 0 is "root" and is not predicted by a head.
    levels = sorted([t for t in active_levels if t in logits_by_level and t >= 1])
    if len(levels) == 0:
        return total

    for t in levels:
        logits_t = logits_by_level[t]  # (N, C_t)

        # y_t: global class indices for level t (aligned with engine.nodes order)
        y_t, cid2idx_t, C_t = engine.get_level_targets(t)
        if y_t is None or cid2idx_t is None:
            continue
        y_t = y_t.to(device)

        # Build children lists: parent_cid -> list of child global indices (in [0..C_t-1])
        children_idx_by_parent: Dict[str, List[int]] = {}
        for child_cid in engine.level_nodes.get(t, []):
            p = engine.parent.get(child_cid)
            if p is None:
                continue
            # convert child cluster id -> global class index at level t
            if child_cid not in cid2idx_t:
                continue
            gi = cid2idx_t[child_cid]
            children_idx_by_parent.setdefault(p, []).append(gi)

        # For each parent cluster at level t-1, compute CE restricted to its children
        # We iterate over all parent clusters present at level t-1.
        parents = engine.level_nodes.get(t - 1, [])
        if t - 1 == 0:
            # include root as parent for level 1
            parents = ["root"]

        level_loss = torch.tensor(0.0, device=device)
        total_count = 0

        for p_cid in parents:
            child_gidx = children_idx_by_parent.get(p_cid, None)
            if child_gidx is None or len(child_gidx) == 0:
                continue

            # nodes that are in parent cluster p_cid (graph node ids)
            members = engine.tree_members.get(p_cid, None)
            if members is None or len(members) == 0:
                continue

            node_idx = torch.tensor([engine.node2idx[v] for v in members], device=device, dtype=torch.long)

            # logits restricted to children: (n_p, K)
            child_gidx_t = torch.tensor(child_gidx, device=device, dtype=torch.long)
            logits_sub = logits_t.index_select(0, node_idx).index_select(1, child_gidx_t)

            # targets for those nodes at level t: global indices in [0..C_t-1]
            y_sub_global = y_t.index_select(0, node_idx)

            # map global child index -> local index in [0..K-1]
            # Create a small lookup tensor of size C_t with -1 default (works because C_t is usually manageable)
            lut = torch.full((C_t,), -1, device=device, dtype=torch.long)
            lut[child_gidx_t] = torch.arange(child_gidx_t.numel(), device=device, dtype=torch.long)

            y_sub_local = lut[y_sub_global]  # (n_p,)

            # Safety: in principle all nodes in parent must map to one of its children.
            valid = y_sub_local >= 0
            if not torch.all(valid):
                logits_sub = logits_sub[valid]
                y_sub_local = y_sub_local[valid]
                if y_sub_local.numel() == 0:
                    continue

            level_loss = level_loss + F.cross_entropy(logits_sub, y_sub_local, reduction="sum")
            total_count += y_sub_local.numel()

        if total_count > 0:
            total = total + (level_loss / total_count)

    return total