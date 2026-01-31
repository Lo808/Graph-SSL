import torch
import torch.nn.functional as F
import random



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
