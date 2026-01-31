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

def wl_contrastive_loss(
    engine,
    z,
    level,
    num_negatives=50,
    temperature=0.2,
):
    """
    WL-guided InfoNCE.

    For each node v at WL level t:
        positives = nodes in same WL cluster
        hard negatives = nodes that split at this level
        easy negatives = random other nodes
    """

    N = z.size(0)
    device = z.device
    loss = torch.tensor(0.0, device=device)

    for v in range(N):
        pos = engine.get_cluster_at_level(v, level)
        if pos is None or len(pos) <= 1:
            continue

        # pick one positive different from v
        pos_candidates = [u for u in pos if u != v]
        u_pos = random.choice(pos_candidates)

        # hard negatives
        hard_neg = engine.get_hard_negatives(v, level)

        # easy negatives (random far nodes)
        all_nodes = set(range(N))
        forbidden = set(pos) | {v}
        easy_pool = list(all_nodes - forbidden)

        easy_neg = random.sample(
            easy_pool, 
            min(num_negatives, len(easy_pool))
        )

        negatives = hard_neg + easy_neg
        if len(negatives) == 0:
            continue

        z_v = z[v]
        z_pos = z[u_pos]
        z_negs = z[negatives]

        # cosine similarities
        sim_pos = F.cosine_similarity(z_v, z_pos, dim=0) / temperature
        sim_negs = F.cosine_similarity(
            z_v.unsqueeze(0),
            z_negs,
            dim=1
        ) / temperature

        numerator = torch.exp(sim_pos)
        denominator = numerator + torch.exp(sim_negs).sum()

        loss = loss - torch.log(numerator / denominator)

    return loss / N
