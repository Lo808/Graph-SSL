from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Set, Tuple, Union

import torch

try:
    import networkx as nx
    import matplotlib.pyplot as plt
except Exception:
    nx = None
    plt = None


Node = Hashable


@dataclass(frozen=True)
class WLBuildConfig:
    max_iterations: int = 5
    force_convergence: bool = False
    early_stop: bool = True


class WLHierarchyEngine:
    """
    Builds a  WL refinement hierarchy.

    Key invariants:
      - Each hierarchy node has exactly one parent (except root).
      - Refinement nodes are keyed by (parent_cluster, new_label), preventing merges across parents.
      - Node ids can be arbitrary hashables; we maintain node2idx for tensor ops.
    """

    def __init__(self, nodes: Iterable[Node], edges: Iterable[Tuple[Node, Node]]):
        self.nodes: List[Node] = sorted(list(nodes))
        self.node2idx: Dict[Node, int] = {n: i for i, n in enumerate(self.nodes)}
        self.idx2node: List[Node] = list(self.nodes)

        # Undirected adjacency list
        self.adj: Dict[Node, List[Node]] = defaultdict(list)
        for u, v in edges:
            self.adj[u].append(v)
            self.adj[v].append(u)

        # Fit state
        self._reset_state()


    def build_wl_tree(
        self,
        max_iterations: int = 5,
        force_convergence: bool = False,
        early_stop: bool = True,
    ) -> "WLHierarchyEngine":
        cfg = WLBuildConfig(
            max_iterations=max_iterations,
            force_convergence=force_convergence,
            early_stop=early_stop,
        )
        return self._fit(cfg)
    


    
    # ================== Utils ==========================   

    def get_cluster_id(self, node: Node, level: int) -> Optional[str]:
        return self.node_path.get(node, {}).get(level)

    def get_cluster_at_level(self, node: Node, level: int) -> Optional[List[Node]]:
        cid = self.get_cluster_id(node, level)
        if cid is None:
            return None
        return self.tree_members.get(cid)

    def get_hard_negatives(self, node: Node, level: int) -> List[Node]:
        """Nodes that shared a cluster with `node` at level-1 but diverged at `level`."""
        if not self.is_fitted or level <= 0:
            return []
        prev_members = self.get_cluster_at_level(node, level - 1)
        now_members = self.get_cluster_at_level(node, level)
        if prev_members is None or now_members is None:
            return []
        
        return [n for n in prev_members if n not in set(now_members) and n != node]

    def get_wl_path(self, node: Node) -> List[Tuple[int, str]]:
        if node not in self.node_path:
            return []
        return [(t, self.node_path[node][t]) for t in sorted(self.node_path[node])]

    def get_wl_cluster_sequence(self, node: Node) -> List[str]:
        if node not in self.node_path:
            return []
        return [self.node_path[node][t] for t in sorted(self.node_path[node])]

    def get_similar_nodes(self, target_node: Node, delta: int = 0) -> List[Node]:
        """
        Unique graph nodes within tree-distance <= delta from the target node's leaf cluster.
        Excludes target_node itself. Order is BFS-stable
        """
        if not self.is_fitted:
            return []
        start_leaf = self.leaf_mapping.get(target_node)
        if start_leaf is None:
            return []

        queue: deque[Tuple[str, int]] = deque([(start_leaf, 0)])
        visited_tree: Set[str] = {start_leaf}
        found: List[Node] = []
        seen_nodes: Set[Node] = {target_node}

        while queue:
            curr, dist = queue.popleft()
            for m in self.tree_members.get(curr, []):
                if m not in seen_nodes:
                    seen_nodes.add(m)
                    found.append(m)
            if dist < delta:
                for nb in self.tree_adj.get(curr, []):
                    if nb not in visited_tree:
                        visited_tree.add(nb)
                        queue.append((nb, dist + 1))

        return found

    def get_wl_distance(self, u: Node, v: Node) -> int:
        """
        d_WL(u, v) = T - max{t : C^(t)(u) == C^(t)(v)}
        where T = max depth (same for all nodes when built together).
        """
        if not self.is_fitted or u == v:
            return 0
        if u not in self.node_path or v not in self.node_path:
            return 0

        path_u = self.node_path[u]
        path_v = self.node_path[v]
        T = self.max_depth

        # Walk from deepest level upward; stop at first shared cluster.
        for t in range(T, -1, -1):
            cu = path_u.get(t)
            cv = path_v.get(t)
            if cu is not None and cv is not None and cu == cv:
                return T - t

        return T  


    def get_wl_similarity(self, u: Node, v: Node) -> float:
        """
        Normalized similarity in [0,1].
        """
        if not self.is_fitted:
            return 0.0
        if u == v:
            return 1.0
        if u not in self.node_path:
            return 0.0
        T = max(self.node_path[u].keys())
        if T <= 0:
            return 1.0 if u == v else 0.0
        return float(1.0 - (self.get_wl_distance(u, v) / T))

    def get_structural_similarity(self, node_u: Node, node_v: Node) -> float:
        """
        Normalized similarity score based on LCA depth:
            sim = depth(LCA) / max_depth
        computed using parent pointers (no NetworkX required).
        """
        if not self.is_fitted:
            return 0.0

        leaf_u = self.leaf_mapping.get(node_u)
        leaf_v = self.leaf_mapping.get(node_v)
        if leaf_u is None or leaf_v is None:
            return 0.0
        if leaf_u == leaf_v:
            return 1.0

        lca = self._lca_by_parent(leaf_u, leaf_v)
        lca_depth = self.tree_level.get(lca, 0)
        max_depth = self.max_depth
        return float(lca_depth / max_depth) if max_depth > 0 else 0.0

    def get_level_targets(self, level: int) -> Tuple[Optional[torch.LongTensor], Optional[Dict[str, int]], Optional[int]]:
        """
        Returns:
            y: LongTensor [N] with class indices for WL clusters at this level
            cid2idx: mapping cluster_id -> class index
            num_classes: number of clusters at this level
        """
        if not self.is_fitted:
            return None, None, None

        cids = [self.get_cluster_id(v, level) for v in self.nodes]
        if any(cid is None for cid in cids):
            return None, None, None

        unique = sorted(set(cids))  # type: ignore[arg-type]
        cid2idx = {cid: i for i, cid in enumerate(unique)}  # type: ignore[misc]
        y = torch.tensor([cid2idx[cid] for cid in cids], dtype=torch.long)  # type: ignore[index]
        return y, cid2idx, len(unique)

    def compute_centroids(self, z: torch.Tensor, level: int) -> Dict[str, torch.Tensor]:
        """
        Compute WL cluster centroids at a given level.
        z: tensor [N, d] aligned with self.nodes ordering.
        """
        if not self.is_fitted:
            return {}
        if z.dim() != 2 or z.size(0) != len(self.nodes):
            raise ValueError(f"`z` must be [N,d] with N={len(self.nodes)} aligned to self.nodes.")

        device = z.device
        out: Dict[str, torch.Tensor] = {}

        for cid in self.level_nodes.get(level, []):
            members = self.tree_members[cid]
            idx = torch.tensor([self.node2idx[m] for m in members], device=device, dtype=torch.long)
            out[cid] = z.index_select(0, idx).mean(dim=0)

        return out

    def visualize_hierarchy(self, figsize: Tuple[int, int] = (12, 7)) -> None:
        """
        Optional visualization using NetworkX (if installed). Root appears at top.
        """
        if nx is None or plt is None:
            raise RuntimeError("NetworkX/matplotlib not available in this environment.")
        if not self.is_fitted or self._viz_graph is None:
            return

        plt.figure(figsize=figsize)
        pos = nx.multipartite_layout(self._viz_graph, subset_key="subset", align="horizontal")

        for n, c in pos.items():
            pos[n] = (c[0], -c[1])

        labels = {}
        node_colors = []
        for n, d in self._viz_graph.nodes(data=True):
            node_colors.append(d.get("subset", 0))
            if self._viz_graph.out_degree(n) == 0:
                labels[n] = f"{{{','.join(map(str, d.get('members', [])))}}}"
            elif n == "root":
                labels[n] = "ROOT"
            else:
                labels[n] = ""

        nx.draw(
            self._viz_graph,
            pos,
            with_labels=True,
            labels=labels,
            node_color=node_colors,
            cmap=plt.cm.coolwarm,
            node_size=650,
            font_weight="bold",
            arrows=True,
        )
        plt.title("WL Hierarchy (Tree-Correct)")
        plt.show()


    # ============================ Fit internals =========================
    def _reset_state(self) -> None:
        self.tree_adj: Dict[str, List[str]] = defaultdict(list)  # undirected view for BFS
        self.tree_members: Dict[str, List[Node]] = {}
        self.leaf_mapping: Dict[Node, str] = {}
        self.parent: Dict[str, Optional[str]] = {"root": None}
        self.node_path: Dict[Node, Dict[int, str]] = defaultdict(dict)

        self.tree_level: Dict[str, int] = {"root": 0}
        self.level_nodes: Dict[int, List[str]] = defaultdict(list)
        self.max_depth: int = 0

        self._viz_graph = None
        self.is_fitted: bool = False


    def _get_level_idx(self, cid: str, device: torch.device) -> torch.Tensor:
        """Return (and cache) a LongTensor of node indices for a cluster."""
        key = (cid, device)
        if key not in self._level_idx_cache:
            members = self.tree_members[cid]
            self._level_idx_cache[key] = torch.tensor(
                [self.node2idx[m] for m in members], device=device, dtype=torch.long
            )
        return self._level_idx_cache[key]


    @staticmethod
    def _canonical_relabel(signatures: List[Tuple[str, ...]]) -> List[str]:
        """
        Collision-free canonical relabeling: deterministic integer id per unique signature,
        in order of first appearance.
        """
        sig2id: Dict[Tuple[str, ...], int] = {}
        out: List[str] = []
        next_id = 0
        for sig in signatures:
            if sig not in sig2id:
                sig2id[sig] = next_id
                next_id += 1
            out.append(str(sig2id[sig]))
        return out


    def _fit(self, cfg: WLBuildConfig) -> "WLHierarchyEngine":
        self._reset_state()

        root_id = "root"
        self.tree_members[root_id] = list(self.nodes)
        for n in self.nodes:
            self.node_path[n][0] = root_id

        current_labels: Dict[Node, str] = {n: "0" for n in self.nodes}
        parent_cluster: Dict[Node, str] = {n: root_id for n in self.nodes}

        if nx is not None:
            self._viz_graph = nx.DiGraph()
            self._viz_graph.add_node(root_id, subset=0, members=list(self.nodes))

        limit = len(self.nodes) if cfg.force_convergence else cfg.max_iterations

        prev_partition: Optional[frozenset] = None

        for it in range(1, limit + 1):

            # Build signatures: (own_label, *sorted_neighbor_labels)
            signatures: List[Tuple[str, ...]] = []
            for n in self.nodes:
                neigh_lbls = sorted(current_labels[nb] for nb in self.adj.get(n, []))
                signatures.append((current_labels[n], *neigh_lbls))

            new_raw_labels = self._canonical_relabel(signatures)
            new_label: Dict[Node, str] = dict(zip(self.nodes, new_raw_labels))

            # Group by (parent_cluster, new_label)
            groups: Dict[Tuple[str, str], List[Node]] = defaultdict(list)
            for n, lbl in new_label.items():
                groups[(parent_cluster[n], lbl)].append(n)

            # Compute partition fingerprint for early-stop comparison
            current_partition = frozenset(
                frozenset(members) for members in groups.values()
            )

            if cfg.early_stop and prev_partition is not None and current_partition == prev_partition:
                break
            prev_partition = current_partition

            # Build tree nodes for this iteration
            for (p_id, h), members in groups.items():
                tree_node_id = f"It{it}_{p_id}_{h}"

                self.tree_adj[p_id].append(tree_node_id)
                self.tree_adj[tree_node_id].append(p_id)
                self.parent[tree_node_id] = p_id
                self.tree_members[tree_node_id] = members
                self.tree_level[tree_node_id] = it
                self.level_nodes[it].append(tree_node_id)

                for m in members:
                    parent_cluster[m] = tree_node_id
                    self.leaf_mapping[m] = tree_node_id
                    self.node_path[m][it] = tree_node_id

                if self._viz_graph is not None:
                    self._viz_graph.add_edge(p_id, tree_node_id)
                    self._viz_graph.nodes[tree_node_id]["subset"] = it
                    self._viz_graph.nodes[tree_node_id]["members"] = members

            self.max_depth = it
            current_labels = new_label

        self.is_fitted = True
        return self

    def _lca_by_parent(self, a: str, b: str) -> str:
        """
        LCA via depth-equalisation then tandem walk — O(depth) instead of O(depth²).
        """
        depth_a = self.tree_level.get(a, 0)
        depth_b = self.tree_level.get(b, 0)

        # Walk the deeper node up until both are at the same depth
        x, y = a, b
        while depth_a > depth_b:
            x = self.parent.get(x) or "root"
            depth_a -= 1
        while depth_b > depth_a:
            y = self.parent.get(y) or "root"
            depth_b -= 1

        # Now walk both up together
        while x != y:
            x = self.parent.get(x) or "root"
            y = self.parent.get(y) or "root"

        return x
