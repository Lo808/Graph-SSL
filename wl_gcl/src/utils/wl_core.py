import networkx as nx
import matplotlib.pyplot as plt
import hashlib
from collections import defaultdict, deque
import torch

class WLHierarchyEngine:
    def __init__(self, nodes, edges):
        self.nodes = sorted(list(nodes))
        
        # Build optimized adjacency list
        self.adj = defaultdict(list)
        for u, v in edges:
            self.adj[u].append(v)
            self.adj[v].append(u)
            
        # Internal structures for the tree
        self.tree_adj = defaultdict(list)
        self.leaf_mapping = {} 
        self.tree_members = {}
        self.is_fitted = False
        
        #latest modifs
        self.parent = {}
        self.parent["root"] = None
        self.node_path = defaultdict(dict)

        # Keep a light NX graph only for visualization/LCA calculation if needed
        self._viz_graph = None 

    def build_wl_tree(self, max_iterations=5, force_convergence=False):
        """Builds the Weisfeiler-Lehman Hierarchical Tree."""
        print(f"Building WL Tree (Convergence Mode: {force_convergence})...")
        temp_tree = nx.DiGraph()
        
        limit = len(self.nodes) if force_convergence else max_iterations
        early_stopping = True
            
        # 1. Initialization (Root)
        current_labels = {n: "0" for n in self.nodes}
        root_id = "root"
        self.tree_members[root_id] = self.nodes
        for n in self.nodes:
            self.node_path[n][0] = "root"

        parent_in_tree = {n: root_id for n in self.nodes}
        
        previous_unique_count = 1
        
        # 2. WL Iteration Loop
        for it in range(1, limit + 1):
            new_labels = {}
            groups = defaultdict(list)
            
            # Hashing Step
            for n in self.nodes:
                my_lbl = current_labels[n]
                # Sort neighbor labels to ensure invariance
                neigh_lbls = sorted([current_labels[neigh] for neigh in self.adj[n]])
                signature = f"{my_lbl}-" + "-".join(neigh_lbls)
                hashed = hashlib.md5(signature.encode()).hexdigest()[:8]
                new_labels[n] = hashed
                groups[hashed].append(n)
            
            # Check Convergence
            if early_stopping and len(groups) == previous_unique_count:
                print(f"-> Stable convergence reached at iteration {it-1}.")
                break
            previous_unique_count = len(groups)
            
            # Update Tree Structure
            for label_hash, members in groups.items():
                tree_node_id = f"It{it}_{label_hash}"
                
                # Determine parent based on the first member's previous group
                representative = members[0]
                parent_id = parent_in_tree[representative]
                
                # Update Pure Python Structures
                # Add edges (Undirected view for BFS)
                self.tree_adj[parent_id].append(tree_node_id)
                self.tree_adj[tree_node_id].append(parent_id)
                self.parent[tree_node_id] = parent_id
                
                self.tree_members[tree_node_id] = members
                
                # Update Mappings
                for m in members:
                    parent_in_tree[m] = tree_node_id
                    self.leaf_mapping[m] = tree_node_id
                    self.node_path[m][it] = tree_node_id

                
                # Update Visualization Graph
                temp_tree.add_edge(parent_id, tree_node_id)
                temp_tree.nodes[tree_node_id]['subset'] = it
                temp_tree.nodes[tree_node_id]['members'] = members

            current_labels = new_labels

        # Add root to viz graph
        temp_tree.add_node("root", subset=0, members=self.nodes)
        self._viz_graph = temp_tree
        self.is_fitted = True
        return self

    def get_similar_nodes(self, target_node, delta=0):
        """
        Retrieves unique nodes within distance <= delta in the WL tree.
        Excludes the target_node itself.
        """
        if not self.is_fitted: return []
            
        start_leaf = self.leaf_mapping.get(target_node)
        if not start_leaf: return []
        
        # BFS on the Tree Structure
        queue = deque([(start_leaf, 0)])
        visited_tree_nodes = {start_leaf}
        found_graph_nodes = set() # Set to prevent duplicates
        
        while queue:
            curr_tree_node, dist = queue.popleft()
            
            # 1. Harvest graph nodes from this tree node
            members = self.tree_members.get(curr_tree_node, [])
            for m in members:
                if m != target_node:
                    found_graph_nodes.add(m)
            
            # 2. Continue exploration if within budget
            if dist < delta:
                for neighbor in self.tree_adj[curr_tree_node]:
                    if neighbor not in visited_tree_nodes:
                        visited_tree_nodes.add(neighbor)
                        queue.append((neighbor, dist + 1))
                        
        return sorted(list(found_graph_nodes))

    def get_structural_similarity(self, node_u, node_v):
        """
        Calculates a normalized similarity score based on LCA depth.
        Score = LCA_Depth / Max_Depth.
        """
        if not self._viz_graph: return 0.0
        leaf_u = self.leaf_mapping.get(node_u)
        leaf_v = self.leaf_mapping.get(node_v)
        
        if not leaf_u or not leaf_v: return 0.0
        if leaf_u == leaf_v: return 1.0
        
        # Use NetworkX for LCA as it's efficient on the small tree
        lca = nx.lowest_common_ancestor(self._viz_graph, leaf_u, leaf_v)
        lca_depth = self._viz_graph.nodes[lca].get('subset', 0)
        max_depth = max(nx.get_node_attributes(self._viz_graph, 'subset').values())
        
        return lca_depth / max_depth if max_depth > 0 else 0.0

    def visualize_hierarchy(self):
        """Plots the WL Tree with the Root at the top."""
        if not self._viz_graph: return
        
        plt.figure(figsize=(10, 6))
        # Multipartite layout organizes nodes by 'subset' (iteration layer)
        pos = nx.multipartite_layout(self._viz_graph, subset_key="subset", align="horizontal")
        
        # Invert Y-axis to place Root at top
        for n, c in pos.items(): pos[n] = (c[0], -c[1])
        
        labels = {}
        node_colors = []
        for n, d in self._viz_graph.nodes(data=True):
            node_colors.append(d.get('subset', 0))
            # Label only leaves (graph nodes) or root
            if self._viz_graph.out_degree(n) == 0:
                labels[n] = f"{{{','.join(map(str, d.get('members', [])))}}}"
            elif n == "root": 
                labels[n] = "ROOT"
            else: 
                labels[n] = ""
            
        nx.draw(self._viz_graph, pos, with_labels=True, labels=labels, 
                node_color=node_colors, cmap=plt.cm.coolwarm, node_size=600, font_weight='bold')
        plt.title("Generated WL Hierarchy")
        plt.show()
    
    #New features
    def get_cluster_id(self, node, level):
        return self.node_path[node].get(level)

    def get_cluster_at_level(self, node, level):
        cid = self.get_cluster_id(node, level)
        return self.tree_members.get(cid)

    def get_hard_negatives(self, node, level):
        prev = set(self.get_cluster_at_level(node, level-1))
        now  = set(self.get_cluster_at_level(node, level))
        return list(prev - now - {node})
    
    def get_wl_path(self, node):
        """
        Returns the WL path of a node as a list:
        [(level, cluster_id), ...]
        """
        path = []
        levels = sorted(self.node_path[node].keys())
        for t in levels:
            path.append((t, self.node_path[node][t]))
        return path
    
    def get_wl_cluster_sequence(self, node):
        """
        Returns [C^(0)(v), C^(1)(v), ..., C^(T)(v)]
        """
        levels = sorted(self.node_path[node].keys())
        return [self.node_path[node][t] for t in levels]
    def compute_centroids(self, z, level):
        """
        Compute WL cluster centroids at a given level.
        z: tensor [N, d] on device
        """
        device = z.device
        centroids = {}

        for tree_node, members in self.tree_members.items():
            if self._viz_graph.nodes[tree_node]['subset'] == level:
                idx = torch.tensor(members, device=device, dtype=torch.long)
                centroids[tree_node] = z.index_select(0, idx).mean(dim=0)

        return centroids
    def get_wl_distance(self, u, v):
        """
        WL structural distance as defined by the professor:
        d_WL(u,v) = T - max{t : C^(t)(u) = C^(t)(v)}
        """
        path_u = self.node_path[u]
        path_v = self.node_path[v]

        common_levels = set(path_u.keys()).intersection(path_v.keys())

        # find deepest level where they share the same cluster
        t_star = max(
            t for t in common_levels
            if path_u[t] == path_v[t]
        )

        T = max(max(path_u.keys()), max(path_v.keys()))
        return T - t_star
    def get_wl_similarity(self, u, v):
        """
        Normalized similarity in [0,1]
        """
        path_u = self.node_path[u]
        T = max(path_u.keys())
        return 1.0 - self.get_wl_distance(u, v) / T
    
    def get_level_targets(self, level):
        """
        Returns:
            y: LongTensor [N] with class indices for WL clusters at this level
            cid2idx: mapping from cluster_id -> class index
            num_classes: number of WL clusters at this level
        """
        cids = [self.get_cluster_id(v, level) for v in self.nodes]

        if any(cid is None for cid in cids):
            return None, None, None

        unique_cids = sorted(set(cids))
        cid2idx = {cid: i for i, cid in enumerate(unique_cids)}
        y = torch.tensor([cid2idx[cid] for cid in cids], dtype=torch.long)

        return y, cid2idx, len(unique_cids)

