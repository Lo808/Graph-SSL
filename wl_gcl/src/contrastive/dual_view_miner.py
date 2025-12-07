import torch
import torch.nn.functional as F
import sys
import os

class DualViewMiner:
    def __init__(self, wl_engine, nodes_list, theta=0.8, delta=2):
        """
        Implements the mining logic from Ziyu's work (Section 3.2).
        
        Args:
            wl_engine: already fitted WL engine.
            nodes_list: The list of node IDs (e.g., [1, 2...]).
            theta (float): Cosine similarity threshold (Feature view).
            delta (int): WL distance threshold (Structural view).
        """
        self.wl_engine = wl_engine
        self.nodes_list = nodes_list
        # Mapping ID Graph -> Index Tensor (ex: Node 5 -> Index 4)
        self.node_to_idx = {node: i for i, node in enumerate(nodes_list)}
        
        self.theta = theta
        self.delta = delta

    def get_feature_candidates(self, embeddings):
        """
        Computes P_feat: Nodes close in embedding space.
        Returns a list of sets of indices.
        """
        # 1. Normalisation & Cosine Similarity Matrix
        # H_norm = H / ||H||
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        # Sim = H_norm @ H_norm.T
        sim_matrix = torch.matmul(emb_norm, emb_norm.t())
        
        # 2. Thresholding (sim > theta)
        # Ignore diagonal (self-similarity) for mining
        n = embeddings.shape[0]
        sim_matrix.fill_diagonal_(-1.0) 
        
        candidates = []
        for i in range(n):
            # Indices where similarity > theta
            indices = torch.where(sim_matrix[i] > self.theta)[0].tolist()
            candidates.append(set(indices))
            
        return candidates

    def get_structural_candidates(self):
        """
        Computes P_struct: Nodes close in the WL tree.
        Returns a list of sets of indices.
        """
        candidates = []
        for node_id in self.nodes_list:
            neighbor_ids = self.wl_engine.get_similar_nodes(node_id, delta=self.delta)
            # Conversion ID -> Index Tenseur
            neighbor_indices = [self.node_to_idx[nid] for nid in neighbor_ids if nid in self.node_to_idx]
            candidates.append(set(neighbor_indices))
            
        return candidates

    def mine_candidates(self, embeddings):
        """
        Executes Stage 2: Intersection and Symmetric Difference.
        
        Returns:
            extended_positives (List[List[int]]): For each node, list of reliable positives.
            hard_negatives (List[List[int]]): For each node, list of hard negatives.
        """
        p_feat_list = self.get_feature_candidates(embeddings)
        p_struct_list = self.get_structural_candidates()
        
        final_positives = []
        final_hard_negatives = []
        
        for i in range(len(self.nodes_list)):
            P_feat = p_feat_list[i]
            P_struct = p_struct_list[i]
            
            #  INTERSECTION 
            # Nodes close in BOTH views (Reliable positives)
            P_inter = P_feat.intersection(P_struct)
            
            #  HARD NEGATIVES 
            # (P_feat U P_struct) \ P_inter
            # Equivalent to symmetric difference (XOR)
            N_hard = P_feat.symmetric_difference(P_struct)
            
            final_positives.append(list(P_inter))
            final_hard_negatives.append(list(N_hard))
            
        return final_positives, final_hard_negatives


if __name__ == "__main__":
    sys.path.append(os.getcwd())     
    try:
        from wl_gcl.src.utils.wl_core import WLHierarchyEngine
    except ImportError:
        print("Could not import WLHierarchyEngine. Make sure you run this from the project root.")
        sys.exit(1)

    # Toy example
    nodes = [1, 2, 3, 4, 5, 6, 7, 8]
    edges = [(1, 2), (1, 8), (2, 3), (8, 3), (3, 4), (3, 7), (4, 5), (7, 6), (5, 6)]
    
    # WL Engine
    wl_engine = WLHierarchyEngine(nodes, edges)
    wl_engine.build_wl_tree(force_convergence=True)
    
    # GIN Encoder simulation
    x = torch.ones((8, 1), dtype=torch.float)
    torch.manual_seed(42)
    embeddings = torch.randn(8, 16) 
    
    # Dual view miner
    miner = DualViewMiner(wl_engine, nodes, theta=0.0, delta=2)
    
    print("\n[Mining Step]")
    positives, hard_negatives = miner.mine_candidates(embeddings)
    
    target_idx = 4 # Index of node 5
    target_id = 5
    
    print(f"Analysis for Node {target_id}:")
    print(f"  > Intersection (Extended Positives): {[nodes[i] for i in positives[target_idx]]}")
    print(f"  > Hard Negatives (XOR): {[nodes[i] for i in hard_negatives[target_idx]]}")