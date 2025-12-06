# wl_gcl/src/trainers/train_wl.py
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import copy
import numpy as np
import random
import sys

from wl_gcl.src.utils.wl_core import WLHierarchyEngine
# --- CHANGED: Import the Model Factory instead of specific models ---
from wl_gcl.src.models import get_model
from wl_gcl.src.augmentations.graph_augmentor import GraphAugmentor
from wl_gcl.src.contrastive.dual_view_miner import DualViewMiner
from wl_gcl.src.contrastive.losses import ExtendedMoCHILoss
from wl_gcl.src.data_loader.dataset import load_dataset, get_splits

# ==========================================
# DATASET-SPECIFIC HYPERPARAMETERS
# ==========================================

CONFIGS = {
    # --- 1. Classiques Homophiles ---
    'cora': {
        'lr': 0.0005, 'hidden': 512, 'out': 256, 'epochs': 200, 
        'theta': 0.6, 'delta': 2, 'temp': 0.2, 'negs': 256,
        'drop_edge': 0.25, 'mask_feat': 0.25
    },
    'citeseer': {
        'lr': 0.0001, 'hidden': 512, 'out': 256, 'epochs': 300, 
        'theta': 0.75, 'delta': 2, 'temp': 0.5, 'negs': 128,
        'drop_edge': 0.1, 'mask_feat': 0.1
    },
    'amazon-photo': {
        'lr': 0.0005, 'hidden': 512, 'out': 256, 'epochs': 200,
        'theta': 0.7, 'delta': 2, 'temp': 0.2, 'negs': 512, 
        'drop_edge': 0.3, 'mask_feat': 0.3
    },

    # --- 2. Hétérophiles Moyens/Denses ---
    'actor': {
        'lr': 0.001, 'hidden': 256, 'out': 128, 'epochs': 400,
        'theta': 0.5, 'delta': 2, 'temp': 0.5, 'negs': 256,
        'drop_edge': 0.4, 'mask_feat': 0.1
    },
    'squirrel': {
        'lr': 0.001, 'hidden': 256, 'out': 128, 'epochs': 500, 
        'theta': 0.4, 'delta': 2, 'temp': 0.5, 'negs': 512, 
        'drop_edge': 0.5, 'mask_feat': 0.2 
    },
    'chameleon': {
        'lr': 0.001, 'hidden': 256, 'out': 128, 'epochs': 500,
        'theta': 0.4, 'delta': 2, 'temp': 0.5, 'negs': 512,
        'drop_edge': 0.5, 'mask_feat': 0.2
    },

    # --- 3. Hétérophiles Petits ---
    'texas': { 
        'lr': 0.01, 'hidden': 64, 'out': 32, 'epochs': 400, 
        'theta': 0.5, 'delta': 2, 'temp': 0.5, 'negs': 32,
        'drop_edge': 0.4, 'mask_feat': 0.1
    },
    'wisconsin': { 
        'lr': 0.01, 'hidden': 64, 'out': 32, 'epochs': 400, 
        'theta': 0.5, 'delta': 2, 'temp': 0.5, 'negs': 32,
        'drop_edge': 0.4, 'mask_feat': 0.1
    }
}

def train_model(dataset_name, model_type='gin'):
    """
    Main training function logic.
    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Launching WL-GCL on {dataset_name.upper()} using {model_type.upper()} ===")
    
    # 1. Load Config & Data
    # Fallback to Cora config if dataset specific config is missing
    cfg = CONFIGS.get(dataset_name.lower(), CONFIGS['cora']) 
    
    node_dataset = load_dataset(dataset_name)
    data = node_dataset.data
    num_classes = node_dataset.num_classes
    
    data = data.to(DEVICE)
    
    num_nodes = data.x.shape[0]
    input_dim = data.x.shape[1]
    
    print(f"Nodes: {num_nodes} | Features: {input_dim} | Classes: {num_classes}")
    
    # 2. Init WL Engine
    print("[1] Building WL Tree...")
    nodes_list = list(range(num_nodes))
    edges_list = data.edge_index.t().tolist()
    
    wl_engine = WLHierarchyEngine(nodes_list, edges_list)
    # Force convergence for small graphs (<1000 nodes) ensures max depth
    force = True if num_nodes < 1000 else False
    wl_engine.build_wl_tree(max_iterations=10, force_convergence=force)
    
    # 3. Init Models (Dynamic Selection)
    # Uses the factory 'get_model' we defined in src/models/__init__.py
    encoder = get_model(
        name=model_type,
        input_dim=input_dim,
        hidden_dim=cfg['hidden'],
        out_dim=cfg['out'],
        dropout=0.1,   # Default, could be moved to config if needed
        tau=2       # Default scaling factor for WLHN
    ).to(DEVICE)

    optimizer = optim.Adam(encoder.parameters(), lr=cfg['lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    
    # Use Dataset-specific augmentations
    augmentor = GraphAugmentor(edge_drop_prob=cfg['drop_edge'], feature_mask_prob=cfg['mask_feat'])
    miner = DualViewMiner(wl_engine, nodes_list, theta=cfg['theta'], delta=cfg['delta'])
    criterion = ExtendedMoCHILoss(temperature=cfg['temp'], num_negatives=cfg['negs'])
    
    # 4. Training Loop
    print(f"[2] Pre-training ({cfg['epochs']} epochs)...")
    encoder.train()
    
    best_acc = 0.0
    best_model_w = None
    ext_pos, hard_neg = [], []
    
    for epoch in range(cfg['epochs']):
        optimizer.zero_grad()
        
        # Mining (Every 10 epochs to stabilize)
        if epoch % 10 == 0:
            with torch.no_grad():
                encoder.eval()
                h = encoder(data.x, data.edge_index)
                ext_pos, hard_neg = miner.mine_candidates(h)
                encoder.train()
        
        # Augmentations & Forward
        x1, ei1 = augmentor.augment(data.x, data.edge_index)
        x2, ei2 = augmentor.augment(data.x, data.edge_index)
        z1 = encoder(x1, ei1)
        z2 = encoder(x2, ei2)
        
        # Loss Calculation
        total_loss = 0
        # Adaptive batch size: full batch for small graphs, 512 for large
        batch_size = min(512, num_nodes) 
        batch_idx = torch.randperm(num_nodes)[:batch_size]
        
        for idx in batch_idx:
            i = idx.item()
            anchor = z1[i].unsqueeze(0)
            pos_v2 = z2[i].unsqueeze(0)
            
            # --- Positive Sampling ---
            if len(ext_pos) > 0 and len(ext_pos[i]) > 0:
                indices = ext_pos[i]
                # Limit extended positives to avoid noise
                if len(indices) > 5: indices = random.sample(indices, 5)
                pos_ext = z2[indices]
                all_pos = torch.cat([pos_v2, pos_ext], dim=0)
            else:
                all_pos = pos_v2
                
            # --- Negative Sampling ---
            if len(hard_neg) > 0 and len(hard_neg[i]) > 0:
                indices = hard_neg[i]
                # Limit hard negatives
                if len(indices) > 20: indices = random.sample(indices, 20)
                h_neg = z2[indices]
            else:
                h_neg = torch.tensor([]).to(DEVICE)
                
            total_loss += criterion(anchor, all_pos, h_neg)
            
        loss = total_loss / len(batch_idx)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Monitoring
        if (epoch + 1) % 10 == 0:
            curr_acc = linear_probing(encoder, data, num_classes, DEVICE)
            
            if curr_acc > best_acc:
                best_acc = curr_acc
                best_model_w = copy.deepcopy(encoder.state_dict())
            
            print(f"Ep {epoch+1:03d} | Loss: {loss.item():.4f} | Acc: {curr_acc:.4f}")

    print(f"--> Best Accuracy for {dataset_name}: {best_acc:.4f}")
    return best_acc

def linear_probing(encoder, data, num_classes, device):
    """Standard Linear Probing Evaluation."""
    encoder.eval()
    with torch.no_grad():
        z = encoder(data.x, data.edge_index).detach()
    
    # Logistic Regression
    clf = nn.Linear(z.shape[1], num_classes).to(device)
    opt = optim.Adam(clf.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # Fast training of classifier
    for _ in range(100):
        opt.zero_grad()
        logits = clf(z)
        loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()
        
    # Test Accuracy
    with torch.no_grad():
        pred = clf(z).argmax(dim=1)
        test_mask = data.test_mask
        
        # Safety check for empty test sets (rare but possible in custom splits)
        if test_mask.sum().item() == 0:
            return 0.0
            
        correct = pred[test_mask].eq(data.y[test_mask]).sum().item()
        acc = correct / test_mask.sum().item()
        
    encoder.train()
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='cora, citeseer, texas, wisconsin')
    # Added model argument
    parser.add_argument('--model', type=str, default='gin', help='gin, gat, wlhn')
    args = parser.parse_args()
    
    train_model(args.dataset, args.model)