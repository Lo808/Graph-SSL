import sys
import os
import argparse

sys.path.append(os.getcwd())

from wl_gcl.src.trainers import train_wl, train_baseline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph SSL Training Hub")
    
    parser.add_argument('--method', type=str, default='wl', choices=['wl', 'baseline'], 
                        help="Choose 'wl' for your advanced model, 'baseline' for the simple GCN")
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset name')
    
    # NEW ARGUMENT:
    parser.add_argument('--model', type=str, default='gin', choices=['gcn', 'gin', 'gat', 'wlhn'],
                        help="Backbone GNN to use (default: gin)")

    args = parser.parse_args()
    
    if args.method == 'wl':
        print(f">>> Mode: WL-GCL | Model: {args.model.upper()}")
        # Pass the model argument here
        train_wl(args.dataset, model_type=args.model)
        
    elif args.method == 'baseline':
        print(">>> Mode: Baseline (Simple)")
        train_baseline(args)