import torch
from torch_geometric.data import Data, Batch
from WLHN import WLHNEncoder

def run_test():
    print("=== Test du WLHN Encoder (Sans torch_scatter) ===")

    # 1. Création de données factices (Dummy Data)
    # Graphe 1 : 3 noeuds, 3 features
    edge_index_1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x_1 = torch.randn(3, 16) # 16 features initiales
    data1 = Data(x=x_1, edge_index=edge_index_1)

    # Graphe 2 : 4 noeuds, 3 features
    edge_index_2 = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    x_2 = torch.randn(4, 16)
    data2 = Data(x=x_2, edge_index=edge_index_2)

    # Création du Batch (PyG gère automatiquement le data.batch)
    batch_data = Batch.from_data_list([data1, data2])
    
    print(f"Input Batch: {batch_data}")
    print(f" - Total Nodes: {batch_data.num_nodes}")
    print(f" - Num Graphs: {batch_data.num_graphs}")
    print(f" - Features Dim: {batch_data.num_features}")

    # 2. Initialisation du Modèle
    hidden_dim = 32
    model = WLHNEncoder(
        input_dim=16,
        hidden_dim=hidden_dim,
        n_layers=2,   # Profondeur de l'arbre WL
        tau=1.0
    )
    print("\nModèle WLHN initialisé avec succès.")

    # 3. Forward Pass
    try:
        embeddings = model(batch_data)
        
        print("\nForward pass terminé.")
        print(f"Output Shape: {embeddings.shape}")
        
        # 4. Vérifications
        expected_shape = (2, hidden_dim) # (Nombre de graphes, Hidden Dim)
        
        if embeddings.shape == expected_shape:
            print("✅ SUCCÈS : Les dimensions de sortie sont correctes.")
        else:
            print(f"❌ ERREUR : Dimensions attendues {expected_shape}, reçues {embeddings.shape}")

        # Vérifier qu'il n'y a pas de NaN (fréquent en hyperbolique si mal géré)
        if torch.isnan(embeddings).any():
            print("❌ ERREUR : La sortie contient des NaNs.")
        else:
            print("✅ SUCCÈS : Pas de NaNs détectés.")
            
        print("\nExemple d'embedding (Graphe 1) :")
        print(embeddings[0][:5]) # Affiche les 5 premières valeurs

    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE pendant le forward pass :\n{e}")

if __name__ == "__main__":
    run_test()