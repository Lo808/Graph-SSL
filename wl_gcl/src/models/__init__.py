from .base_gnn import GCNEncoder, GATEncoder, GINEncoder 
#from .wlhn import WLHNEncoder

__all__ = ["GCNEncoder", "GINEncoder", "GATEncoder", "WLHNEncoder", "get_model"]

def get_model(name, input_dim, hidden_dim, out_dim, **kwargs):
    """
    Factory function to initialize a model by name.
    """
    name = name.lower()
    
    if name == 'gcn':
        return GCNEncoder(
            in_dim=input_dim, 
            hidden_dim=hidden_dim, 
            out_dim=out_dim, 
            dropout=kwargs.get('dropout', 0.5)
        )
        
    elif name == 'gin':
        return GINEncoder(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=out_dim,
            num_layers=kwargs.get('num_layers', 3)
        )
        
    elif name == 'gat':
        return GATEncoder(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=out_dim,
            heads=kwargs.get('heads', 4),
            dropout=kwargs.get('dropout', 0.6)
        )

    # elif name == 'wlhn':
    #     return WLHNEncoder(
    #         input_dim=input_dim, 
    #         hidden_dim=hidden_dim, 
    #         output_dim=out_dim,
    #         n_layers=kwargs.get('num_layers', 3),
    #         tau=kwargs.get('tau', 1.0),
    #         dropout=kwargs.get('dropout', 0.5)
    #     )
        
    else:
        raise ValueError(f"Model '{name}' is not implemented. Options: gcn, gin, gat, wlhn")