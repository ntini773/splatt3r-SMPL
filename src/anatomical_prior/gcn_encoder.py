import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, adj):
        """
        Dense matrix multiplication for fixed small graphs (e.g., 512 nodes).
        This bypasses the need for heavily engineered libraries like PyTorch3D or 
        torch_geometric, natively supporting PyTorch Lightning DistributedDataParallel (DDP) 
        without complex sparse tensor gather operations.
        x: [B, N, in_features] or [N, in_features]
        adj: [N, N] or [B, N, N]
        """
        support = self.linear(x)
        if adj.dim() == 2 and support.dim() == 3:
            # Broadcast adjacency matrix across batch
            out = torch.matmul(adj.unsqueeze(0), support)
        else:
            out = torch.matmul(adj, support)
        return out + self.bias

class AnatomicalGCNEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128, out_channels=32, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(DenseGCNLayer(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.layers.append(DenseGCNLayer(hidden_channels, hidden_channels))
            
        self.output_layer = DenseGCNLayer(hidden_channels, out_channels)

    def forward(self, x, adj):
        """
        x: [B, N, 3] or [B, N, 6] (Vertices + optional normal features)
        adj: [N, N] fixed adjacency matrix
        """
        for layer in self.layers:
            x = F.relu(layer(x, adj))
        x = self.output_layer(x, adj)
        return x
