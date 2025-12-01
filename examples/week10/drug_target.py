"""
Drug-Target Interaction Prediction using Graph Neural Networks
This script demonstrates how to build a simple Graph Neural Network (GNN)
to predict interactions between drugs and target proteins.
pip install torch torch-geometric
"""

import torch
from torch_geometric.nn import GCNConv

class DrugTargetGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(16, 32)
        self.conv2 = GCNConv(32, 16)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = DrugTargetGNN()

# node feature matrix (drug + protein)
x = torch.rand(10, 16)

# example edges (drug → protein)
edge_index = torch.tensor([[0,1,2,3],[5,6,7,8]])

out = model(x, edge_index)
print(out)
