import os
import torch
import numpy as np
import networkx as nx
from torch.utils.data import Dataset
from utils.data_helper import create_gamma_graphs

class GammaSchemeDataset(Dataset):
    """
    Loads your simulation .npz files as padded adjacency matrices.
    """
    def __init__(self, root_dir, max_nodes=64):
        self.max_nodes = max_nodes
        self.graphs = create_gamma_graphs(root_dir)

    def __len__(self):
        return len(self.graphs)

    @staticmethod
    def _pad(adj, N):
        if adj.shape[0] < N:
            pad = N - adj.shape[0]
            adj = np.pad(adj, ((0,pad),(0,pad)))
        return adj[:N,:N]

    def __getitem__(self, idx):
        G, mat = self.graphs[idx]      # change create_gamma_graphs to return (G, matrix)
        adj = nx.to_numpy_array(G, dtype='float32')
        adj = self._pad(adj, self.max_nodes)
        yy = torch.from_numpy(mat).unsqueeze(0).float()  # 1×64×64
        yy = yy / 255. if yy.max() > 1 else yy
        return {
            'adj': torch.from_numpy(adj),
            'n_nodes': G.number_of_nodes(),
            'yy_matrix': yy,
        }
