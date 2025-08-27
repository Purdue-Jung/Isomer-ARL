import torch
from dataset.gamma_dataset import GammaSchemeDataset
from dataset.gran_data     import GRANData

class GammaGraphLoader(GRANData):
    """
    Wraps GammaSchemeDataset into GRANData packs and adds yy_matrix.
    """
    def __init__(self, cfg, graphs, tag):
        super().__init__(cfg, graphs, tag)
        self.yy_ds = GammaSchemeDataset(
            cfg.dataset.data_path,
            max_nodes=cfg.model.max_num_nodes
        )

    # add graph_id so we can fetch the right matrix later
    def __getitem__(self, idx):
        pack = super().__getitem__(idx)
        for d in pack:
            d["graph_id"] = idx         
        return pack

    def collate_fn(self, batch):
        parent    = super().collate_fn(batch)
        graph_ids = torch.tensor([pack[0]["graph_id"]    
                                  for pack in batch])
        yy_stk    = torch.stack([self.yy_ds[i]['yy_matrix'] for i in graph_ids])

        for d, gid in zip(parent, graph_ids):
            d["yy_matrix"] = yy_stk                    
            d["graph_id"]  = gid                       
        return parent                                   
