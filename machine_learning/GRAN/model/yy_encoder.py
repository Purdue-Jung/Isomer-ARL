import torch
import torch.nn as nn
import torch.nn.functional as F

class YyEncoder(nn.Module):
    """
    CNN that consumes a 1×512×512 γ–γ matrix
    and returns a 256‑D context vector.

    Conv chain: 512→256→128→64→32→16→8→4  (7 stride‑2 layers)
    Output tensor: 512 ch × 4 × 4  → fc → 256
    """
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(  1,  32, 4, 2, 1), nn.ReLU(inplace=True),   # 512→256
            nn.Conv2d( 32,  64, 4, 2, 1), nn.ReLU(inplace=True),   # 256→128
            nn.Conv2d( 64, 128, 4, 2, 1), nn.ReLU(inplace=True),   # 128→64
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True),   #  64→32
            nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU(inplace=True),   #  32→16
            nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU(inplace=True),   #  16→8
            nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU(inplace=True),   #   8→4
        )
        self.fc = nn.Linear(512 * 4 * 4, out_dim)

    def forward(self, x):
        b = x.size(0)
        feat = self.conv(x)              
        feat = feat.view(b, -1)
        return self.fc(feat)