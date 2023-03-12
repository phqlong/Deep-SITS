import torch
from torch import nn

from src.models.components.TSViT import TSViT


class BiTSTransformer(nn.Module):
    """
    Bi-TSViT model for Sentinel 1 & Sentinel 2 data
    """
    def __init__(self,
                 num_channels1: int = 6,
                 num_channels2: int = 6,
                 hidden_dim: int = 128,
                 temporal_depth: int = 12,
                 heads: int = 3,
                 dim_head: int = 64,
                 scale_dim: int = 2,
                 dropout: float = 0.1
                ):
        super().__init__()

        self.s1_model = TSViT(num_channels=num_channels1, 
                              hidden_dim=hidden_dim, 
                              temporal_depth=temporal_depth, 
                              heads=heads, 
                              dim_head=dim_head, 
                              scale_dim=scale_dim, 
                              dropout=dropout,
                              drop_last_fc=True)
        
        self.s2_model = TSViT(num_channels=num_channels2, 
                              hidden_dim=hidden_dim, 
                              temporal_depth=temporal_depth, 
                              heads=heads, 
                              dim_head=dim_head, 
                              scale_dim=scale_dim, 
                              dropout=dropout,
                              drop_last_fc=True)
        
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            # nn.LayerNorm(self.hidden_dim),
            nn.Linear(2 * hidden_dim, 1),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=self.dropout),
            # nn.BatchNorm1d(self.hidden_dim),
            # nn.LayerNorm(64),
            # nn.Linear(64, 1)
        )

    def forward(self, inputs):
        """ 
        inputs: {'s1': {'data': B x C x T x W x H, 
                        'time': B x T,
                        'mask': B x T},
                 's2': ...}
        """
        s1_x = inputs['s1']
        s2_x = inputs['s2']

        # Get embeddings of s1 and s2 data
        # Dims: s1_x: B x D
        s1_x = self.s1_model(s1_x)
        s2_x = self.s2_model(s2_x)
        
        # Concatenate embeddings
        # Dims: x: B x 2D
        x = torch.cat([s1_x, s2_x], dim=1)

        # Apply FC Regression layer 
        # Dims: x: B x 1
        x = self.fc(x)
        return x.squeeze().float()
        
        