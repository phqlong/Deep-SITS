import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from src.models.components.transformer import Transformer


class MeanDimReduction(nn.Module):
    def __init__(self, dims=[3, 4]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        # Mean dims W, H ==> x: B x C x T
        return x.mean(dim=self.dims)


class CLSPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # B, 1+T, D ==> x: B x 1 x D
        return x[:, 0, :]


class TSViT(nn.Module):
    """
    Temporal-Spatial ViT for object classification (used in main results, section 4.3)
    """

    def __init__(self,
                 num_channels: int = 6,
                 hidden_dim: int = 128,
                 temporal_depth: int = 12,
                 heads: int = 3,
                 dim_head: int = 64,
                 scale_dim: int = 2,
                 dropout: float = 0.1,
                 drop_last_fc: bool = False
                ):
        super().__init__()

        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.temporal_depth = temporal_depth
        self.heads = heads
        self.dim_head = dim_head
        self.scale_dim = scale_dim
        self.dropout = dropout
        self.drop_last_fc = drop_last_fc

        self.embedding = nn.Sequential(
            MeanDimReduction(dims=[3, 4]),
            Rearrange('B C T -> B T C'),
            nn.Linear(self.num_channels, self.hidden_dim)
        )
        
        self.temporal_token = nn.Parameter(torch.randn(1, self.hidden_dim))

        self.temporal_positional_embedding = nn.Linear(365, self.hidden_dim)

        self.temporal_transformer = Transformer(dim=self.hidden_dim, 
                                                depth=self.temporal_depth, 
                                                heads=self.heads, 
                                                dim_head=self.dim_head,
                                                mlp_dim=self.hidden_dim * self.scale_dim, 
                                                dropout=self.dropout)

        self.temporal_pool = MeanDimReduction(dims=[0])
        # self.temporal_pool = CLSPooling()

        if not self.drop_last_fc:
            # Fully connected layers
            self.fc = nn.Sequential(
                nn.Dropout(p=self.dropout),
                # nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, 1),
                # nn.ReLU(inplace=True),
                # nn.Dropout(p=self.dropout),
                # nn.BatchNorm1d(self.hidden_dim),
                # nn.LayerNorm(64),
                # nn.Linear(64, 1)
            )


    def forward(self, inputs):
        """ 
        inputs: {'data': B x C x T x W x H, 
                 'time': B x T,
                 'mask': B x T}
        """
        x = inputs['data']
        x_time = inputs['time']
        x_mask = inputs['mask']

        B, C, T, W, H = x.size()

        # Embedding Layer
        x = self.embedding(x)

        # Create one-hot encoding of each time step (day of year)
        x_time = F.one_hot(x_time.to(torch.int64), num_classes=365).to(torch.float32)
        x_time = x_time.reshape(-1, 365)

        # Temporal positional embedding
        pt_emb = self.temporal_positional_embedding(x_time)
        pt_emb = pt_emb.reshape(B, T, self.hidden_dim)
        
        # Add temporal positional embedding => x: B x T x Hdim
        x += pt_emb

        # Concat temporal CLS tokens to each batch => x: B x (1+T) x Hdim
        cls_temporal_tokens = repeat(self.temporal_token, '() D -> b 1 D', b=B)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        # Temporal Transformer => x: B x (1+T) x Hdim
        x = self.temporal_transformer(x)

        # Mask out by x_mask & Temporal Mean Pooling => x: B x Hdim
        x = torch.stack([self.temporal_pool(x[i][x_mask[i]]) for i in range(B)])

        # Return Embedding if drop_last_fc
        if self.drop_last_fc:
            return x
        
        # FC Regression
        x = self.fc(x)
        return x.squeeze().float()
