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

    def __init__(self):
        super().__init__()

        self.num_channels = 6
        self.hidden_dim = 128
        self.temporal_depth = 12
        self.heads = 3
        self.dim_head = 64
        self.scale_dim = 2
        self.dropout = 0.1

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

        self.temporal_pool = MeanDimReduction(dims=[1])
        # self.temporal_pool = CLSPooling()

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
        inputs: {'data': B x C x T x H x W, 
                 'time': B x T}
        """
        x = inputs['data']
        xt = inputs['time']

        B, C, T, W, H = x.size()

        # Embedding Layer
        x = self.embedding(x)

        # Create one-hot encoding of each time step (day of year)
        xt = F.one_hot(xt.to(torch.int64), num_classes=365).to(torch.float32)
        xt = xt.reshape(-1, 365)

        # Temporal positional embedding
        pt_emb = self.temporal_positional_embedding(xt)
        pt_emb = pt_emb.reshape(B, T, self.hidden_dim)
        
        # Add temporal positional embedding => x: B x T x Hdim
        x += pt_emb

        # Concat temporal CLS tokens to each batch => x: B x (1+T) x Hdim
        cls_temporal_tokens = repeat(self.temporal_token, '() D -> b 1 D', b=B)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        # Temporal Transformer
        x = self.temporal_transformer(x)

        # Temporal Mean Pooling => x: B x Hdim
        x = self.temporal_pool(x)

        # FC Regression
        x = self.fc(x)
        return x.squeeze().float()
