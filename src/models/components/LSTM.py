import torch
from torch import nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class LSTM(nn.Module):
    def __init__(self,
                 num_features=6,
                 hidden_channel=32,
                 kernel_size=3,
                 hidden_size1=64, 
                 hidden_size2=64, 
                 num_layers=3, 
                 dropout=0.1,
                ):
        super().__init__()
        
        # LSTM Layers
        self.lstm = nn.LSTM(4*num_features, hidden_channel, num_layers, batch_first=True, dropout=dropout, bidirectional=False)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(10 * hidden_channel, hidden_size1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(hidden_size1, momentum=0.1, affine=True),
            
            # nn.Linear(hidden_size1, hidden_size2),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.3),
            # nn.BatchNorm1d(hidden_size2, momentum=0.1, affine=True),

            nn.Linear(hidden_size2, 1)
        )
        
    def forward(self, x):
        B, C, T, W, H = x.size()

        # Calculate mean, median, max, and min across W and H axes
        mean_tensor = x.mean(dim=[3, 4])
        median_tensor = x.median(dim=3).values.median(dim=3).values
        max_tensor = x.max(dim=3).values.max(dim=3).values
        min_tensor = x.min(dim=3).values.min(dim=3).values

        # Concatenate the four tensors along the channel axis
        x = torch.cat([mean_tensor, median_tensor, max_tensor, min_tensor], dim=1)

        # Flatten output of LSTM layers
        x = x.view(B, T, 4*C)

        # LSTM layer
        x, (h_n, c_n) = self.lstm(x)

        # Flatten output of FC layers
        x = x.contiguous().view(B, -1)
        # print(x.shape)

        # Fully connected layers
        x = self.fc(x)        
        return x.squeeze().float()