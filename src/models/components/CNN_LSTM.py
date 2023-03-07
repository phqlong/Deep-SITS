import torch
from torch import nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CNN_LSTM(nn.Module):
    def __init__(self,
                 num_features=6,
                 hidden_channel=64,
                 kernel_size=3,
                 hidden_size1=32, 
                 hidden_size2=64, 
                 num_layers=3, 
                ):
        super().__init__()
        
        # 2D Convolutional layers for spacial
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=num_features, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.BatchNorm2d(32, momentum=1, affine=True),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.BatchNorm2d(64, momentum=1, affine=True),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.BatchNorm2d(128, momentum=1, affine=True),

            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
       )
                
        # LSTM Layers
        self.lstm = nn.LSTM(10 * 2 * 2, hidden_size1, num_layers, batch_first=True, dropout=0.3)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(64 * hidden_size1, hidden_size2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size2, 1)
        )
        
    def forward(self, x):
        B, C, T, W, H = x.size()

        # 2D Convolutional layers
        x = [self.conv1(x[:, :, t, :, :]) for t in range(T)]
        x = torch.stack(x)
                        
        # Flatten output of LSTM layers
        x = x.contiguous().view(B, 64, -1)

        # LSTM layer
        x, (h_n, c_n) = self.lstm(x)

        # Flatten output of FC layers
        x = x.contiguous().view(B, -1)

        # Fully connected layers
        x = self.fc(x)        
        return x.squeeze().float()