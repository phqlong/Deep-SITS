import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms

def freezing(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Resnet_LSTM(nn.Module):
    def __init__(self,
                 num_features=6,
                 hidden_channel=64,
                 kernel_size=3,
                 hidden_size1=64, 
                 hidden_size2=64, 
                 num_layers=3
                ):
        super().__init__()

        # ResNet feature extractor
        self.resnet = models.resnet18(pretrained=True)

        # Freeze all layers
        self.resnet = freezing(self.resnet)

        # Replace first layer with 6 channels
        self.resnet.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace last layer with 256 channels
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 128)

        # LSTM Layers
        self.lstm = nn.LSTM(128, hidden_size1, num_layers, batch_first=True, dropout=0.3)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(10 * 64, 1),
            # nn.ReLU(inplace=True),
            # nn.Linear(hidden_size2, 1)
        )
        
    def forward(self, x):
        B, C, T, W, H = x.size()

        # 2D Convolutional layers
        x = [self.resnet(x[:, :, t, :, :]) for t in range(T)]
        x = torch.stack(x)

        # Flatten output of LSTM layers
        x = x.contiguous().view(B, T, -1)

        # LSTM layer
        x, (h_n, c_n) = self.lstm(x)

        # Flatten output of FC layers
        x = x.contiguous().view(B, -1)

        # Fully connected layers
        x = self.fc(x)        
        return x.squeeze().float()