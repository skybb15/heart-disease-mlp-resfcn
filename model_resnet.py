# model_resnet.py
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)

class ResFCNet(nn.Module):
    def __init__(self, input_dim):
        super(ResFCNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU()
        )
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.output_layer = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.output_layer(x)
