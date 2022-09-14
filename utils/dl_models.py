import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import utils


class Block(nn.Module):
    def __init__(self, inp, out, ks):
        super().__init__()
        self.activ = F.gelu
        self.conv1 = nn.Conv1d(inp, out, ks, padding=1)
        self.bn1 = nn.BatchNorm1d(out)

    def forward(self, x):
        x = self.activ(self.bn1(self.conv1(x)))
        return x

class ConvMlpNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop_p = 0.1
        self.h_dim = 240
        self.h_dims = [1, 32, 64, self.h_dim]
        self.backbone = nn.Sequential(*[Block(self.h_dims[i], self.h_dims[i + 1], 3) for i in range(len(self.h_dims) - 1)])
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(p=self.drop_p)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.lin = nn.Sequential(nn.Linear(self.h_dim, self.h_dim * 2),
                                 nn.BatchNorm1d(self.h_dim * 2),
                                 nn.GELU(),
                                 nn.Dropout(p=self.drop_p),
                                 nn.Linear(self.h_dim * 2, self.h_dim // 2),
                                 nn.BatchNorm1d(self.h_dim // 2),
                                 nn.GELU(),
                                 nn.Dropout(p=self.drop_p),
                                 nn.Linear(self.h_dim // 2, 4))

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.drop(x)
        x = self.lin(x)
        return x



class MlpNetLight(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.drop_p = 0.1
        self.drop = nn.Dropout(p=self.drop_p)
        self.lin = nn.Sequential(
                                 nn.Linear(in_features, in_features * 8),
                                 nn.BatchNorm1d(in_features * 8),
                                 nn.GELU(),
                                 nn.Linear(in_features * 8, in_features * 32),
                                 nn.BatchNorm1d(in_features * 32),
                                 nn.GELU(),
                                 nn.Linear(in_features * 32, in_features * 8), 
                                 nn.BatchNorm1d(in_features * 8),
                                 nn.GELU(),
                                 nn.Dropout(p=self.drop_p),
                                 nn.Linear(in_features * 8, in_features * 4),
                                 nn.BatchNorm1d(in_features * 4),
                                 nn.GELU(),
                                 nn.Dropout(p=self.drop_p),
                                 nn.Linear(in_features * 4, 4)
                            )

    def forward(self, x):
        x = self.lin(x)
        return x



def main():

    mlp_model = MlpNeLight(in_features=16)
    summary(mlp_model, (16, ), 4)

    print()

    cnn_model = ConvMlpNet()
    summary(cnn_model, (1, 16, ), 4)


if __name__ == "__main__":
    main()