import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import DoubleConv, Down, Up, FiLM


class GameUNet(nn.Module):
    def __init__(self, n_classes: int = 3, n_actions: int = 4):
        super().__init__()

        self.inc = DoubleConv(n_classes, 32)
        self.down1 = Down(32, 64)
        self.bottleneck = DoubleConv(64, 64)
        self.up1 = Up(128, 32)
        self.up2 = Up(64, 32)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

        self.film1 = FiLM(n_actions, 32)
        self.film2 = FiLM(n_actions, 64)
        self.film3 = FiLM(n_actions, 64)
        self.film4 = FiLM(n_actions, 32)
        self.film5 = FiLM(n_actions, 32)

    def forward(self, x, a):
        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.bottleneck(x2)
        x3 = self.film3(x3, a)

        x = self.up1(x3, x2)
        x = self.film4(x, a)

        x = self.up2(x, x1)
        x = self.film5(x, a)

        return self.outc(x)
