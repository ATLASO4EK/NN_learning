import torch.nn as nn

class GAN_gen(nn.Module):
    def __init__(self):
        super().__init__()

        self.emb = nn.Sequential(
            nn.Linear(2, 512 * 7 * 7, bias=False),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(512 * 7 * 7),
            nn.Unflatten(1, (512, 7, 7)),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(512, 256, 5, 1, padding='same', bias=False),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 5, 1, padding='same', bias=False),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(128)
        )

        self.transpose = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 4, 2, padding=1, bias=False),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(32)
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 1, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        embeded = self.emb(x)
        out_conv = self.conv(embeded)
        out_transpose = self.transpose(out_conv)
        out = self.out_conv(out_transpose)

        return out

class GAN_dis(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 5, 2, padding=2, bias=False),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, 2, padding=2, bias=False),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Flatten()
        )

        self.lin_layer = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1)
        )

    def forward(self, x):
        out_conv = self.conv(x)
        out = self.lin_layer(out_conv)

        return out