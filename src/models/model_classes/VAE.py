import torch
import torch.nn as nn

class VAEMNIST(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, hid_dim:int):
        super().__init__()
        self.hid_dim = hid_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64),
        )

        self.h_mean = nn.Linear(64, self.hid_dim)
        self.h_log_var = nn.Linear(64, self.hid_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.hid_dim, 64),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        enc = self.encoder(x)

        h_mean = self.h_mean(enc)
        h_log_var = self.h_log_var(enc)

        noise = torch.normal(mean=torch.zeros_like(h_mean), std=torch.ones_like(h_log_var))
        h = noise * torch.exp(h_log_var / 2) + h_mean

        x = self.decoder(h)

        return x, h, h_mean, h_log_var

class VAELoss(nn.Module):
    def forward(self, x, y, h_mean, h_log_var):
        img_loss = torch.sum(torch.square(x - y), dim=-1)
        kl_loss = -0.5 * torch.sum(1 + h_log_var - torch.square(h_mean) - torch.exp(h_log_var), dim=-1)
        return torch.mean(img_loss + kl_loss)