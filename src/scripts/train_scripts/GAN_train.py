import torch
import torchvision.transforms.v2 as tfs
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from src.models.model_classes.GAN import GAN_gen, GAN_dis
from src.datasets.datasets_classes.GANDataset import GANDatasetMNIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Current device is > {device}')

model_gen = GAN_gen().to(device)
model_dis = GAN_dis().to(device)

epochs = 20
hid_dim = 2
batch_size = 16

transforms = tfs.Compose(
    [
        tfs.ToImage(),
        tfs.ToDtype(dtype=torch.flloat32, scale=True)
    ]
)

d_train = GANDatasetMNIST(r'D:\pythonic-shit\NN_learning\src\datasets\datasets_files\MNIST',
                          train=True, transform=transforms)
train_data = data.DataLoader(d_train, batch_size=batch_size,
                             shuffle=True, drop_last=True)

optimizer_gen = optim.Adam(params=model_gen.parameters(), lr=0.001)
optimizer_dis = optim.Adam(params=model_dis.parameters(), lr=0.001)
loss_func = nn.BCEWithLogitsLoss()

targets_0 = torch.zeros(batch_size, 1).to(device)
targets_1 = torch.ones(batch_size, 1).to(device)

loss_gen_lst = []
loss_dis_lst = []

model_gen.train()
model_dis.train()

for epoch in range(epochs):
    loss_mean_gen = 0
    loss_mean_dis = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True)
    for x, y in train_tqdm:
        x = x.to(device)

        h = torch.normal(mean=torch.zeros((batch_size, hid_dim)), std=torch.ones((batch_size, hid_dim)))
        h= h.to(device)

        # generator learning
        img_gen = model_gen(h)
        fake_out = model_dis(img_gen)

        loss_gen = loss_func(fake_out, targets_1)

        optimizer_gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        # discriminator learning
        img_gen = model_gen(h)
        fake_out = model_dis(img_gen)
        real_out = model_dis(x)

        outputs = torch.cat([real_out, fake_out], dim=0).to(device)
        targets = torch.cat([targets_1, targets_0], dim=0).to(device)

        loss_dis = loss_func(outputs, targets)

        optimizer_dis.zero_grad()
        loss_dis.backward()
        optimizer_dis.step()

        lm_count += 1
        loss_mean_gen = 1/lm_count * loss_gen.item() + (1 - 1/lm_count) * loss_mean_gen
        loss_mean_dis = 1/lm_count * loss_dis.item() + (1 - 1/lm_count) * loss_mean_dis

        train_tqdm.set_description(f"Epoch [{epoch+1}/{epochs}], loss_mean_gen={loss_mean_gen:.3f}, loss_mean_dis={loss_mean_dis:.3f}")

    loss_gen_lst.append(loss_mean_gen)
    loss_dis_lst.append(loss_mean_dis)

import os
os.chdir('D:/pythonic-shit/NN_learning/src/models/model_saves/GAN')

st = model_gen.to('cpu').state_dict()
torch.save(st, 'model_gen.tar')

st = model_dis.to('cpu').state_dict()
torch.save(st, 'model_dis.tar')

st = {'loss_gen': loss_gen_lst, 'loss_dis': loss_dis_lst}
torch.save(st, 'model_gan_losses.tar')

