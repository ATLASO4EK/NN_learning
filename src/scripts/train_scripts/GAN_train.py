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
        
