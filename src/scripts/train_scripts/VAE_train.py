import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms.v2 as tfs
from tqdm import tqdm

from src.models.model_classes.VAE import VAEMNIST, VAELoss

import os

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = VAEMNIST(784, 784, 2).to(dev)

transforms = tfs.Compose(
    [
        tfs.ToImage(),
        tfs.ToDtype(dtype=torch.float32, scale=True),
        tfs.Lambda(lambda _img: _img.ravel())
    ]
)

d_train = torchvision.datasets.MNIST(r'D:\pythonic-shit\NN_learning\src\datasets\datasets_files\MNIST',
                                     download=True, train=True, transform=transforms)
train_data = data.DataLoader(d_train, batch_size=100, shuffle=True)

optimizer = optim.Adam(params=model.parameters(), lr=0.001)
loss_func = VAELoss().to(dev)

epochs = 10
model.train()

for epoch in range(epochs):
    loss_mean = 0
    lm_count = 0

    t_tqdm = tqdm(train_data, leave=True)
    for x, y in t_tqdm:
        pr, _, h_mean, h_log_var = model(x.to(dev))
        loss = loss_func(pr, x.to(dev), h_mean, h_log_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1/lm_count * loss.item() + (1 - 1/lm_count) * loss_mean
        t_tqdm.set_description(f'Epoch[{epoch+1}/{epochs}] l_mean={loss_mean:.4f}')

os.chdir('D:/pythonic-shit/NN_learning/src/models/model_saves/VAE')
torch.save(model.to('cpu').state_dict(), f'VAE.tar')