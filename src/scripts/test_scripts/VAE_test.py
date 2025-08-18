import torch
import torchvision
import torchvision.transforms.v2 as tfs
import matplotlib.pyplot as plt

from src.models.model_classes.VAE import VAEMNIST

import os

dev = torch.device('cpu')

model = VAEMNIST(784, 784, 2).to(dev)
os.chdir('D:/pythonic-shit/NN_learning/src/models/model_saves/VAE')
st = torch.load('VAE.tar', weights_only=True)
model.load_state_dict(st)

transforms = tfs.Compose(
    [
        tfs.ToImage(),
        tfs.ToDtype(dtype=torch.float32, scale=True),
        tfs.Lambda(lambda _img: _img.ravel())
    ]
)

model.eval()

d_test = torchvision.datasets.MNIST(r'D:\pythonic-shit\NN_learning\src\datasets\datasets_files\MNIST', download=True, train=False, transform=transforms)
x_data = transforms(d_test.data).view(len(d_test), -1)

_, h, _, _ = model(x_data)
h = h.detach().numpy()

plt.scatter(h[:, 0], h[:, 1])
plt.grid()


n = 5
total = 2*n+1

plt.figure(figsize=(total, total))

num = 1
for i in range(-n, n+1):
    for j in range(-n, n+1):
        ax = plt.subplot(total, total, num)
        num += 1
        h = torch.tensor([3*i/n, 3*j/n], dtype=torch.float32)
        predict = model.decoder(h.unsqueeze(0))
        predict = predict.detach().squeeze(0).view(28, 28)
        dec_img = predict.numpy()

        plt.imshow(dec_img, cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.show()