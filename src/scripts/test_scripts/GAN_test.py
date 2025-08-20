import torch
import os
import matplotlib.pyplot as plt
from src.models.model_classes.GAN import GAN_gen

os.chdir('D:/pythonic-shit/NN_learning/src/models/model_saves/GAN')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_gen = GAN_gen()
model_gen.load_state_dict(state_dict=torch.load('model_gen.tar'))
model_gen.to(device)

model_gen.eval()
n = 2
total = 2*n+1

plt.figure(figsize=(total, total))

num = 1
for i in range(-n, n+1):
  for j in range(-n, n+1):
    ax = plt.subplot(total, total, num)
    num += 1
    h = torch.tensor([[1 * i / n, 1 * j / n]], dtype=torch.float32)
    predict = model_gen(h.to(device))
    predict = predict.detach().squeeze()
    dec_img = predict.cpu().numpy()

    plt.imshow(dec_img, cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()