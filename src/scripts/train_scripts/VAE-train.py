import torch
import torch.optim as optim
import torchvision
import torchvision.transforms.v2 as tfs
from tqdm import tqdm

from src.models.model_classes.VAE import *

model = VAE.VAEMNIST()