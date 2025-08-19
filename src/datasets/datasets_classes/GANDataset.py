import torch
import torch.utils.data as data
import torchvision

class GANDatasetMNIST(data.Dataset):
    def __init__(self, path, train=True, target=5, transform=None):
        _dataset = torchvision.datasets.MNIST(path, download=True, train=train)
        self.dataset = _dataset.data[_dataset.targets==target]
        self.length = self.dataset.size(0)
        self.target = torch.tensor([target], dtype=torch.float32)

        if transform:
            self.dataset = transform(self.dataset).view(-1,1,28,28)

    def __getitem__(self, item):
        return self.dataset[item], self.target

    def __len__(self):
        return self.length
