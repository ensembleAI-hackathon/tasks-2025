from typing import Tuple
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TaskDataset(Dataset):
    def __init__(self, transform=None):

        self.ids = []
        self.imgs = []
        self.labels = []

        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if not self.transform is None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)
    
t = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
    ]
)