from torch.utils.data import Dataset
from taming.data.base import ImagePaths, ImagePathsRect
import socket
import os


class CustomBase(Dataset):
    def __init__(self, data_path, augment=False):
        super().__init__()
        with open(data_path, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, random_crop=False, augment=augment)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example

class CustomTrain(CustomBase):
    def __init__(self, path, augment=False):
        super().__init__(data_path=path, augment=augment)


class CustomTest(CustomBase):
    def __init__(self, path, augment=False):
        super().__init__(data_path=path, augment=augment)


class CustomBaseRect(Dataset):
    def __init__(self, data_path, h, w, random_crop=True, augment=False):
        super().__init__()
        with open(data_path, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePathsRect(paths=paths, h=h, w=w, random_crop=random_crop, augment=augment)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example


class CustomTrainCrop(CustomBaseRect):
    def __init__(self, h, w, path):
        super().__init__(data_path=path, h=h, w=w)


class CustomTestCrop(CustomBaseRect):
    def __init__(self, h, w, path):
        super().__init__(data_path=path, h=h, w=w, random_crop=False)


