from abc import abstractmethod
from torch.utils.data import Dataset, ConcatDataset, ChainDataset, IterableDataset
from PIL import Image
from pathlib import Path
from torchvision import transforms
import json
import numpy as np
import socket
import os


class Txt2ImgIterableBaseDataset(IterableDataset):
    '''
    Define an interface to make the IterableDatasets for text2img data chainable
    '''
    def __init__(self, num_records=0, valid_ids=None, size=256):
        super().__init__()
        self.num_records = num_records
        self.valid_ids = valid_ids
        self.sample_ids = valid_ids
        self.size = size

        print(f'{self.__class__.__name__} dataset contains {self.__len__()} examples.')

    def __len__(self):
        return self.num_records

    @abstractmethod
    def __iter__(self):
        pass


class ImageFolder(Dataset):
    """
    a generic data loader where the images and the annotations (optional) are given

    args:
    data_path: str - all image paths
    transform: torchvision.transforms.Compose - image transforms
    return_text: bool - return text annotations
    pair_json_path: str - path to json file with image annotations recorded
    """
    def __init__(self, data_path, return_text=False, pair_json_path=None, aug=False):
        self.aug = aug

        with open(data_path, "r") as f:
            paths = f.read().splitlines()
        self.samples = paths
        self.return_text = return_text
        
        self.pair_json_path = pair_json_path
        if self.return_text:
            assert self.pair_json_path is not None, f"self.pair_json_path ({self.pair_json_path}) needs to be given to get the image annotations"
            with open(self.pair_json_path, "r") as f:
                self.pair_json_path = json.load(f)
    
    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        width = image.width
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        if self.aug:
            shift = np.random.randint(0, width)
            image = np.roll(image, shift, axis=1)
        # image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        example = {}
        img = self.preprocess_image(self.samples[index])
        if self.return_text:
            example["image"] = img
            example["caption"] = self.pair_json_path[self.samples[index]]
            return example
        else:
            return img

    def __len__(self):
        return len(self.samples)

