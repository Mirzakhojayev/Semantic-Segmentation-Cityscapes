from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from augmentations import get_augmentation


IGNORE_INDEX = 255

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic_light", "traffic_sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]

CITYSCAPES_COLORS = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
    (0, 80, 100), (0, 0, 230), (119, 11, 32),
]


class CityscapesKaggleDataset(Dataset):
    def __init__(self, root="dataset", split="train", image_size=(512, 1024)):
        self.image_dir = Path(root) / split / "img"
        self.label_dir = Path(root) / split / "label_indexed"
        self.image_paths = sorted(self.image_dir.glob("*.png"))
        self.label_paths = sorted(self.label_dir.glob("*.png"))

        self.image_tf = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.label_tf = T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST)
        self.augment  = get_augmentation(image_size) if split == "train" else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask  = Image.open(self.label_paths[idx]).convert("L")

        if self.augment:
            img_t, mask_np = self.augment(image, mask)
            return img_t.contiguous(), torch.as_tensor(mask_np, dtype=torch.long)

        return self.image_tf(image), torch.as_tensor(np.array(self.label_tf(mask)), dtype=torch.long)
