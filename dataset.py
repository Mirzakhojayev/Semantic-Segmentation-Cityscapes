from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F


# Standard Cityscapes 19-class palette (Kaggle 5K version uses the same colors).
CITYSCAPES_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic_light",
    "traffic_sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

CITYSCAPES_COLORS = [
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
]

IGNORE_INDEX = 255


class CityscapesKaggleDataset(Dataset):
    """Cityscapes (5K Kaggle) image/label loader for SegFormer-style training.

    - Reads paired RGB images and colorized label masks from dataset/train or dataset/val.
    - Resizes images and masks to `image_size`.
    - Converts RGB masks to class indices 0..18 with `IGNORE_INDEX=255` for unknown/unlabeled pixels.
    - Normalizes images with ImageNet mean/std.
    """

    def __init__(
        self,
        root: str = "dataset",
        split: str = "train",
        image_size: tuple[int, int] = (512, 1024),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.mean = mean
        self.std = std

        self.image_dir = self.root / split / "img"
        self.label_dir = self.root / split / "label"

        if not self.image_dir.exists() or not self.label_dir.exists():
            raise FileNotFoundError(
                f"Could not find dataset folders under {self.root / split}. "
                f"Expected: {self.image_dir} and {self.label_dir}"
            )

        self.image_paths = sorted(self.image_dir.glob("*.png"))
        self.label_paths = sorted(self.label_dir.glob("*.png"))

        if len(self.image_paths) != len(self.label_paths):
            raise ValueError(
                f"Mismatched image/label counts in {self.root / split}: "
                f"{len(self.image_paths)} images vs {len(self.label_paths)} labels"
            )

        self.color_to_label = {color: idx for idx, color in enumerate(CITYSCAPES_COLORS)}

        self.image_tf = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )
        self.label_tf = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def _color_mask_to_indices(self, mask: Image.Image) -> torch.Tensor:
        """Convert a colorized label image to a class-index tensor."""
        mask_np = np.asarray(mask.convert("RGB"), dtype=np.uint8)
        h, w, _ = mask_np.shape

        flat = mask_np.reshape(-1, 3)
        labels = np.full((flat.shape[0],), IGNORE_INDEX, dtype=np.int64)

        for color, class_id in self.color_to_label.items():
            match = (flat[:, 0] == color[0]) & (flat[:, 1] == color[1]) & (flat[:, 2] == color[2])
            labels[match] = class_id

        return torch.from_numpy(labels.reshape(h, w))

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        if image_path.stem != label_path.stem:
            raise ValueError(
                f"Mismatched pair at index {idx}: "
                f"{image_path.name} != {label_path.name}"
            )

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(label_path).convert("RGB")

        image = self.image_tf(image)
        mask = self.label_tf(mask)
        target = self._color_mask_to_indices(mask)

        return image, target
