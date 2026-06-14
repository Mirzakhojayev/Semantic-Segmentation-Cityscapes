import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegFormerAugmentation:
    """Two-stage augmentation wrapper for segmentation.

    - Spatial transforms: apply to image + mask together.
    - Pixel-level transforms: apply only to the image.
    """

    def __init__(self, image_size=(512, 1024)):
        self.image_size = image_size

        self.spatial = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomScale(scale_limit=0.2, p=0.5),
                A.RandomCrop(
                    height=int(image_size[0] * 0.9),
                    width=int(image_size[1] * 0.9),
                    p=0.4,
                ),
            ],
            additional_targets={"mask": "mask"},
        )

        self.pixel = A.Compose(
            [
                A.RandomBrightnessContrast(p=0.35),
                A.HueSaturationValue(p=0.2),
                A.GaussianBlur(blur_limit=(3, 7), p=0.15),
                A.GaussNoise(std_range=(0.01, 0.05), p=0.1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def __call__(self, image: torch.Tensor, target: torch.Tensor):
        image_np = image.permute(1, 2, 0).cpu().numpy()
        target_np = target.cpu().numpy().astype(np.uint8)

        spatial = self.spatial(image=image_np, mask=target_np)
        image_spatial = spatial["image"]
        target_out = spatial["mask"].astype(np.int64)

        pixel = self.pixel(image=image_spatial)
        image_out = pixel["image"]

        return image_out, torch.from_numpy(target_out)
