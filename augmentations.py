import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegFormerAugmentation:
    def __init__(self, image_size=(512, 1024)):
        self.spatial = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=5, border_mode=0, fill=0, fill_mask=255, p=0.35),
                A.RandomCrop(
                    height=int(image_size[0] * 0.95),
                    width=int(image_size[1] * 0.95),
                    pad_if_needed=True,
                    fill=0,
                    fill_mask=255,
                    p=0.35,
                ),
                A.Resize(height=image_size[0], width=image_size[1], p=1.0),
            ],
            additional_targets={"mask": "mask"},
        )
        self.pixel = A.Compose(
            [
                A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.35),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.15),
                A.GaussianBlur(blur_limit=(3, 5), p=0.10),
                A.GaussNoise(std_range=(0.01, 0.05), p=0.08),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def __call__(self, image, target):
        image_np = image.permute(1, 2, 0).cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
        target_np = target.cpu().numpy() if torch.is_tensor(target) else np.asarray(target)

        spatial = self.spatial(image=image_np, mask=target_np)
        return self.pixel(image=spatial["image"])["image"], spatial["mask"].astype(np.uint8)
