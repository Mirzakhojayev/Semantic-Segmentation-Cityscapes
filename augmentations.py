import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_augmentation(image_size=(512, 1024)):
    spatial = A.Compose([
        A.RandomScale(scale_limit=(-0.5, 1.0), p=1.0),
        A.PadIfNeeded(min_height=image_size[0], min_width=image_size[1],
                      border_mode=0, value=0, mask_value=255, p=1.0),
        A.RandomCrop(height=image_size[0], width=image_size[1], p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=5, border_mode=0, fill=0, fill_mask=255, p=0.3),
    ], additional_targets={"mask": "mask"})

    pixel = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.35),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.15),
        A.GaussianBlur(blur_limit=(3, 5), p=0.10),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.08),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    def augment(image, mask):
        image_np = np.asarray(image)
        mask_np  = np.asarray(mask)
        out = spatial(image=image_np, mask=mask_np)
        return pixel(image=out["image"])["image"], out["mask"].astype(np.uint8)

    return augment
