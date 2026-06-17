from pathlib import Path

import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader

from dataset import CityscapesKaggleDataset, IGNORE_INDEX
from loss import SegmentationLoss
from model import SegFormer


NUM_CLASSES = 19
ROOT = "dataset"
IMAGE_SIZE = (512, 1024)
EPOCHS = 100
BATCH_SIZE = 16
LR = 3e-4
WEIGHT_DECAY = 0.01
NUM_WORKERS = 4


def build_dataloaders():
    train_ds = CityscapesKaggleDataset(root=ROOT, split="train", image_size=IMAGE_SIZE, use_augmentations=True)
    val_ds = CityscapesKaggleDataset(root=ROOT, split="val", image_size=IMAGE_SIZE, use_augmentations=False)

    pin_memory = torch.cuda.is_available()
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin_memory),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_memory),
    )


def compute_miou(logits, targets):
    preds = logits.argmax(dim=1)
    mask = targets != IGNORE_INDEX

    preds = preds[mask]
    targets = targets[mask]

    if preds.numel() == 0:
        return 0.0

    intersection = torch.zeros(
        NUM_CLASSES, device=preds.device, dtype=torch.long)
    union = torch.zeros(NUM_CLASSES, device=preds.device, dtype=torch.long)

    for cls in range(NUM_CLASSES):
        pred_mask = preds == cls
        target_mask = targets == cls
        inter = (pred_mask & target_mask).sum().item()
        union_i = (pred_mask | target_mask).sum().item()
        intersection[cls] = inter
        union[cls] = union_i

    miou = (intersection / union.clamp_min(1)).sum().item() / NUM_CLASSES
    return miou


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if not (Path(ROOT) / "train" / "label_indexed").exists() or not (Path(ROOT) / "val" / "label_indexed").exists():
        from preprocess_labels import preprocess_all
        preprocess_all(ROOT)

    train_loader, val_loader = build_dataloaders()
    model = SegFormer(num_classes=NUM_CLASSES)

    model.to(device)

    criterion = SegmentationLoss(ignore_index=IGNORE_INDEX)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = PolynomialLR(optimizer, total_iters=EPOCHS, power=0.9)

    use_amp = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)
    best_miou = -1.0
    best_path = Path("checkpoints") / "best_miou.pt"
    best_path.parent.mkdir(exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        model.train()
        train_loss = 0.0
        train_miou = 0.0
        num_train_batches = 0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [train]", leave=False):

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.detach().item()
            train_miou += compute_miou(logits.detach(), targets)
            num_train_batches += 1

        train_loss /= max(1, num_train_batches)
        train_miou /= max(1, num_train_batches)

        model.eval()
        val_loss = 0.0
        val_miou = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [val]", leave=False):

                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with autocast(device_type="cuda", enabled=use_amp):
                    logits = model(images)
                    loss = criterion(logits, targets)

                val_loss += loss.item()
                val_miou += compute_miou(logits, targets)
                num_val_batches += 1

        val_loss /= max(1, num_val_batches)
        val_miou /= max(1, num_val_batches)

        scheduler.step()

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train_loss={train_loss:.4f} | train_mIoU={train_miou:.4f} | "
            f"val_loss={val_loss:.4f} | val_mIoU={val_miou:.4f}"
        )

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "miou": best_miou,
            }, best_path)
            print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
