from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CityscapesKaggleDataset, IGNORE_INDEX
from model import SegFormer


NUM_CLASSES = 19
ROOT = "dataset"
IMAGE_SIZE = (512, 1024)
EPOCHS = 140
WARMUP_EPOCHS = 10
BATCH_SIZE = 16
LR = 3e-4
WEIGHT_DECAY = 0.01
NUM_WORKERS = 4


def compute_miou(logits, targets):
    preds = logits.argmax(dim=1)[targets != IGNORE_INDEX]
    labels = targets[targets != IGNORE_INDEX]
    if preds.numel() == 0:
        return 0.0

    n = NUM_CLASSES
    idx = labels * n + preds
    cm = torch.bincount(idx, minlength=n * n).reshape(n, n)
    tp = cm.diagonal()
    union = cm.sum(0) + cm.sum(1) - tp
    present = union > 0
    return (tp[present].float() / union[present].float()).mean().item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Preprocess labels if needed
    if not (Path(ROOT) / "train" / "label_indexed").exists():
        from preprocess_labels import preprocess_all
        preprocess_all(ROOT)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        CityscapesKaggleDataset(ROOT, "train", IMAGE_SIZE),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=pin, persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        CityscapesKaggleDataset(ROOT, "val", IMAGE_SIZE),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=pin, persistent_workers=NUM_WORKERS > 0,
    )

    model = SegFormer(num_classes=NUM_CLASSES).to(device)

    # Load pretrained encoder weights if available
    ckpt_path = Path("segformer_b2_pretrained_encoder.pth")
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        state = state.get("model", state.get("state_dict", state))
        target = model if any(k.startswith("encoder.")
                              for k in state) else model.encoder
        missing, unexpected = target.load_state_dict(state, strict=False)
        print(
            f"Loaded pretrained weights — missing: {len(missing)}, unexpected: {len(unexpected)}")
    else:
        print("WARNING: No pretrained weights found, training from scratch.")

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = AdamW([
        {"params": model.encoder.parameters(),     "lr": LR * 0.1},
        {"params": model.decode_head.parameters(), "lr": LR},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.01,
                 end_factor=1.0, total_iters=WARMUP_EPOCHS),
        CosineAnnealingLR(optimizer, T_max=EPOCHS -
                          WARMUP_EPOCHS, eta_min=LR * 0.01),
    ], milestones=[WARMUP_EPOCHS])

    use_amp = torch.cuda.is_available(
    ) and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    scaler = GradScaler(enabled=False)
    print(f"AMP: {use_amp}  dtype: {amp_dtype}")

    best_miou = -1.0
    best_path = Path("checkpoints/best_miou.pt")
    best_path.parent.mkdir(exist_ok=True)

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        t_loss, t_miou = 0.0, 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]", leave=False):
            images, targets = images.to(device, non_blocking=True), targets.to(
                device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            t_loss += loss.detach().item()
            t_miou += compute_miou(logits.detach(), targets)
        scheduler.step()
        t_loss /= len(train_loader)
        t_miou /= len(train_loader)

        # ── Validate (every 2 epochs after ep50, every 10 before, always on last) ──
        do_val = ((epoch + 1) < 50 and (epoch + 1) % 10 == 0) \
            or ((epoch + 1) >= 50 and (epoch + 1) % 2 == 0) \
            or (epoch + 1) == EPOCHS
        if do_val:
            model.eval()
            v_loss, v_miou = 0.0, 0.0
            with torch.no_grad():
                for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]", leave=False):
                    images, targets = images.to(device, non_blocking=True), targets.to(
                        device, non_blocking=True)
                    with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                        logits = model(images)
                        loss = criterion(logits, targets)
                    v_loss += loss.item()
                    v_miou += compute_miou(logits, targets)
            v_loss /= len(val_loader)
            v_miou /= len(val_loader)
            print(
                f"Epoch {epoch+1:03d} | train_loss={t_loss:.4f} train_mIoU={t_miou:.4f} | val_loss={v_loss:.4f} val_mIoU={v_miou:.4f}")
            if v_miou > best_miou:
                best_miou = v_miou
                torch.save(
                    {"epoch": epoch + 1, "model": model.state_dict(), "miou": best_miou}, best_path)
                print(f"  ✓ Saved best checkpoint (mIoU={best_miou:.4f})")
        else:
            print(
                f"Epoch {epoch+1:03d} | train_loss={t_loss:.4f} train_mIoU={t_miou:.4f}")


if __name__ == "__main__":
    main()
