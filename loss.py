import torch
import torch.nn as nn

from dataset import IGNORE_INDEX


class SegmentationLoss(nn.Module):
    def __init__(self, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets.long())
