import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, num_classes=2):
        probs = torch.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
        inter = (probs * targets_oh).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_oh.sum(dim=(2, 3))
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CFCELoss(nn.Module):
    """Categorical Focal Cross Entropy."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        ce_loss = F.cross_entropy(logits, targets.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DSPC_Loss(nn.Module):
    """
    Combined Loss for DSPC-Net as described in the methodology.
    Total Loss = 0.8 * CFCE + 0.2 * Dice
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.cfce = CFCELoss()
        self.dice = DiceLoss()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        l_cfce = self.cfce(logits, targets)
        l_dice = self.dice(logits, targets, num_classes=self.num_classes)

        # Hardcoded weights strictly aligned with the paper
        total_loss = 0.8 * l_cfce + 0.2 * l_dice
        return total_loss