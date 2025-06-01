import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure


class ssimloss(nn.Module):
    def __init__(self):
        super(ssimloss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def forward(self, preds, targets):
        loss = 1 - self.ssim(preds, targets)
        return loss


class WeightedSSIMLoss(nn.Module):
    def __init__(self, alpha=0.8, lesion_weight=2.0, data_range=1.0):
        super().__init__()
        self.alpha = alpha
        self.lesion_weight = lesion_weight
        self.data_range = data_range
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")

    def forward(self, y_pred: torch.Tensor, 
                      y_true: torch.Tensor, 
                      lesion_mask: torch.Tensor) -> torch.Tensor:
        # ssim_val = ssim(y_pred, y_true, data_range=self.data_range, size_average=True)
        ssim_val = self.ssim(y_pred, y_true)
        ssim_loss = 1.0 - ssim_val

        mae_map = torch.abs(y_true - y_pred)          # [B, C, H, W]

        # mae_map = mae_map.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        weight_map = 1.0 + (self.lesion_weight - 1.0) * lesion_mask
        weighted_mae = (mae_map * weight_map).mean()

        loss = self.alpha * ssim_loss + (1.0 - self.alpha) * weighted_mae
        return loss