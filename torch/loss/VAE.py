import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure


class ssimloss(nn.Module):
    def __init__(self):
        super(ssimloss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def forward(self, preds, targets):
        loss = 1 - self.ssim(preds, targets)
        return loss