# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing   = smoothing
        self.reduction = reduction
        self.weight    = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
         if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target, mask=None):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        log_preds = torch.nan_to_num(log_preds, posinf=0., neginf=0.)
        loss = -log_preds
        if mask is not None and mask.dtype == torch.bool:
            mask = 1 - mask.float()
            loss = loss * mask
        loss = loss.sum(-1)
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)
