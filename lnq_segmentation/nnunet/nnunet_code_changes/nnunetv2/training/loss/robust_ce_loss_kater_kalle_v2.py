import torch
from torch import nn, Tensor
import numpy as np

# class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
#     """
#     this is just a compatibility layer because my target tensor is float and has an extra dimension
#
#     input must be logits, not probabilities!
#     """
#     def forward(self, input: Tensor, target: Tensor, loss_mask: Tensor = None, epoch: int = 0) -> Tensor:
#         if len(target.shape) == len(input.shape):
#             assert target.shape[1] == 1
#             target = target[:, 0]
#         return super().forward(input, target.long())

class RobustCrossEntropyLoss(nn.Module):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, label_smoothing: float = 0, pseudo_labeling: bool = False, T1: int = 0, T2: int = 0):
        super(RobustCrossEntropyLoss, self).__init__()
        self.pseudo_labeling = pseudo_labeling
        reduction = "mean"

        if self.pseudo_labeling:
            reduction = "none"
            self.T1 = T1
            self.T2 = T2

        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, label_smoothing=label_smoothing, reduction=reduction)

    def forward(self, input: Tensor, target: Tensor, loss_mask: Tensor = None, epoch: int = 0) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]

        # print('ce loss forward')
        #
        # print('input.size()')
        # print(input.size())
        #
        # print('target.size()')
        # print(target.size())
        #
        # print('loss_mask.size()')
        # print(loss_mask.size())

        if self.pseudo_labeling:
            # target is already adjusted to pseudo label

            pseudo_mask = torch.where(loss_mask == False, True, False)
            max_input = torch.max(torch.clone(input).detach(), dim=1, keepdim=True).values
            confidence_mask = torch.where(max_input >= 0.95, True, False) * pseudo_mask

            loss_all = self.ce_loss(input, target.long())

            # print('loss_all.size()')
            # print(loss_all.size())

            loss_labeled = loss_all * loss_mask
            loss_pseudo = loss_all * confidence_mask

            # bs=2
            # num_voxels = bs * pseudo_mask.size(-1) * pseudo_mask.size(-2) * pseudo_mask.size(-3)
            # pseudo_voxel_weighting = (num_voxels - torch.count_nonzero(confidence_mask)) / num_voxels
            # labeled_voxel_weighting = (num_voxels - torch.count_nonzero(loss_mask)) / num_voxels
            # print()
            # print(labeled_voxel_weighting)
            # print(pseudo_voxel_weighting)

            if epoch < self.T1:
                alpha = 0
            elif epoch >= self.T1 and epoch < self.T2:
                alpha = ((epoch - self.T1) / (self.T2-self.T1))*1
            else:
                alpha = 1

            labeled_voxel_weighting = 1
            pseudo_voxel_weighting = 1

            ce = (labeled_voxel_weighting * loss_labeled.mean()) + (alpha * pseudo_voxel_weighting * loss_pseudo.mean())
            ce = ce.mean()
        else:
            ce = self.ce_loss(input, target.long())
        return ce


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()

