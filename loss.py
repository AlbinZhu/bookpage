'''
Author: bin.zhu
Date: 2022-07-04 14:38:58
LastEditors: Albin
LastEditTime: 2022-07-15 14:23:00
Description: file content
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLSSigmoid(nn.Module):

    def __init__(self, cls_weights) -> None:
        super().__init__()
        self.weight = torch.tensor(cls_weights).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        normalizer = float(target.shape[0])
        normalizer = torch.maximum(torch.tensor(1.0), torch.tensor(normalizer))
        # labels = target.reshape(target.shape[0], -1, 1)
        # input = input.reshape(input.shape[0], -1, 1)
        bce_loss = F.binary_cross_entropy(input, target, self.weight, reduction="sum")
        loss = bce_loss / normalizer
        return loss


class RegLoss(nn.Module):

    def __init__(self, poly_weight) -> None:
        super().__init__()
        self.weight = poly_weight

    def forward(self, input, target):
        regression = torch.reshape(input, (input.shape[0], 2, -1))
        regression_target = target[:, :, :-1]
        anchor_state = target[:, :, -1]
        label_mask = (anchor_state == 1).unsqueeze(2)
        labels = torch.masked_select(regression_target, label_mask)
        labels = labels.view(-1, 3)

        batch_index = torch.nonzero(label_mask)[:, 0]
        regression_indices = labels[..., 2].long()
        regression_target = labels[..., 0:2]
        regression = regression[batch_index,:, regression_indices]
        normalizer = float(regression_indices.shape[0])
        normalizer = torch.maximum(torch.tensor(1.0), torch.tensor(normalizer))
        poly_loss = F.smooth_l1_loss(regression,
                                     regression_target,
                                     reduction='sum')
        loss = poly_loss / normalizer * self.weight
        return loss


class HeatmapFocalLoss(nn.Module):

    def __init__(self, num_channels, weights=1.0, alpha=2.0, beta=4.0) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.weights = weights
        self.alpha = alpha
        self.beta = beta

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        target = torch.reshape(target, (-1, self.num_channels))
        # input1 = torch.reshape(input, (-1, self.num_channels))
        input = torch.reshape(torch.permute(input, (0, 2, 3, 1)),(-1, self.num_channels))

        # labels_shape = target.shape
        # batches = labels_shape[0]
        focal_weight = torch.where(target == 1, (1 - input)**self.alpha,
                                   ((1 - target)**self.beta) *
                                   (input**self.alpha)).detach()
        # labels = torch.reshape(target, [batches, -1, 1])
        # classification = torch.reshape(input, [batches, -1, 1])

        bce_loss = F.binary_cross_entropy(
            input,
            target,
            weight=focal_weight,
            reduction='sum')
        # bce_loss = F.binary_cross_entropy(
        #     classification,
        #     labels,
        #     #   torch.tensor(1),
        #     reduction='none')
        # positive_indices = torch.where(target == 1.0)
        positive_indices = torch.nonzero(target == 1.0)
        normalizer = float(positive_indices.shape[0])
        bce_loss = bce_loss / normalizer
        return bce_loss


if __name__ == "__main__":
    a = torch.tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
    print(a.shape[0])