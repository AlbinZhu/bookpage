'''
Author: bin.zhu
Date: 2022-06-28 15:54:40
LastEditors: Albin
LastEditTime: 2022-07-14 18:10:09
Description: file content
'''

from numpy import int64, pad
import torch
import torch.nn as nn
import timm
import torch.functional as F
from cabm import CBAM

class PageNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model('resnet50d',
                                          pretrained=True,
                                          features_only=True)
        self.common_head = CommonHead()
        self.classify = PageClassifyNet()
        self.regression = PageRegressionNet()
        self.heatmap = AuxiliaryHeatmapNet()
        self.cbam = CBAM(512)
        self.activ = nn.Sigmoid()
    def forward(self, x):
        x = self.backbone(x)
        feature = x[2]
        attention = self.cbam(feature)
        common = self.common_head(attention)
        cls = self.classify(common)
        reg = self.regression(attention)
        heatmap = self.heatmap(attention)

        if self.training:
            return [cls, reg, heatmap]
        else:
            return self.__getOutput(cls, reg, heatmap)

    def __getOutput(self, cls, reg, heatmaps):
        batchsize = cls.shape[0]
        num_channels = 8
        score_threshold = 0.6
        down_scale = 8
        results = []
        for i in range(batchsize):
            labels = cls[i]
            offset = reg[i]
            heatmap = heatmaps[i]

            [feat_h, feat_w] = heatmap.shape[1:]
            reshape_heatmap = torch.reshape(heatmap, (-1, num_channels))
            left_heat = reshape_heatmap[..., :4]
            right_heat = reshape_heatmap[..., 4:]
            new_heapmap = torch.stack([left_heat, right_heat], 0)
            label_indices = torch.nonzero(labels > score_threshold)
            if label_indices.shape[0] == 0:
                return torch.ones((2, 9)) * -1

            pred_heat = new_heapmap[label_indices.squeeze()]
            pred_argmax_index = torch.argmax(pred_heat, -2)

            # offset2 = torch.reshape(offset, (offset.shape[0], -1))
            # index = offset2[pred_argmax_index]


            feat_h = int64(feat_h)
            feat_w = int64(feat_w)

            pred_argmax_index = torch.flatten(pred_argmax_index)

            y_index = pred_argmax_index // feat_w
            x_index = pred_argmax_index % feat_w

            # point_indices = torch.stack([y_index, x_index], 0)
            # point_indices2 = list(torch.split(point_indices, 1, 1))
            # point_indices2 = [x.squeeze() for x in point_indices2]
            # regression1 = torch.take(offset, point_indices)

            
            offset = torch.reshape(offset, (2, -1))
            regression = offset.index_select(1, pred_argmax_index)
            # regression = regression.view(-1, 4)

            # regression = []
            # for point in point_indices2:
            #     regression.append(offset[:, point[1], point[0]])
            # regression = torch.stack(regression, 1)
            # regression = [offset[:, x] for x in point_indices2]
            # regression2 = offset[:, point_indices2[0]]
            # a = regression[-1][-1]
            # b = offset[1, y_index[1][-1], x_index[1][-1]]
            # b1 = offset[1][point_indices[-1][-1][0], [point_indices[-1][-1][1]]]
            # c = (a == b)
            # print(a)
            # regression = offset[list[point_indices.T]]

            x = (regression[0] + x_index.int()) * down_scale
            y = (regression[1] + y_index.int()) * down_scale

            img_h, img_w = feat_h * down_scale, feat_w * down_scale
            x = x / img_w
            y = y / img_h

            x = torch.clip(x, 0, 1)
            y = torch.clip(y, 0, 1)

            x = torch.reshape(x, (-1, 4))
            y = torch.reshape(y, (-1, 4))

            label_indices = label_indices.float()
            pred_points = torch.concat([label_indices, x, y], -1)

            if label_indices.shape[0] == 1:
                pads = torch.ones((1, 9)) * -1
                pred_points = torch.concat([pred_points, pads], -2)

            results.append(pred_points)

        results = torch.stack(results, dim=0)
        return results




class CommonHead(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.stride_2_convs = nn.Conv2d(512, 192, 3, 2, (1, 1))
        self.stride_2_convs_1 = nn.Conv2d(192, 192, 3, 2, (1, 1))
        self.conv = nn.Conv2d(192, 32, 1, 1)
        self.bn1 = nn.BatchNorm2d(192)
        self.bn2 = nn.BatchNorm2d(32)
        self.activ = nn.Hardswish()

    def forward(self, x):
        output = self.stride_2_convs(x)
        output = self.bn1(output)
        output = self.stride_2_convs_1(output)
        output = self.bn1(output)
        output = self.conv(output)
        output = self.bn2(output)
        output = self.activ(output)
        return output


class PageClassifyNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(32, 32, 3, 1, (1, 1))
        self.classifier = nn.Conv2d(32, 1, 3, 1, (1, 1))
        self.bn = nn.BatchNorm2d(32)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(1024, 2)
        self.acti = nn.Hardswish()
        self.out = nn.Sigmoid()

    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        output = self.acti(output)
        output = self.classifier(output)
        output = self.flatten(output)
        output = self.dense(output)
        output = self.out(output)
        return output


class PageRegressionNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.trans_conv = nn.ConvTranspose2d(512, 32, 3, 2, 1, 1)
        self.bn = nn.BatchNorm2d(32)
        self.acti = nn.Hardswish()
        self.regression = nn.Conv2d(32, 2, 3, 1, 1)

    def forward(self, x):
        output = self.trans_conv(x)
        output = self.bn(output)
        output = self.acti(output)
        output = self.regression(output)
        return output


class AuxiliaryHeatmapNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(512, 8, 3, 2, (1, 1), 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        output = self.conv(x)
        output = self.out(output)
        return output


if __name__ == "__main__":

    input = torch.randn((1, 3, 1024, 1024))
    pagenet = PageNet()(input)

    print(pagenet)