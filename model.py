'''
Author: bin.zhu
Date: 2022-06-28 15:54:40
LastEditors: bin.zhu
LastEditTime: 2022-07-06 13:57:03
Description: file content
'''

from numpy import pad
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

    def forward(self, x):
        x = self.backbone(x)
        feature = x[2]
        attention = self.cbam(feature)
        common = self.common_head(attention)
        cls = self.classify(common)
        reg = self.regression(attention)
        heatmap = self.heatmap(attention)

        return [cls, reg, heatmap]


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

    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        output = self.acti(output)
        output = self.classifier(output)
        output = self.flatten(output)
        output = self.dense(output)
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

    def forward(self, x):
        output = self.conv(x)
        return output


if __name__ == "__main__":

    input = torch.randn((1, 3, 1024, 1024))
    pagenet = PageNet()(input)

    print(pagenet)