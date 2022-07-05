from asyncio.log import logger
import numpy as np
import torch
from model import PageNet
from data import DataPipe
from torch.utils.data import DataLoader
from loss import CLSSigmoid, RegLoss, HeatmapFocalLoss
from log import get_logger

train_datapipe = DataPipe(
    '/home/albin/Documents/data/annoDir/ImageSet/page_trainval.txt')
train_dataloader = DataLoader(train_datapipe, 4, True)

model = PageNet().cuda()
optimizer = torch.optim.AdamW(model.parameters, 0.0025)

cls_loss_func = CLSSigmoid(1.0)
reg_loss_func = RegLoss(1.0)
heatmap_loss_func = HeatmapFocalLoss(8)

logger = get_logger("log/train.log")
logger.info("Starting training...")
model.train()
epoch = 50
for i in range(epoch):
    logger.info("Epoch {}".format(i + 1))
    train_loss = []
    for data, (cls_t, reg_t, heatmap_t) in train_dataloader:
        optimizer.zero_grad()
        [cls, reg, heatmap] = model(data.cuda())
        cls_loss = cls_loss_func(cls, cls_t)
        reg_loss = reg_loss_func(reg, reg_t)
        heatmap_loss = heatmap_loss_func(heatmap, heatmap_t)

        total_loss = cls_loss + reg_loss + heatmap_loss

        total_loss.backward()
        optimizer.step()
        train_loss.append(total_loss)
    if (i + 1) % 100 == 0:
        logger.info(
            "Setp: {} CLS_loss: {} reg_loss: {} heatmap_loss: {} Total: {}".
            format(cls_loss, reg_loss, heatmap_loss, total_loss))
    train_loss = np.mean(train_loss)

torch.save(model.state_dict(), "model.pth")
logger.info("Finish training")