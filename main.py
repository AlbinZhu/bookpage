from asyncio.log import logger
from matplotlib.colors import to_rgba_array
import numpy as np
import torch
from dataset import BookPage
from model import PageNet
from torch.utils.data import DataLoader
from loss import CLSSigmoid, RegLoss, HeatmapFocalLoss
from log import get_logger

train_dataset = BookPage(
    '/home/albin/Documents/data/annoDir/ImageSet/page_trainval.txt')
train_dataloader = DataLoader(train_dataset, 2, True, num_workers=1)

model = PageNet().cuda()
optimizer = torch.optim.AdamW(model.parameters(), 0.0025)

cls_loss_func = CLSSigmoid(1.0)
reg_loss_func = RegLoss(1.0)
heatmap_loss_func = HeatmapFocalLoss(8)

logger = get_logger("log/train.log")
logger.info("Starting training...")
model.train()
epoch = 50
for i in range(epoch):
    logger.info("Epoch {}".format(i + 1))
    total = []
    cls_total = []
    reg_total = []
    heatmap_total = []
    for j, (data, (cls_t, reg_t, heatmap_t)) in enumerate(train_dataloader):
        data = data.cuda()
        cls_t = cls_t.cuda()
        reg_t = reg_t.cuda()
        heatmap_t = heatmap_t.cuda()
        optimizer.zero_grad()
        [cls, reg, heatmap] = model(data.cuda())
        cls_loss = cls_loss_func(cls, cls_t)
        reg_loss = reg_loss_func(reg, reg_t)
        heatmap_loss = heatmap_loss_func(heatmap, heatmap_t)

        total_loss = cls_loss + reg_loss + heatmap_loss

        total_loss.backward()

        cls_total.append(cls_loss.item())
        reg_total.append(reg_loss.item())
        heatmap_total.append(heatmap_loss.item())
        total.append(total_loss.item())
        optimizer.step()
        # train_loss.append(total_loss)

        if (j + 1) % 100 == 0:
            logger.info(
                "Setp: {} CLS_loss: {} reg_loss: {} heatmap_loss: {} Total: {}"
                .format(j, np.mean(cls_total), np.mean(reg_total),
                        np.mean(heatmap_total), np.mean(total)))
            cls = []
            reg = []
            heatmap = []
            total = []
    # train_loss = np.mean(train_loss)

torch.save(model.state_dict(), "model.pth")
logger.info("Finish training")