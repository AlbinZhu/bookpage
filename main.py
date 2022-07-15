'''
Author: Albin
Date: 2022-06-29 17:54:44
LastEditors: Albin
LastEditTime: 2022-07-15 17:39:06
FilePath: /bookpage/main.py
'''
from asyncio.log import logger
import numpy as np
import torch
from dataset import BookPage
from model import PageNet
from torch.utils.data import DataLoader
from loss import CLSSigmoid, RegLoss, HeatmapFocalLoss
from log import get_logger
import argparse
from pathlib import Path
import torch.distributed as dist
import os
from torch.utils.data.distributed import DistributedSampler

parser = argparse.ArgumentParser(description='training')
parser.add_argument('--train_file', type=str, required=True)
parser.add_argument('--page_anno_dir', type=str, required=True)
parser.add_argument('--bs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()

local_rank = int(os.environ["LOCAL_RANK"])

dist.init_process_group("nccl", init_method='env://')
train_dataset = BookPage(args.train_file, args.page_anno_dir)
train_samper = DistributedSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, args.bs, num_workers=args.num_workers, sampler=train_samper)

model = PageNet().cuda()
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
optimizer = torch.optim.AdamW(model.parameters(), args.lr)

# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 2)
iters = len(train_dataloader)
cls_loss_func = CLSSigmoid(1.0)
reg_loss_func = RegLoss(1.0)
heatmap_loss_func = HeatmapFocalLoss(8)

ckpt_dir = Path('ckpt')
if not ckpt_dir.exists():
    ckpt_dir.mkdir()

logger = get_logger("log/train.log")
logger.info("Starting training...")
model.train()
for i in range(args.epochs):
    logger.info("Epoch {} / {}".format(i + 1, args.epochs))
    total = []
    cls_total = []
    reg_total = []
    heatmap_total = []
    for j, (data, (cls_t, reg_t, heatmap_t)) in enumerate(train_dataloader):
        data = data.cuda()
        cls_t = cls_t.cuda()
        reg_t = reg_t.cuda()
        heatmap_t = heatmap_t.cuda()

        data.to(local_rank, non_blocking=True)
        cls_t.to(local_rank, non_blocking=True)
        reg_t.to(local_rank, non_blocking=True)
        heatmap_t.to(local_rank, non_blocking=True)

        optimizer.zero_grad()
        cls, reg, heatmap = model(data)
        cls_loss = cls_loss_func(cls, cls_t)
        reg_loss = reg_loss_func(reg, reg_t)
        heatmap_loss = heatmap_loss_func(heatmap, heatmap_t)

        total_loss = cls_loss + 2 * reg_loss + 10 * heatmap_loss

        total_loss.backward()

        cls_total.append(cls_loss.item())
        reg_total.append(reg_loss.item())
        heatmap_total.append(heatmap_loss.item())
        total.append((cls_loss + reg_loss + heatmap_loss).item())
        optimizer.step()
        # lr_scheduler.step(i + j / iters)
        # train_loss.append(total_loss)

        if (j + 1) % 100 == 0:
            logger.info(
                "Setp: {} / {} CLS_loss: {} reg_loss: {} heatmap_loss: {} Total: {} lr: {}"
                .format(j, iters, np.mean(cls_total), np.mean(reg_total),
                        np.mean(heatmap_total), np.mean(total), optimizer.state_dict()['param_groups'][0]['lr']))
            cls_total = []
            reg_total = []
            heatmap_total = []
            total = []
    # train_loss = np.mean(train_loss)

    if (i + 1) % 10 == 0:
        logger.info("Saving model for epoch {}".format(i + 1))
        torch.save(model.state_dict(), "{}/model-epoch{}.pth".format(ckpt_dir.name, i + 1))


torch.save(model.state_dict(), "{}/fian-model.pth".format(ckpt_dir.name))
logger.info("Finish training")