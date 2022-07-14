'''
Author: Albin
Date: 2022-07-13 14:21:51
LastEditors: Albin
LastEditTime: 2022-07-14 18:47:24
FilePath: /bookpage/eval.py
'''
import torch
from model import PageNet
import cv2
import numpy as np
import argparse

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse_args():
    parser = argparse.ArgumentParser(description="eval")
    parser.add_argument('--load-from', type=str, default='./model.pth', required=True)
    parser.add_argument('--image', type=str, required=True)

    return parser.parse_args()
    

def inference(input: torch.Tensor, model_path:str):
    model: torch.nn.Module = PageNet().cuda()

    model.load_state_dict(torch.load(model_path))
    model.eval()

    
    pred = model(input.cuda())
    return pred

def show(img:np.ndarray, pred:np.ndarray):
    h, w = img.shape[:2]
    for i in range(pred.shape[1]):
        if i == 0:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        x = pred[0][i][1:5] * w
        y = pred[0][i][5:] * h
        points = zip(x,y)
        for point in points:
            cv2.circle(img, (int(point[0]), int(point[1])), 3, color, 2)
            print(point) 
    
    cv2.imshow('img', img)
    cv2.waitKey(0)


def img_process(image_file:str):
    img = cv2.imread(image_file)
    [img_h, img_w] = img.shape[:2]

    scale = img_w / 1024
    scale_h = int(img_h / scale)
    resize_img = cv2.resize(img, (1024, scale_h))
    # cv2.imshow('img', resize_img)
    # cv2.waitKey(0)
    resize_img = resize_img.astype(np.float32) / 255.0
    # resize_img /= 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    resize_img -= mean
    resize_img /= std
    pad = int((1024 - scale_h) / 2)
    pad_img = cv2.copyMakeBorder(resize_img, pad, 1024 - scale_h - pad, 0, 0, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    tensor = torch.from_numpy(pad_img)
    tensor = torch.permute(tensor, (2, 0, 1))
    

    return pad_img, tensor.unsqueeze(0)

if __name__ == '__main__':
    args = parse_args()
    img, input = img_process(args.image)
    # img, input = img_process("./1646637019196.jpg")
    res = inference(input, args.load_from)
    show(img, res)
    
    # image = img_process(image)
    # print(image.shape)
    # inference(image)

