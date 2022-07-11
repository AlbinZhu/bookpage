'''
Author: bin.zhu
Date: 2022-07-05 13:50:56
LastEditors: bin.zhu
LastEditTime: 2022-07-05 14:58:12
Description: file content
'''

import torch
from utils.util import load_polyGonAnnotations, get_polyGon_target
from data_generator.heatmap import PolyGonHeatmapGenerator
from torch.utils.data import Dataset
import image_utils
import cv2

page_anno_dir = '/home/albin/Documents/data/annoDir/page_label/'


class BookPage(Dataset):

    def __init__(self, anno_file):
        super(BookPage, self).__init__()
        with open(anno_file, 'r') as f:
            self.anno_files = f.read().splitlines()

        self.common_aug_method = image_utils.VisualEffect(
            image_utils.configUtils(True, True, True, True, True, True, True))
        self.preprocess = image_utils.polyGonProcessImage()
        self.heatmap_option = {
            "heat_H": 256,
            "heat_W": 256,
            "sigma": 5,
        }
        self.heatmap = PolyGonHeatmapGenerator(8, **self.heatmap_option)

    def __len__(self):
        return len(self.anno_files)

    def __getitem__(self, index):
        image_file = self.anno_files[index]
        img = cv2.imread(image_file)
        img = self.common_aug_method(img)

        anno_file = page_anno_dir + image_file.split('/')[-1].replace(
            '.jpg', '.txt')
        anno = load_polyGonAnnotations(anno_file)
        processed_image, points = self.preprocess(
            img, 1024, image_utils.ProcessType.ENCODE_ANNO, anno["points"])

        anno['points'] = points
        batch_annos, batch_labels, batch_reg = get_polyGon_target(
            anno, (self.heatmap.heat_h, self.heatmap.heat_w))
        batch_heatmap = self.heatmap.generate_heatmap_anno(anno)

        image_tensor = torch.from_numpy(processed_image)
        # image_tensor = torch.tensor(processed_image, torch.float32)
        image_tensor = torch.permute(image_tensor, (2, 0, 1))

        return image_tensor, (batch_labels, batch_reg, batch_heatmap)