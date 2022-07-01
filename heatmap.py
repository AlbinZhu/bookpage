'''
Author: bin.zhu
Date: 2022-06-30 15:12:41
LastEditors: bin.zhu
LastEditTime: 2022-06-30 15:12:43
Description: file content
'''
import numpy as np
import cv2
import math


class BaseHeatmapGenerator():

    def __init__(
        self,
        heat_H,
        heat_W,
        sigma,
    ) -> None:
        self.heat_h = heat_H
        self.heat_w = heat_W
        self.sigma = sigma

    def gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def gaussian_radius(self, det_size, min_overlap=0.7):
        height, width = det_size

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return min(r1, r2, r3)


class PolyGonHeatmapGenerator(BaseHeatmapGenerator):

    def __init__(self, num_joints, **kwargs) -> None:
        self.num_joints = num_joints
        super(PolyGonHeatmapGenerator, self).__init__(**kwargs)
        #sigma = 8 remember

    def generate_heatmap(self, heatmap, plane_idx, center, sigma):
        _, height, width = heatmap.shape
        tmp_size = sigma * 3  # 这里控制高斯核大小，可改为你想要的高斯核大小
        mu_x, mu_y = center
        ul = [int(mu_x - tmp_size),
              int(mu_y - tmp_size)]  # 关键点高斯分布左上角坐标 up_left
        br = [int(mu_x + tmp_size + 1),
              int(mu_y + tmp_size + 1)]  # 关键点高斯分布右下角坐标 bottom_right
        size = 2 * tmp_size + 1  # 高斯核大小
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2  # 6
        g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
        g_x = max(0, -ul[0]), min(
            br[0], width) - ul[0]  # (0,13) 差值为2 * tmp_size + 1，即size大小
        g_y = max(0, -ul[1]), min(br[1], height) - ul[1]
        img_x = max(0, ul[0]), min(br[0], width)
        img_y = max(0, ul[1]), min(br[1], height)
        # print(img_x, img_y, g_x, g_y, "center: ", center, "computed center: ",
        #       (img_x[0] + img_x[1]) / 2, (img_y[0] + img_y[1]) / 2)
        temp_heatmap = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        am = np.amax(temp_heatmap)
        heatmap[plane_idx, img_y[0]:img_y[1],
                img_x[0]:img_x[1]] = temp_heatmap / am
        return heatmap

    def get_heatmap(self, annos, height, width, num_joints):
        joints_heatmap = np.zeros((num_joints, height, width),
                                  dtype=np.float32)
        for i, points in enumerate(annos):
            if points[0] < 0 or points[1] < 0:
                continue
            joints_heatmap = self.generate_heatmap(joints_heatmap, i, points,
                                                   self.sigma)
        # transpose to match model output (H, W, C)
        joints_heatmap = np.transpose(joints_heatmap, (1, 2, 0))
        return joints_heatmap

    def generate_batch(self, batch_annos):
        batches = len(batch_annos)
        target_batch_annos = np.zeros(
            (batches, self.heat_h, self.heat_w, self.num_joints), dtype=float)
        for index, anno in enumerate(batch_annos):
            for _, (points,
                    label) in enumerate(zip(anno['points'], anno['labels'])):
                if label == 0:
                    start_index = 0
                    end_index = 4
                elif label == 1:
                    start_index = 4
                    end_index = 8
                polygon_points = points.reshape(
                    (-1, 2)) * (self.heat_w, self.heat_h)
                num = int(self.num_joints / 2)

                heat_map = self.get_heatmap(polygon_points, self.heat_h,
                                            self.heat_w, num)
                target_batch_annos[index, :, :,
                                   start_index:end_index] = heat_map
                #print(target_batch_annos[index])
        return target_batch_annos


class DetHeatmapGenerator(BaseHeatmapGenerator):

    def __init__(self, num_classes, max_objects, **kwargs) -> None:
        self.num_classes = num_classes
        self.max_objs = max_objects
        super(DetHeatmapGenerator, self).__init__(**kwargs)

    def draw_umich_gaussian(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom,
                                   radius - left:radius + right]

        # TODO debug
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def generate_batch(self, batch_annos, anchor=(9, 5)):
        batches = len(batch_annos)
        # 4 (x, y, w, h), 2(indices, (b, ind)), +1 mask (mark valid box)
        boxes_batch = np.zeros((batches, self.max_objs, 4 + 2 + 1),
                               dtype=np.float)
        labels_batch = np.zeros(
            (batches, self.heat_h, self.heat_w, self.num_classes), dtype=float)
        for index, anno in enumerate(batch_annos):
            labels = anno['labels']
            gt_boxes = anno['bboxes']
            for i, (label, box) in enumerate(zip(labels, gt_boxes)):
                cx, cy, w, h = box * [
                    self.heat_w, self.heat_h, self.heat_w, self.heat_h
                ]
                radius = self.gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                points = np.array([cx, cy], np.float)
                points_int = points.astype(np.int)
                self.draw_umich_gaussian(labels_batch[index, :, :, label],
                                         points_int, radius)
                reg_diff_x, reg_diff_y = points - points_int

                boxes_batch[index, i, 0] = reg_diff_x
                boxes_batch[index, i, 1] = reg_diff_y
                boxes_batch[index, i, 2] = math.log(w / anchor[0])
                boxes_batch[index, i, 3] = math.log(h / anchor[1])
                # print("origin: ", box, ", convert: ", w / self.heat_w, " ",
                #       h / self.heat_h, ", points: ", points, ", points_int: ",
                #       points_int, ", heatmap reg: ", reg_diff_x, " ",
                #       reg_diff_y, " ", w, " ", h)
                boxes_batch[index, i, 4] = index
                boxes_batch[index, i,
                            5] = points_int[0] + points_int[1] * self.heat_w
                boxes_batch[index, i, 6] = 1
        return labels_batch, boxes_batch


import unittest, os, sys

sys.path.append("./")
from utils import util


class TestHeapMap(unittest.TestCase):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    np.set_printoptions(threshold=np.inf)

    def test_heatmap(self):
        image_path = "/scanImages/new_images/1642572257213.jpg"
        anno_path = "/home/baseuser/workspace/annoDir/page_label/1642572257213.txt"
        anno = util.load_polyGonAnnotations(anno_path)
        src_image = cv2.imread(image_path)
        print("src_anno: ", anno['points'])
        annos = (anno['points'].reshape((-1, 2)) * (128, 128)).astype(np.int32)
        heatmap_option = {
            "heat_H": 128,
            "heat_W": 128,
            "sigma": 1,
        }
        print("annos: ", annos)
        polygon_heatmap = PolyGonHeatmapGenerator(8, **heatmap_option)
        heat_map = polygon_heatmap.get_heatmap(annos, heatmap_option["heat_H"],
                                               heatmap_option["heat_W"], 8)
        saved_root = "/home/baseuser/workspace/verifyDataset/heatmap/"
        print("heat_map shape: ", heat_map.shape)
        for i in range(8):
            heat = heat_map[:, :, i]
            print("heat shape: ", heat.shape)
            indices = np.where(heat == 1.0)
            print("heat indices: ", indices)
            reshape_heat = heat.reshape((-1, ))
            reshape_indices = np.where(reshape_heat == 1.0)
            print("reshape_indices: ", reshape_indices)

            y = (reshape_indices[0] / 128).astype(np.int32)
            x = (reshape_indices[0] % 128)
            print("x: ", x, "y: ", y)

            heat = (heat * 255).astype(int)
            save_file = saved_root + "heat_{}.jpg".format(i)
            cv2.imwrite(save_file, heat)


if __name__ == "__main__":
    unittest.main()
