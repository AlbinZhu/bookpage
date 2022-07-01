import numpy as np
# iou only for center_bbox


def overlap(a, b, bbox_kind="center"):
    if bbox_kind == "center":
        l1 = a[..., 0] - a[..., 1] / 2
        l2 = b[..., 0] - b[..., 1] / 2
        left = np.where(l1 > l2, l1, l2)
        r1 = a[..., 0] + a[..., 1] / 2
        r2 = b[..., 0] + b[..., 1] / 2
        right = np.where(r1 < r2, r1, r2)
        # print("overlape left: ", left, ",  right: ", right, ", right - left: ",
        #       right - left)
        return right - left
    elif bbox_kind == "left_right":
        left = np.where(a[..., 0] > b[..., 0], a[..., 0], b[..., 0])
        right = np.where(a[..., 1] < b[..., 1], a[..., 1], b[..., 1])
        return right - left
    else:
        raise ValueError("we donot support other bbox represent type")


def box_intersection(a, b, bbox_kind="center"):
    w = overlap(a[..., 0::2], b[..., 0::2], bbox_kind)
    h = overlap(a[..., 1::2], b[..., 1::2], bbox_kind)
    area = w * h
    area = np.where(h <= 0, 0, area)
    area = np.where(w <= 0, 0, area)
    return area


def box_union(a, b, bbox_kind="center"):
    i = box_intersection(a, b, bbox_kind)
    u = a[..., 2] * a[..., 3] + b[..., 2] * b[..., 3] - i
    return u


def box_encompise(a, b):
    top = np.minimum(a[..., 1] - a[..., 3] / 2, b[..., 1] - b[..., 3] / 2)
    bottom = np.maximum(a[..., 1] + a[..., 3] / 2, b[..., 1] + b[..., 3] / 2)
    left = np.minimum(a[..., 0] - a[..., 2] / 2, b[..., 0] - b[..., 2] / 2)
    right = np.maximum(a[..., 0] + a[..., 2] / 2, b[..., 0] + b[..., 2] / 2)
    return np.stack([left, right, top, bottom], axis=-1)


def compute_iou(box_1, box_2):
    I = box_intersection(box_1, box_2)
    U = box_union(box_1, box_2)
    iou = np.where(I == 0, 0, I)
    iou = np.where(U == 0, 0, iou / U)
    return iou


def compute_bbox_coverage(box_1, box_2, box_kind="center"):
    I = box_intersection(box_1, box_2, box_kind)
    if box_kind == "center":
        box_1_area = box_1[..., 2] * box_1[..., 3]
    elif box_kind == "left_right":
        box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] -
                                                        box_1[..., 1])
    iou = np.where(I == 0, 0, I)
    iou = np.where(box_1_area == 0, 0, iou / (box_1_area + 0.0000000001))
    return iou


class IouJob(object):

    def __init__(self, truths_box, preds_box, iou_kind):
        self.truths = truths_box
        self.pred = preds_box
        self.kind = iou_kind

    def __call__(self):
        if self.kind == "giou":
            return self.compute_giou()
        elif self.kind == "diou":
            return self.compute_diou()
        elif self.kind == "ciou":
            return self.compute_ciou()
        else:
            raise ValueError("we do not support iou_kind: {}".format(
                self.kind))

    def compute_giou(self):
        iou = compute_iou(self.pred, self.truths)
        encompose_c = box_encompise(self.pred, self.truths)
        w = encompose_c[..., 1] - encompose_c[..., 0]
        h = encompose_c[..., 3] - encompose_c[..., 2]
        c = w * h
        u = box_union(self.pred, self.truths)
        giou = np.where(c == 0, iou, iou - (c - u) / c)
        return giou

    def compute_ciou(self):
        encompose_c = box_encompise(self.pred, self.truths)
        w = encompose_c[..., 1] - encompose_c[..., 0]
        h = encompose_c[..., 3] - encompose_c[..., 2]
        c = w * w + h * h
        d = np.power(self.pred[..., 0] - self.truths[..., 0], 2) + \
                                np.power(self.pred[..., 0] - self.truths[..., 0], 2)
        iou = compute_iou(self.pred, self.truths)
        ar_gt = self.truths[..., 2] / self.truths[..., 3]
        ar_pred = self.pred[..., 2] / self.pred[..., 3]
        ar_loss = 4 / (np.pi * np.pi) * np.power(
            np.arctan(ar_gt) - np.arctan(ar_pred), 2)
        alpha = ar_loss / (1 - iou + ar_loss + 0.000001)
        ciou = np.where(c == 0, iou, iou - (d / c + alpha * ar_loss))
        return ciou

    def compute_diou(self):
        encompose_c = box_encompise(self.pred, self.truths)
        w = encompose_c[..., 1] - encompose_c[..., 0]
        h = encompose_c[..., 3] - encompose_c[..., 2]
        c = w * w + h * h
        iou = compute_iou(self.pred, self.truths)
        d = np.power(self.pred[..., 0] - self.truths[..., 0], 2) + \
                                np.power(self.pred[..., 0] - self.truths[..., 0], 2)
        diou = np.where(c == 0, iou, iou - np.power(d / c, 0.6))
        return diou


import unittest


class TestStringMethods(unittest.TestCase):

    def test_iou(self):
        truth = np.array([0.703125, 0.66049383, 0.22878086, 0.41769547])
        pred = np.array([0.19463735, 0.31044239, 0.00810185, 0.01286008])
        IOU = IouJob(truth, pred, "giou")
        iou = IOU._compute_iou()
        print(iou)
        self.assertAlmostEqual(iou, 0.0, 1)


if __name__ == "__main__":
    unittest.main()
