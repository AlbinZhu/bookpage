import numpy as np


class BaseAnchor(object):

    def __init__(self, Net_W, Net_H, features, nums_per_feature):
        self.Net_W = Net_W
        self.Net_H = Net_H
        self.features = features
        self.Num_feat = len(self.features)
        self.nums_feature = nums_per_feature
        assert self.Num_feat == len(self.nums_feature), "the num of feature\
            should be equal to nums_per_feature!"

        self.anchors = self.generate_anchors()

    def generate_anchors(self):
        raise NotImplementedError(
            "this method should be implemented by it's sub-class")


class CenterAnchor(BaseAnchor):

    def __init__(self, anchor_mask, strides, **kwargs):
        self.strides = strides
        self.anchor_mask = anchor_mask
        super(CenterAnchor, self).__init__(**kwargs)

        assert len(self.strides) == self.Num_feat, "the num of strides\
            should be equal to nums_per_feature!"

        assert len(self.anchor_mask) == self.Num_feat, "the num of anchor_mask\
            should be equal to nums_per_feature!"

        for _, (stride, feat) in enumerate(zip(self.strides, self.features)):
            feat_W, feat_H = feat
            New_W, New_H = feat_W * stride, feat_H * stride
            assert New_H == self.Net_H
            assert New_W == self.Net_W

    def generate_anchors(self):
        all_anchors = np.zeros((0, 4), dtype=np.float32)
        for i in range(self.Num_feat):
            for j in range(self.nums_feature[i]):
                features_count = self.features[i][0] * self.features[i][1]
                anchors_tensor = np.zeros((features_count, 4),
                                          dtype=np.float32)
                grid_x, grid_y = np.meshgrid(np.arange(self.features[i][0]),
                                             np.arange(self.features[i][1]))
                grid = np.reshape(np.transpose([grid_x, grid_y], (1, 2, 0)),
                                  (-1, 2))
                anchors_tensor[:, 0:2] = grid
                anchors_tensor[:, 2:4] = self.anchor_mask[i][j]
                all_anchors = np.append(all_anchors, anchors_tensor, axis=0)
        return all_anchors


class TransformerBbox(object):

    def __init__(self, boxes, box_kind, target_kind):
        self.boxes = self.convert_to(boxes, box_kind, target_kind)

    def convert_to(self, boxes, box_kind, target_kind):
        if target_kind == "xy_xy":
            return self.convert_to_x1y1_x2y2(boxes, box_kind)
        elif target_kind == "cxy_wh":
            return self.convert_to_cxy_wh(boxes, box_kind)
        elif target_kind == "xy_wh":
            return self.convert_xy_wh(boxes, box_kind)
        else:
            raise ValueError(
                "we do not support the kind {}".format(target_kind))

    def convert_to_x1y1_x2y2(self, boxes, box_kind):
        if box_kind == "cxy_wh":
            xmin = boxes[..., 0] - boxes[..., 2] / 2
            ymin = boxes[..., 1] - boxes[..., 3] / 2
            xmax = boxes[..., 0] + boxes[..., 2] / 2
            ymax = boxes[..., 1] + boxes[..., 3] / 2
            return np.stack([xmin, ymin, xmax, ymax], axis=-1)
        elif box_kind == "xy_xy":
            return boxes
        elif box_kind == "xy_wh":
            xmax = boxes[..., 0] + boxes[..., 2]
            ymax = boxes[..., 1] + boxes[..., 3]
            return np.stack([boxes[..., 0], boxes[..., 1], xmax, ymax],
                            axis=-1)
        else:
            raise ValueError("we do not support the kind {}".format(box_kind))

    def convert_to_cxy_wh(self, boxes, box_kind):
        if box_kind == "cxy_wh":
            return boxes
        elif box_kind == "xy_xy":
            cx = (boxes[..., 2] + boxes[..., 0]) / 2
            cy = (boxes[..., 3] + boxes[..., 1]) / 2
            w = boxes[..., 2] - boxes[..., 0]
            h = boxes[..., 3] - boxes[..., 1]
            return np.stack([cx, cy, w, h], axis=-1)
        elif box_kind == "xy_wh":
            cx = boxes[..., 0] + boxes[..., 2] / 2
            cy = boxes[..., 1] + boxes[..., 3] / 2
            return np.stack([cx, cy, boxes[..., 2], boxes[..., 3]], axis=-1)
        else:
            raise ValueError("we do not support the kind {}".format(box_kind))

    def convert_xy_wh(self, boxes, box_kind):
        if box_kind == "cxy_wh":
            xmin = boxes[..., 0] + boxes[..., 2] / 2
            ymin = boxes[..., 1] + boxes[..., 3] / 2
            return np.stack([xmin, ymin, boxes[..., 2], boxes[..., 3]],
                            axis=-1)
        elif box_kind == "xy_xy":
            w = boxes[..., 2] - boxes[..., 0]
            h = boxes[..., 3] - boxes[..., 1]
            return np.stack([boxes[..., 0], boxes[..., 1], w, h], axis=-1)
        elif box_kind == "xy_wh":
            return boxes
        else:
            raise ValueError("we do not support the kind {}".format(box_kind))

    def encode_bboxes(self, labels, anchors, feat_H, feat_W):
        result_targets = np.zeros((anchors.shape[0], anchors.shape[1], 8),
                                  dtype=float)

        result_labels = np.zeros(anchors.shape[0:2], dtype=int)
        for index, (label, gt_box, select_anchor) in enumerate(
                zip(labels, self.boxes, anchors)):
            target = gt_box - select_anchor
            target[:, 2:4] = np.log(gt_box[2:4] / select_anchor[:, 2:4])
            result_targets[index, :, 0:4] = target
            result_targets[index, :,
                           4:8] = gt_box / ([feat_W, feat_H, feat_W, feat_H])
            result_labels[index] = label
        return result_targets, result_labels

    def decode_bboxes(self):
        pass

    def clip_bboxes(self, image_size=None):
        img_H, img_W = image_size
        if image_size is not None:
            xmin = np.clip(self.boxes[..., 0] * img_W, 0, img_W)
            ymin = np.clip(self.boxes[..., 1] * img_H, 0, img_H)
            xmax = np.clip(self.boxes[..., 2] * img_W, 0, img_W)
            ymax = np.clip(self.boxes[..., 3] * img_H, 0, img_H)
        else:
            xmin = np.clip(self.boxes[..., 0], 0, 1)
            ymin = np.clip(self.boxes[..., 1], 0, 1)
            xmax = np.clip(self.boxes[..., 2], 0, 1)
            ymax = np.clip(self.boxes[..., 3], 0, 1)
        return np.stack([xmin, ymin, xmax, ymax], axis=-1)


def compute_overlap(boxes, query_boxes):
    """
    Args
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float

    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)
    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + 1) *
                    (query_boxes[k, 3] - query_boxes[k, 1] + 1))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]) + 1)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]) + 1)
                if ih > 0:
                    ua = np.float64((boxes[n, 2] - boxes[n, 0] + 1) *
                                    (boxes[n, 3] - boxes[n, 1] + 1) +
                                    box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def encode_bboxes(bboxes, labels, anchors, feat_H, feat_W):
    result_targets = np.zeros((anchors.shape[0], anchors.shape[1], 8),
                              dtype=float)
    result_labels = np.zeros(anchors.shape[0:2], dtype=int)
    for index, (label, gt_box,
                select_anchor) in enumerate(zip(labels, bboxes, anchors)):
        target = gt_box - select_anchor
        target[:, 2:4] = np.log(gt_box[2:4] / select_anchor[:, 2:4])
        result_targets[index, :, 0:4] = target
        result_targets[index, :,
                       4:8] = gt_box / ([feat_W, feat_H, feat_W, feat_H])
        result_labels[index] = label
    return result_targets, result_labels


def decode_bboxes(preds, anchors):
    x = (preds[..., 0] + anchors[..., 0])
    y = (preds[..., 1] + anchors[..., 1])
    w = np.exp(preds[..., 2]) * anchors[..., 2]
    h = np.exp(preds[..., 3]) * anchors[..., 3]
    return np.stack([x, y, w, h], axis=-1)


def compute_gt_annotations(anchors,
                           annotations,
                           negative_overlap=0.4,
                           positive_overlap=0.5):
    """
    Obtain indices of gt annotations with the greatest overlap.
    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        annotations: np.array of shape (K, 5) for (x1, y1, x2, y2, label).
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).
    Returns
        positive_indices: indices of positive anchors, (N, )
        ignore_indices: indices of ignored anchors, (N, )
        argmax_overlaps_inds: ordered overlaps indices, (N, )
    """
    # (N, K)
    overlaps = compute_overlap(anchors.astype(np.float64),
                               annotations.astype(np.float64))
    # (N, )
    argmax_overlaps_inds = np.argmax(overlaps, axis=1)
    # (N, )
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

    # assign "dont care" labels
    # (N, )
    positive_indices = max_overlaps >= positive_overlap

    # adam: in case of there are gt boxes has no matched positive anchors
    # nonzero_inds = np.nonzero(overlaps == np.max(overlaps, axis=0))
    # positive_indices[nonzero_inds[0]] = 1

    # (N, )
    ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices

    return positive_indices, ignore_indices, argmax_overlaps_inds


def _compute_distance(annos_gt, anchors_feat, K_tops):
    '''
    Args:
        annos_gt : [num_gt, 4]
        anchors_feat: [num_feat, 4]
    Returns:
        center distance between annos_gt and anchors_feat, shape [num_gt, K_tops]
    '''
    N = annos_gt.shape[0]
    K = anchors_feat.shape[0]
    assert K >= K_tops, "k_tops should be less than number of anchors"
    candidate_index = np.zeros((N, K_tops), dtype=np.int)
    candidate_anchors = np.zeros((N, K_tops, 4), dtype=np.float)
    for n in range(N):
        gt_distances = np.power(annos_gt[n][0] - anchors_feat[:, 0], 2)  + \
                            np.power(annos_gt[n][1] - anchors_feat[:, 1], 2)
        sort_index = np.argsort(gt_distances)
        candidate_anchors[n] = anchors_feat[sort_index[:K_tops]]
        candidate_index[n] = sort_index[:K_tops]
    return candidate_index, candidate_anchors


def _compute_gt_threshold(overlaps_):
    '''
    Args:
        mean: [num_gt, num_anchors]
    Return:
        gt_threshold [num_gt]
    '''
    mean_overlaps = np.mean(overlaps_, axis=1)
    std_var_overlaps = np.std(overlaps_, axis=1)

    gt_threshold = mean_overlaps + std_var_overlaps
    return gt_threshold


def compute_overlap_by_one(box, query_boxes):
    """
    Args
        a: (1, 4) ndarray of float
        b: (K, 4) ndarray of float

    Returns
        overlaps: (1, K) ndarray of overlap between boxes and query_boxes
    """
    K = query_boxes.shape[0]
    overlaps = np.zeros((K), dtype=np.float64)
    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + 1) *
                    (query_boxes[k, 3] - query_boxes[k, 1] + 1))
        iw = (min(box[2], query_boxes[k, 2]) - max(box[0], query_boxes[k, 0]) +
              1)
        ih = (min(box[3], query_boxes[k, 3]) - max(box[1], query_boxes[k, 1]) +
              1)
        if iw > 0 and ih > 0:
            ua = np.float64((box[2] - box[0] + 1) * (box[3] - box[1] + 1) +
                            box_area - iw * ih)
            overlaps[k] = iw * ih / ua
    return overlaps


def anchor_targers_bbox_by_atss(anchors,
                                image_group,
                                annotations_group,
                                num_classes,
                                features,
                                tops_k=9):
    '''
    Args:
        anchors: List[...], 
        annotations_group: np.array(Batches, ...)
        features: like this [[32, 32, 3], [64, 64, 3], [128, 128, 3]], means
        feat_H, feat_W, num_anchors
    '''
    total_anchors = 0
    for feat_anchr in features:
        sum_value = 1
        for value in feat_anchr:
            sum_value *= value
        total_anchors += sum_value
    assert total_anchors == len(anchors)
    assert (len(image_group) == len(annotations_group)),\
                    "The length of the images and annotations need to be equal."
    assert (len(annotations_group) >0), \
                               "No data received to compute anchor targets for."
    for annotations in annotations_group:
        assert ('bboxes' in annotations), "Annotations should contain bboxes."
        assert ('labels' in annotations), "Annotations should contain labels."

    batch_size = len(image_group)
    regression_batch = np.zeros((batch_size, anchors.shape[0], 4 + 1),
                                dtype=np.float32)

    labels_batch = np.zeros((batch_size, anchors.shape[0], num_classes + 1),
                            dtype=np.float32)

    def _compute_gt_overlapes_(bboxes_, anchors_):

        N = bboxes_.shape[0]
        K = anchors_.shape[1]
        assert N == anchors_.shape[0]
        overlaps = np.zeros((N, K), dtype=np.float64)
        for i in range(N):
            overlaps_by_anchors = compute_overlap_by_one(
                bboxes_[i], anchors_[i])
            overlaps[i] = overlaps_by_anchors
        return overlaps

    for index, (image,
                annotations) in enumerate(zip(image_group, annotations_group)):
        if annotations["bboxes"].shape[0] <= 0:
            continue
        anno_bboxes = annotations['bboxes']
        anno_labels = annotations["labels"]
        Num_bboxes = anno_bboxes.shape[0]
        start = 0
        select_anchor_index = np.empty((Num_bboxes, 0), dtype=np.int)
        overlapes = np.empty((Num_bboxes, 0), dtype=np.int)
        select_anchor = np.empty((Num_bboxes, 0, 4), dtype=np.float)
        transform_anchors = np.empty((Num_bboxes, 0, 4), dtype=np.float)
        transform_labels = np.empty((Num_bboxes, 0), dtype=np.float)
        for feat_W, feat_H, num_anchor in features:
            boxes = anno_bboxes * [feat_W, feat_H, feat_W, feat_H]
            total_nums = feat_H * feat_W * num_anchor
            end = start + total_nums
            anchor = anchors[start:end]
            select_index, dis_anchors = _compute_distance(
                boxes, anchor, tops_k)
            select_index += start
            dis_anchors[:, :, 2:4] = dis_anchors[:, :, 2:4] * [feat_H, feat_W]
            # transform gt_target to cented delta
            select_trans_anchors, select_trans_labels = encode_bboxes(
                boxes, anno_labels, dis_anchors)
            # compute overlaps with gt_bboxes and selected_distance_anchors
            xy_bboxes = TransformerBbox(boxes, [feat_H, feat_W])
            xy_anchors = TransformerBbox(dis_anchors, [feat_H, feat_W])
            select_overlapes = _compute_gt_overlapes_(xy_bboxes, xy_anchors)
            # accumate to total
            select_anchor_index = np.concatenate(
                [select_anchor_index, select_index], axis=1)
            select_anchor = np.concatenate([select_anchor, dis_anchors],
                                           axis=1)
            overlapes = np.concatenate([overlapes, select_overlapes], axis=1)
            # trans_gt_target to concatenate
            transform_anchors = np.concatenate(
                [transform_anchors, select_trans_anchors], axis=1)
            transform_labels = np.concatenate(
                [transform_labels, select_trans_labels], axis=1)
            #update start_index
            start = end
        gt_threshold = _compute_gt_threshold(overlapes)
        pos_indices = overlapes >= gt_threshold.reshape((-1, 1))
        positive_index = select_anchor_index[pos_indices]
        regression_batch[index, positive_index, -1] = 1
        labels_batch[index, positive_index, -1] = 1

        regression_batch[index, positive_index,
                         0:-1] = transform_anchors[pos_indices]
        labels_batch[index, positive_index,
                     transform_labels[pos_indices].reshape(-1).astype(int)] = 1

    return labels_batch, regression_batch


def anchor_targets_bbox_by_yolo(anchors,
                                Net_W,
                                Net_H,
                                image_group,
                                annotations_group,
                                num_classes,
                                features,
                                iou_threshold,
                                neg_thershlod=0.35):  #0.7
    """
    Generate anchor targets for bbox detection.

    Returns
        labels_batch: batch that contains labels & anchor states (np.array of shape (batch_size, N, num_classes + 1),
                        where N is the number of anchors for an image and the last column defines the anchor state
                        (-1 for ignore, 0 for bg, 1 for fg).
        regression_batch: batch that contains bounding-box regression targets for an image & anchor states
                        (np.array of shape (batch_size, N, 6 + 1), where N is the number of anchors for an image,
                        the first 8 columns define regression targets for (encode_box, and origin_box)
                        (encode_x, decode_y, encode_w, encode_h, x, y, w, h) and the last column defines
                        anchor states (-1 for ignore, 0 for bg, 1 for fg).
    """

    def _compute_overlaps_(gt_bboxes, select_anchors):
        '''
        gt_bboxes: [num, 4]
        select_anchors: [num, num_anchor, 4]
        return: [num, num_anchor]
        '''
        N = gt_bboxes.shape[0]
        K = select_anchors.shape[1]
        result = np.zeros((N, K), dtype=float)
        for i in range(N):
            result[i] = compute_overlap_by_one(gt_bboxes[i], select_anchors[i])
        return result

    assert (len(image_group) == len(annotations_group)),\
                    "The length of the images and annotations need to be equal."
    assert (len(annotations_group) >0), \
                               "No data received to compute anchor targets for."
    for annotations in annotations_group:
        assert ('bboxes' in annotations), "Annotations should contain bboxes."
        assert ('labels' in annotations), "Annotations should contain labels."
    batch_size = len(image_group)

    regression_batch = np.zeros((batch_size, anchors.shape[0], 4 + 4 + 1),
                                dtype=np.float32)
    labels_batch = np.zeros((batch_size, anchors.shape[0], num_classes + 1),
                            dtype=np.float32)

    # compute labels and regression targets
    for index, (image,
                annotations) in enumerate(zip(image_group, annotations_group)):
        if annotations['bboxes'].shape[0] == 0:
            continue
        anno_bboxes = annotations["bboxes"]
        anno_labels = annotations["labels"]
        Num_bboxes = anno_bboxes.shape[0]
        truth_bboxes = anno_bboxes.copy()
        truth_bboxes[..., 0:2] = 0
        convrt_gt_bboxes = TransformerBbox(truth_bboxes, "cxy_wh",
                                           "xy_xy").boxes
        # (Num_bboxes, total_anchors_by_single_bbox)
        related_bboxes_indices = np.empty((Num_bboxes, 0), dtype=int)
        trans_anchors = np.empty((Num_bboxes, 0, 8), dtype=int)
        trans_labels = np.empty((Num_bboxes, 0), dtype=int)
        total_anchors_by_single_bbox = 0
        start = 0

        # calculate ignore anchors
        total_ignore_indices = np.array([], int)
        for feat_W, feat_H, num_anchor in features:
            bboxes = anno_bboxes * [feat_W, feat_H, feat_W, feat_H]
            int_bboxes = np.floor(bboxes)
            # print("selected int_bboxes: ", int_bboxes, ", trans_bboxes: ",
            #       trans_bboxes)
            feat_num_anchor = feat_H * feat_W * num_anchor
            end = start + feat_num_anchor

            real_gt_bboxes = convrt_gt_bboxes * [feat_W, feat_H, Net_W, Net_H]
            ct_feat_anchors = anchors[start:end]
            feat_anchors = TransformerBbox(ct_feat_anchors, "cxy_wh",
                                           "xy_xy").boxes

            _, ignore_indices, _ = compute_gt_annotations(
                feat_anchors, real_gt_bboxes, neg_thershlod)
            ignore_indices = np.where(ignore_indices == True)[0] + start

            total_ignore_indices = np.append(total_ignore_indices,
                                             ignore_indices)

            select_index = np.matmul(int_bboxes[:, 0:2], [1, feat_W]) + start
            # print("selected select_index: ", select_index)
            select_indices = np.array([[(j + i * feat_H * feat_W)
                                        for i in range(num_anchor)]
                                       for j in select_index])
            # print("selected anchor shape: ", num_anchor, ", feat_H: ", feat_H,
            #       ", feat_W: ", feat_W)
            selected_anchors = anchors[select_indices.reshape(-1).astype(
                int)].reshape(-1, num_anchor, 4)
            # print("selected anchor: ", selected_anchors)
            trans_candiate_anchors, trans_candiate_labels =\
                    TransformerBbox(bboxes, "cxy_wh", "cxy_wh").encode_bboxes(
                                  anno_labels, selected_anchors, feat_H, feat_W)

            trans_anchors = np.concatenate(
                [trans_anchors, trans_candiate_anchors], axis=1)
            trans_labels = np.concatenate(
                [trans_labels, trans_candiate_labels], axis=1)
            related_bboxes_indices = np.concatenate(
                [related_bboxes_indices, select_indices], axis=1)
            total_anchors_by_single_bbox += num_anchor
            start = end

        # compute overlaps (Num_boxes, total_anchors_by_single_bbox, 4)
        select_anchors = anchors[related_bboxes_indices.reshape(-1).astype(int)\
                                  ].reshape(-1, total_anchors_by_single_bbox, 4)

        select_anchors[...,
                       2:4] = select_anchors[..., 2:4] / ([feat_W, feat_H])
        select_anchors[..., 0:2] = 0
        convert_anchors = TransformerBbox(select_anchors, "cxy_wh",
                                          "xy_xy").boxes
        # print("select_anchors: ", select_anchors.shape, ", convert_anchors: ",
        #       convert_anchors.shape)
        # (Num_boxes, total_anchors_by_single_bbox)
        overlaps = _compute_overlaps_(convrt_gt_bboxes, convert_anchors)
        # (Num_boxes, )
        argmax_indices = np.argmax(overlaps, axis=1)
        max_iou_indices = related_bboxes_indices[np.arange(Num_bboxes),
                                                 argmax_indices]

        # add more positive (Num_boxes, total_anchors_by_single_bbox)
        additional_positive_indices = overlaps >= iou_threshold
        for i in range(Num_bboxes):
            max_index = max_iou_indices[i].astype(int)
            additional_indices = related_bboxes_indices[
                i, additional_positive_indices[i]]
            if max_index not in additional_indices:
                additional_positive_indices[i][argmax_indices[i]] = True

        transform_anchors = trans_anchors[additional_positive_indices].reshape(
            -1, 8)
        transform_labels = trans_labels[additional_positive_indices].reshape(
            -1)

        select_anchor_index = related_bboxes_indices[
            additional_positive_indices].astype(int)
        regression_batch[index, select_anchor_index, -1] = 1
        regression_batch[index, select_anchor_index, 0:8] = transform_anchors

        print("total_ignore_indices: ", total_ignore_indices.shape)

        labels_batch[index, total_ignore_indices, -1] = -1
        labels_batch[index, select_anchor_index, -1] = 1
        labels_batch[index, select_anchor_index,
                     transform_labels.astype(int)] = 1

    return labels_batch, regression_batch


import unittest


class TestCase(unittest.TestCase):

    def test_transfor_bboxes(self):
        box_1 = np.random.rand(4)
        print("origin_box: ", box_1)
        trans_box = TransformerBbox(box_1, "cxy_wh", "xy_xy").clip_bboxes(
            (640, 640))
        print("trans_box: ", trans_box)

        boxes_2 = np.random.rand(6, 4)
        trans_box = TransformerBbox(boxes_2, "cxy_wh", "xy_xy").boxes
        print("trans_box: ", trans_box)

    def test_decode_bboxes(self):
        anchor_mask = [[[76, 85], [61, 73]]]
        image_group = np.random.rand(6, 16, 16, 3)
        num_classes = 5
        features = np.array([[16, 16, 2]])
        common_args = {
            "Net_W": 640,
            "Net_H": 640,
            "features": features[:, 0:2],
            "nums_per_feature": [2]
        }
        all_anchors = CenterAnchor(anchor_mask, [40], **common_args)

        preds = np.random.rand(1, 512, 4)

        preds = decode_bboxes(preds, all_anchors.anchors)

        print(preds.shape)

    def test_anchors(self):
        np.set_printoptions(threshold=np.inf)
        feature_shapes = [[64, 64], [32, 32], [16, 16]]
        anchor_mask = [[[0.1, 0.1], [0.2, 0.3], [0.3, 0.21]], [[0.31, 0.12], \
                            [0.14, 0.1],[0.1, 0.1]], [[0.231, 0.315], [0.61, 0.132]]]
        common_args = {
            "Net_W": 640,
            "Net_H": 640,
            "features": feature_shapes,
            "nums_per_feature": [3, 3, 2]
        }
        all_anchors = CenterAnchor(anchor_mask, [10, 20, 40], **common_args)
        #print(all_anchors.anchors)
        for anchor_tensors in all_anchors.anchors:
            print(anchor_tensors.shape)
            #print(anchor_tensors)

    def test_pos_neg_bbox_atss(self):
        np.set_printoptions(threshold=np.inf)
        anchor_mask = [[[0.1, 0.1], [0.2, 0.3], [0.3, 0.21]],
                       [[0.31, 0.12], [0.14, 0.1], [0.1, 0.1]],
                       [[0.231, 0.315], [0.61, 0.132]]]
        image_group = np.random.rand(6, 16, 16, 3)
        num_classes = 5
        features = np.array([[64, 64, len(anchor_mask[0])], [32, 32, len(anchor_mask[1])], \
                                                      [16, 16, len(anchor_mask[2])]])
        common_args = {
            "Net_W": 640,
            "Net_H": 640,
            "features": features[:, 0:2],
            "nums_per_feature": [3, 3, 2]
        }
        all_anchors = CenterAnchor(anchor_mask, [10, 20, 40], **common_args)
        annotations_group = []
        for i in range(6):
            num_boxes = np.random.randint(0, 4)
            annotations = {
                'labels': np.empty((0, ), dtype=np.int32),
                'bboxes': np.empty((0, 4), dtype=np.float32)
            }
            boxes = np.random.rand(num_boxes, 4)
            labels = np.random.randint(0, num_classes, size=num_boxes)
            annotations['bboxes'] = np.concatenate(
                [annotations['bboxes'], boxes])
            annotations['labels'] = np.concatenate(
                [annotations['labels'], labels])
            annotations_group.append(annotations)

        # regresion_batch, labels_batch = anchor_targers_bbox_by_atss(
        #     all_anchors.anchors, image_group, annotations_group, num_classes,
        #     features)

        regresion_batch, labels_batch = anchor_targets_bbox_by_yolo(
            all_anchors.anchors, all_anchors.Net_W, all_anchors.Net_H,
            image_group, annotations_group, num_classes, features, 0.5)
        # print("==========regresion_batch=========")
        # print(regresion_batch)
        # print("==========labels_batch============")
        # print(labels_batch)


if __name__ == "__main__":
    unittest.main()