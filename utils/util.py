# import tensorflow as tf
import numpy as np
import math, cv2
import os


def load_polyGonAnnotations(anno_path):
    anno_file = open(anno_path, "r")
    annotations = {
        'labels': np.empty((0, ), dtype=np.int32),
        'points': np.empty((0, 8), dtype=np.float32),
    }
    #print(anno_file)
    for line in anno_file.readlines():
        cur_line = line.strip("\n").split(" ")
        cur_line = [float(x) for x in cur_line]
        anno_id = int(cur_line[0])
        points = np.array(cur_line[1:]).astype(np.float32)
        annotations['points'] = np.concatenate(
            [annotations['points'], [points]])
        annotations['labels'] = np.concatenate(
            [annotations['labels'], [anno_id]])
    anno_file.close()
    return annotations


def load_OcrNumAnnotations(anno_path):

    anno_file = open(anno_path, "r")
    annotations = {
        'labels': np.empty((0, ), dtype=np.int32),
    }
    cur_line = anno_file.readlines()[0].strip("\n").split(" ")
    cur_line = [float(x) for x in cur_line]
    anno_id = np.array(cur_line).astype(np.int32)
    annotations['labels'] = anno_id
    anno_file.close()
    #print("anno path: ", anno_path, ", anno: ", annotations)
    return annotations


def load_BoxAnnotations(anno_path):
    anno_file = open(anno_path, "r")
    annotations = {
        'labels': np.empty((0, ), dtype=np.int32),
        'bboxes': np.empty((0, 4), dtype=np.float32),
    }
    for line in anno_file.readlines():
        cur_line = line.strip("\n").split(" ")
        cur_line = [float(x) for x in cur_line]
        anno_id = int(cur_line[0])
        bbox = np.array(cur_line[1:])
        annotations['bboxes'] = np.concatenate([annotations['bboxes'], [bbox]])
        annotations['labels'] = np.concatenate(
            [annotations['labels'], [anno_id]])
    anno_file.close()
    return annotations


def get_polyGon_targets(annotations, batch_size, input_shape):
    regression_batch = np.zeros((batch_size, 9 * 2), dtype=np.float32)
    input_H, input_W = input_shape
    polyGon_labels_batch = np.zeros((batch_size, 2), dtype=np.float32)
    # 2 (x, y), 2(indices, (b, ind)), 1 mask (mark valid box)
    polyGon_regression_batch = np.zeros((batch_size, 8, 5), dtype=np.float32)
    for index, anno in enumerate(annotations):
        labels = anno['labels']
        points = anno['points'] * (input_W, input_H, input_W, input_H, input_W,
                                   input_H, input_W, input_H)
        round_points = points.astype(np.int32)
        diff_points = points - round_points
        reshape_points = round_points.reshape((-1, 2))
        reshape_diff_points = diff_points.reshape((-1, 2))
        for j, (reshape_point, diff_point) in enumerate(zip(reshape_points,
                                                          reshape_diff_points)):
            polyGon_regression_batch[index, j, 0] = diff_point[0]
            polyGon_regression_batch[index, j, 1] = diff_point[1]
            # print("origin: ", box, ", convert: ", w / self.heat_w, " ",
            #       h / self.heat_h, ", points: ", points, ", points_int: ",
            #       points_int, ", heatmap reg: ", reg_diff_x, " ",
            #       reg_diff_y, " ", w, " ", h)
            polyGon_regression_batch[index, j, 2] = index
            polyGon_regression_batch[
                index, j, 3] = reshape_point[0] + reshape_point[1] * input_W
            polyGon_regression_batch[index, j, 4] = 1
        if len(labels) == 1:
            if labels[0] == 0:
                polyGon_labels_batch[index, 0] = 1
                regression_batch[index, 0] = 1
                regression_batch[index, 1:9] = diff_points[0]
            elif labels[0] == 1:
                polyGon_labels_batch[index, 1] = 1
                regression_batch[index, 9] = 1
                regression_batch[index, 10:] = diff_points[0]
        elif len(labels) == 2:
            polyGon_labels_batch[index] = 1
            for i, (_, point) in enumerate(zip(labels, diff_points)):
                regression_batch[index, i * 9] = 1
                regression_batch[index, i * 9 + 1:i * 9 + 9] = point
        else:
            raise ValueError("we do not support this points length")
    return regression_batch, polyGon_labels_batch, polyGon_regression_batch


def get_polyGon_target(annotations, input_shape):
    regression_batch = np.zeros((9 * 2), dtype=np.float32)
    input_H, input_W = input_shape
    polyGon_labels_batch = np.zeros(2, dtype=np.float32)
    # 2 (x, y), 2(indices, (b, ind)), 1 mask (mark valid box)
    polyGon_regression_batch = np.zeros((8, 4), dtype=np.float32)
    # for index, anno in enumerate(annotations):
    labels = annotations['labels']
    points = annotations['points'] * (input_W, input_H, input_W, input_H,
                                      input_W, input_H, input_W, input_H)
    round_points = points.astype(np.int32)
    diff_points = points - round_points
    reshape_points = round_points.reshape((-1, 2))
    reshape_diff_points = diff_points.reshape((-1, 2))
    for j, (reshape_point,
            diff_point) in enumerate(zip(reshape_points, reshape_diff_points)):
        polyGon_regression_batch[j, 0] = diff_point[0]
        polyGon_regression_batch[j, 1] = diff_point[1]
        # print("origin: ", box, ", convert: ", w / self.heat_w, " ",
        #       h / self.heat_h, ", points: ", points, ", points_int: ",
        #       points_int, ", heatmap reg: ", reg_diff_x, " ",
        #       reg_diff_y, " ", w, " ", h)
        # polyGon_regression_batch[index, j, 2] = index
        polyGon_regression_batch[
            j, 2] = reshape_point[0] + reshape_point[1] * input_W
        polyGon_regression_batch[j, 3] = 1
    if len(labels) == 1:
        if labels[0] == 0:
            polyGon_labels_batch[0] = 1
            regression_batch[0] = 1
            regression_batch[1:9] = diff_points[0]
        elif labels[0] == 1:
            polyGon_labels_batch[1] = 1
            regression_batch[9] = 1
            regression_batch[10:] = diff_points[0]
    elif len(labels) == 2:
        polyGon_labels_batch[:] = 1
        for i, (_, point) in enumerate(zip(labels, diff_points)):
            regression_batch[i * 9] = 1
            regression_batch[i * 9 + 1:i * 9 + 9] = point
    else:
        raise ValueError("we do not support this points length")
    return regression_batch, polyGon_labels_batch, polyGon_regression_batch


def get_ocrNum_targets(annotations, ocr_dict_length, batch_size):
    # ocr_dict_length: number of ocr labels
    labels_batch = np.full((batch_size, ocr_dict_length + 1),
                           -1,
                           dtype=np.int32)
    for index, anno in enumerate(annotations):
        labels = anno['labels']
        label_length = len(labels)
        # print("label_length: ", label_length, labels)
        labels_batch[index, -1] = label_length
        labels_batch[index, :label_length] = labels

    # print(labels_batch, annotations)
    return labels_batch


def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def get_detections(output, score_threshold, image_H, image_W):
    outputs = np.reshape(output, (-1, 9))
    for output in outputs:
        if output[0] < score_threshold:
            output[0] = 0
        else:
            points = output[1:]
            points = np.reshape(points, (-1, 2))
            points = points * [image_W, image_H]
            output[1:] = np.reshape(points, (8, ))
            output[0] = 1
    return outputs


# def ctc_greedy_decoder(logits, sequence_length, num_classes=3):
#     '''
#   sequence_length : batch中每个label最大的长度time_step, 当前例子中是每个label长度是7
#   '''
#     logits = tf.transpose(logits, (1, 0, 2))
#     decoded, log_prob = tf.nn.ctc_greedy_decoder(logits,
#                                                  sequence_length,
#                                                  merge_repeated=False)
#     return decoded[0].values


# def ctc_beam_search_decoder(logits, sequence_length):
#     logits = tf.transpose(logits, (1, 0, 2))
#     decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits,
#                                                       sequence_length,
#                                                       merge_repeated=False)
#     return decoded[0].values


# def accurateOfCTC(decoded, labels):
#     acc = tf.reduce_mean(
#         tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))
#     return acc


def compute_distances(points_1, points_2):
    p1_x, p1_y = points_1
    p2_x, p2_y = points_2
    distance = pow(p1_x - p2_x, 2) + pow(p1_y - p2_y, 2)
    return math.sqrt(distance)


# def Get_correct_boxes(boxes, input_shape, image_shape):
#     '''
#     Get corrected boxes
#     boxes: (NUm, 4) --->[(xmin, ymin, xmax, ymax)...]
#     image_shape: [image_W, image_H]
#     '''
#     input_shape = tf.cast(input_shape, tf.dtype(boxes))
#     image_shape = tf.cast(image_shape, tf.dtype(boxes))
#     new_shape = tf.round(image_shape * tf.min(input_shape / image_shape))
#     diff_x, diff_y = (input_shape - new_shape) / 2. / input_shape

#     scale_x, scale_y = input_shape / new_shape
#     boxes = (boxes[:, 4], - [diff_x, diff_y, diff_x, diff_y]) * [
#         scale_x, scale_y, scale_x, scale_y
#     ]

#     boxes *= tf.concat([image_shape, image_shape], axis=-1)
#     return boxes


# def predict_box(image_path, model, anchors, input_size, preprocess_func,
#                 label_to_name):
#     '''
#     Returns:
#       [<class_name> <confidence> <left> <top> <right> <bottom>]
#     '''
#     image = cv2.imread(image_path)
#     src_image = image.copy()
#     h, w = src_image.shape[:2]
#     image = preprocess_func(image, input_size)
#     # run network
#     predict_results, scores, labels = model.predict_on_batch(
#         [np.expand_dims(image, axis=0),
#          np.expand_dims(anchors, axis=0)])

#     class_names = [label_to_name(label) for label in labels]

#     detections = Get_correct_boxes(predict_results, input_size, (w, h))
#     detections = np.concatenate([
#         np.expand_dims(class_names, axis=1),
#         np.expand_dims(scores, axis=1),
#         detections,
#     ],
#                                 axis=1)
#     return detections
