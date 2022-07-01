from datetime import date
from anchors.anchor_box import CenterAnchor
from utils import const, label
import os


#------------------------Model Config---------------------------------#
class ModelConfig(object):

    def __init__(self, image_size, classes):
        self.image_size = image_size
        self.classes = classes


class OcrModelConfig(ModelConfig):

    def __init__(self, num_classes, score_threshold, **kwargs):
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        super(OcrModelConfig, self).__init__(**kwargs)


def GetOcrnumModelParams():
    config_options = {
        "image_size": 60,
        'classes': label.ocr_number_classes,
    }
    return OcrModelConfig(len(label.ocr_number_classes.label_classes), 0.5,
                          **config_options)


class PolyGonModelConfig(ModelConfig):

    def __init__(self, heatmap_channels, score_threshold, **kwargs):
        self.heatmap_channels = heatmap_channels
        self.score_threshold = score_threshold
        super(PolyGonModelConfig, self).__init__(**kwargs)


def GetPolyGonModelParams():
    config_options = {"image_size": 1024, 'classes': label.polyGon_classes}
    return PolyGonModelConfig(8, 0.5, **config_options)


class DetModelConfig(ModelConfig):

    def __init__(self, image_size, num_classes, score_threshold, nms,
                 nms_threshold, max_detes, iou_threshlod, neg_threshold,
                 anchors_per_feat, features, anchors_mask, strides, **kwargs):
        self.image_size = image_size
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.nms = nms
        self.nms_threshold = nms_threshold
        self.max_detes = max_detes
        self.iou_threshold = iou_threshlod
        self.neg_threshold = neg_threshold
        self.anchors_per_feat = anchors_per_feat
        self.feats = features
        self.anchor_maskes = anchors_mask
        self.anchor_strides = strides

        super(DetModelConfig, self).__init__(**kwargs)


class EfficientDetModelConfig(ModelConfig):

    def __init__(self,
                 num_classes,
                 score_threshold,
                 nms_threshold,
                 max_detes,
                 down_scale,
                 top_k_per_class,
                 freeze_backbone=True,
                 weighted_bifpn=False,
                 **kwargs):
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detes = max_detes
        self.down_scale = down_scale
        self.top_k = top_k_per_class
        self.freeze_backbone = freeze_backbone
        self.weighted_bifpn = weighted_bifpn
        super(EfficientDetModelConfig, self).__init__(**kwargs)


def GetDetModelParams():
    config_options = {"image_size": 1024, 'classes': label.det_classes}
    score_threshlod = 0.5
    top_k_per_class = 20
    nms_threshold = 0.5
    max_dets = const.max_dets
    down_scale = 4

    return EfficientDetModelConfig(len(label.det_classes.label_classes),
                                   score_threshlod,
                                   nms_threshold,
                                   max_dets,
                                   down_scale,
                                   top_k_per_class,
                                   freeze_backbone=True,
                                   weighted_bifpn=False,
                                   **config_options)


#------------------------Train Config---------------------------------#


class TrainConfig(object):

    def __init__(self,
                 batch_size,
                 epochs,
                 gpu,
                 snapshot,
                 snapshot_path,
                 tensorboard_dir,
                 random_transform,
                 compute_val_loss,
                 learning_rate,
                 img_folder,
                 anno_folder,
                 set_folder,
                 weight_file,
                 multiprocessing,
                 workers,
                 max_queue_size,
                 set_name='polygon'):
        self.batch_size = batch_size
        self.epochs = epochs
        self.gpu = gpu
        self.snapshot_path = snapshot_path
        self.snapshot = snapshot
        self.tensorboard_dir = tensorboard_dir
        self.random_transform = random_transform
        self.compute_val_loss = compute_val_loss
        self.multiprocessing = multiprocessing
        self.workers = workers
        self.learning_rate = learning_rate
        self.multiprocessing = multiprocessing
        self.max_queue_size = max_queue_size
        self.img_folder = img_folder
        self.anno_folder = anno_folder
        self.weight_file = weight_file
        self.set_folder = set_folder
        self.train_file = "{}_trainval.txt".format(set_name)
        self.val_file = "{}_val.txt".format(set_name)
        if self.gpu and self.batch_size < len(self.gpu.split(',')):
            raise ValueError("Batch size ({}) must be equal to or higher "
                             "than the number of GPUs ({})".format(
                                 self.batch_size, len(self.gpu.split(','))))
 

def GetPolyGonTrainConfig():
    snapshot_path = "./checkpoints/{}_{}".format("polyGon", date.today())
    tensorboard_dir = "./logs/{}_{}".format("polyGon", date.today())
    return TrainConfig(batch_size=1,
                       epochs=80,
                       gpu="0",
                       snapshot=const.polygon_weights_file,
                       snapshot_path=snapshot_path,
                       tensorboard_dir=tensorboard_dir,
                       random_transform=True,
                       compute_val_loss=True,
                       multiprocessing=True,
                       learning_rate=0.0025,
                       img_folder=const.img_folder,
                       anno_folder=const.polygon_anno_folder,
                       set_folder=const.set_folder,
                       weight_file=const.polygon_weights_file,
                       workers=1,
                       max_queue_size=64,
                       set_name="page")


def GetOcrNumTrainConfig():
    snapshot_path = "checkpoints/{}_{}".format("ocr_num", date.today())
    tensorboard_dir = "logs/{}_{}".format("ocr_num", date.today())
    return TrainConfig(batch_size=512,
                       epochs=300,
                       gpu="3",
                       snapshot=const.ocr_weights_file,
                       snapshot_path=snapshot_path,
                       tensorboard_dir=tensorboard_dir,
                       random_transform=True,
                       compute_val_loss=True,
                       multiprocessing=True,
                       learning_rate=0.025,
                       img_folder=const.ocr_img_folder,
                       anno_folder=const.ocr_anno_folder,
                       set_folder=const.set_folder,
                       weight_file=const.ocr_weights_file,
                       workers=16,
                       max_queue_size=16,
                       set_name="ocr")


class DetTrainConfig(TrainConfig):

    def __init__(self, use_mosaic, use_random_crop, use_commom_aug, **options):
        self.use_mosaic = use_mosaic
        self.use_random_crop = use_random_crop
        self.use_common_aug = use_commom_aug
        super(DetTrainConfig, self).__init__(**options)


def GetDetTrainConfig():
    train_options = {
        "snapshot_path":
        "checkpoints/{}_{}".format("det", date.today()),
        "tensorboard_dir":
        "logs/{}_{}".format("det", date.today()),
        "snapshot":
        const.det_weights_file \
        if os.path.exists(const.det_weights_file) else "imagenet",
        "batch_size": 16,
        "epochs":50,
        "gpu": "0",
        "random_transform":True,
        "compute_val_loss": True,
        "multiprocessing": True,
        "learning_rate": 0.0125,
        "img_folder": const.det_img_folder,
        "anno_folder":const.det_anno_folder,
        "set_folder":const.set_folder,
        "weight_file":const.det_weights_file,
        "workers":16,
        "max_queue_size":64,
        "set_name":"det"
    }
    return DetTrainConfig(use_mosaic=True,
                          use_random_crop=True,
                          use_commom_aug=True,
                          **train_options)
