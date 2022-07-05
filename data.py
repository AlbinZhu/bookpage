'''
Author: bin.zhu
Date: 2022-06-29 17:54:44
LastEditors: bin.zhu
LastEditTime: 2022-07-04 13:43:20
Description: file content
'''

from ctypes import util
import cv2
import torchdata.datapipes as dp
import image_utils
from utils.util import load_polyGonAnnotations, get_polyGon_target
from data_generator.heatmap import PolyGonHeatmapGenerator

page_anno_dir = '/home/albin/Documents/data/annoDir/page_label/'


def DataPipe(anno_file: str):
    common_aug_method = image_utils.VisualEffect(
        image_utils.configUtils(True, True, True, True, True, True, True))
    preprocess = image_utils.polyGonProcessImage()
    heatmap_option = {
        "heat_H": 256,
        "heat_W": 256,
        "sigma": 5,
    }
    heatmap = PolyGonHeatmapGenerator(8, **heatmap_option)
    anno_files = []
    with open(anno_file, 'r') as f:
        anno_files = f.read().splitlines()

    assert len(anno_file) > 0, 'anno_file must exists.'
    anno_iter = dp.iter.IterableWrapper(anno_files)

    # print(list(anno_iter))

    # def getAnno(imageFile: str):
    #     filename = page_anno_dir + imageFile.split('/')[-1].replace(
    #         '.jpg', '.txt')
    #     with open(filename, 'r') as f:
    #         anno = f.read().splitlines()
    #     assert anno is not None
    #     return anno

    # def getImage(imageFile: str):

    #     img = cv2.imread(imageFile)
    #     img = common_aug_method(img)

    def dataProcess(imageFile: str):
        img = cv2.imread(imageFile)
        img = common_aug_method(img)

        anno_file = page_anno_dir + imageFile.split('/')[-1].replace(
            '.jpg', '.txt')
        anno = load_polyGonAnnotations(anno_file)
        processed_image, points = preprocess(
            img, 1024, image_utils.ProcessType.ENCODE_ANNO, anno["points"])

        anno['points'] = points
        batch_annos, batch_labels, batch_reg = get_polyGon_target(
            anno, (heatmap.heat_h, heatmap.heat_w))
        batch_heatmap = heatmap.generate_heatmap_anno(anno)

        return processed_image, (batch_labels, batch_reg, batch_heatmap)

    datapipe = anno_iter.map(lambda row: dataProcess(row))
    print(list(datapipe)[0])


if __name__ == "__main__":
    train_file = "/home/albin/Documents/data/annoDir/ImageSet/page_trainval.txt"
    val_file = "/home/albin/Documents/data/annoDir/ImageSet/page_val.txt"

    DataPipe(train_file)