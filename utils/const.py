img_folder = '/home/albin/Documents/data/new_images/'
set_folder = '/home/albin/Documents/data/annoDir/ImageSet/'
ocr_img_folder = "../ocr_dataSet/"
det_img_folder = "/home/baseuser/workspace/det_dataset"
polygon_anno_folder = '/home/albin/Documents/data/annoDir/page_label/'
polygon_weights_file = './weights/{}_weights.h5'.format("polyGon")
ocr_anno_folder = '../annoDir/ocr_label/'
ocr_weights_file = './weights/{}_weights.h5'.format("ocr_num")
det_anno_folder = '../annoDir/det_label/'
det_weights_file = './weights/{}_weights.h5'.format("det")
Debug_root = "/Samba/share/"

max_dets = [30, 20, 1, 1, 1, 1, 1, 1, 1]

hosts = ""

MOMENTUM = 0.997
EPSILON = 1e-4