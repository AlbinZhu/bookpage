'''
Author: bin.zhu
Date: 2022-06-29 17:07:22
LastEditors: bin.zhu
LastEditTime: 2022-06-29 17:40:01
Description: file content
'''

from torchdata.datapipes.map import MapDataPipe

import csv
import random

# def generate_csv(file_label,
#                  num_rows: int = 5000,
#                  num_features: int = 20) -> None:
#     fieldnames = ['label'] + [f'c{i}' for i in range(num_features)]
#     writer = csv.DictWriter(open(f"sample_data{file_label}.csv", "w"),
#                             fieldnames=fieldnames)
#     writer.writerow({col: col for col in fieldnames})  # writing the header row
#     for i in range(num_rows):
#         row_data = {col: random.random() for col in fieldnames}
#         row_data['label'] = random.randint(0, 9)
#         writer.writerow(row_data)

# num_files_to_generate = 3
# for i in range(num_files_to_generate):
#     generate_csv(file_label=i)

import numpy as np
import torchdata.datapipes as dp


def build_datapipes(root_dir="."):
    # 获取所有文件
    datapipe = dp.iter.FileLister(root_dir)
    # 筛选csv文件
    datapipe = datapipe.filter(filter_fn=lambda filename: "sample_data" in
                               filename and filename.endswith(".csv"))
    # 打开文件，FileOpener没有对应的函数格式，如果安装了iopath，可以使用
    # datapipe = datapipe.open_by_iopath(mode='rt')
    datapipe = dp.iter.FileOpener(datapipe, mode='rt')
    # 解析csv
    datapipe = datapipe.parse_csv(delimiter=",", skip_lines=1)
    # 分离label和特征
    datapipe = datapipe.map(
        lambda row: {
            "label": np.array(row[0], np.int32),
            "data": np.array(row[1:], dtype=np.float64)
        })
    return datapipe


datapipe = build_datapipes()
from torch.utils.data import DataLoader

dl = DataLoader(dataset=datapipe, batch_size=50, shuffle=True)
first = next(iter(dl))
labels, features = first['label'], first['data']
print(f"Labels batch shape: {labels.size()}")
print(f"Feature batch shape: {features.size()}")