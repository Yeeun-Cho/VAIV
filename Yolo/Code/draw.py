import argparse
import time
from pathlib import Path

import cv2
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, xywh2xyxy, xywhn2xyxy, \
    set_logging, increment_path
from utils.plots import plot_one_box
from utils.pixel import StockImage
from utils.torch_utils import select_device, load_classifier, \
    TracedModel

colors = [(119,216,121), (219,63,63)]

def coco_to_yolo(x1, y1, w, h):
    image_w = 1800
    image_h = 650
    return [((2*x1 + w)/(2*image_w)) , ((2*y1 + h)/(2*image_h)), w/image_w, h/image_h]

def xyxy_to_xywh(xyxy):
    size = [1800, 650]
    shape = (size[1], size[0], 3)
    gn = torch.tensor(shape)[[1, 0, 1, 0]]
    xywh = (
        xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn
    ).view(-1).tolist()  # normalized xywh
    # xmin, ymin, xmax, ymax = xyxy
    # width, height = size
    # x = xmin / width
    # y = ymin / height
    # w = (xmax - xmin) / width
    # h = (ymax - ymin) / height
    # xywh = [x, y, w, h]
    return xywh

def draw(label, im0, xyxy):
    plot_one_box(
        xyxy, im0, label=label,
        color=colors[int(label)],
        line_thickness=2
    )
    return im0
    
if __name__ == '__main__':
    img_path = '/home/ubuntu/2022_VAIV_Cho/VAIV/Yolo/Dataset/Kospi50_4/Train2019/images/000020_2019-01-02.png'
    label_path = Path('/home/ubuntu/2022_VAIV_Cho/VAIV/Yolo/Dataset/Kospi50_4/Train2019/labels/000020_2019-01-02.txt')
    
    im0 = cv2.imread(img_path)
    f = open(label_path, 'r')
    # while True:
    #     x = []
    #     line = f.readline().rstrip()
    #     if line == "":
    #         break
    #     label, *xywh = line.split(' ')
    #     xywh = [float(pixel) for pixel in xywh]
    #     # xyxy = xywh2xyxy(torch.tensor(xywh).view(1, 4)).tolist()[0]
    #     x.append(np.array(xywh))
    #     xyxy = xywhn2xyxy(np.array(x), 1800, 650, 12, 40)
    #     # print(xyxy[0])
    #     Axyxy = [1412.0320855614975,491.25577812018486,1437.4331550802142,620.4545454545455]
    #     # xyxy2xywh(np.array(Axyxy, ndmin=2))
    #     im0 = draw(label, im0, Axyxy)
    
    df = pd.read_csv('/home/ubuntu/2022_VAIV_Cho/VAIV/Yolo/Labeling/Kospi/DataFrame_1.0.5/000020_2019-01-02.csv')
    for one in df.to_dict('records'):
        xyxy = [one['CenterX'], one['CenterY'], one['Width'], one['Height']]
        label = one['Label']
        im0 = draw(str(label), im0, xyxy)
        
    cv2.imwrite('test.png', im0)
    f.close()
