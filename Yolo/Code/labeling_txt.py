from pathlib import Path
from tqdm import tqdm
import pandas as pd
import torch
import sys
import numpy as np
import shutil
from multiprocessing import Process
from itertools import product


sys.path.append('/home/ubuntu/2022_VAIV_Cho/VAIV/Yolo/Code/yolov7')
from utils.general import xyxy2xywh  # noqa: E402

ROOT = Path('/home/ubuntu/2022_VAIV_Cho/VAIV')
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / 'Common' / 'Code'))

from manager import VAIV  # noqa: E402


def make_labelings(df, mode, year, prior_thres, pattern_thres, name):
    txt_path = Path(f'/home/ubuntu/2022_VAIV_Cho/VAIV/Yolo/Dataset/Kospi200/prior{prior_thres}_pattern{pattern_thres}/{mode.capitalize()}{year}/labels')
    txt_path.mkdir(parents=True, exist_ok=True)
    for row in df.to_dict('records'):
        label = row.get('Label')
        x = row.get('CenterX')
        y = row.get('CenterY')
        w = row.get('Width')
        h = row.get('Height')
        # prob = row.get('Probability')
        priority = row.get('Priority')
        pattern = row.get('Pattern')
        pattern = list(filter(None, pattern.split(',')))
        xywh = [x, y, w, h]
        line = (label, *xywh)
        if (priority > prior_thres):
            continue
        if (priority > pattern_thres) & (len(pattern) == 0):
            # print(label, xmin, ymin, xmax, ymax, priority, pattern)
            continue
        with open(txt_path / f'{name}.txt', 'a') as f:
            f.write(('%s ' * len(line)).rstrip() % line + '\n')
        
def make_all_labelings(mode, year, prior_thres, pattern_thres):
    p = Path('/home/ubuntu/2022_VAIV_Cho/VAIV/Yolo/Labeling/Kospi/Kospi200')
    files = list(p.iterdir())
    files = [f for f in files if f'{year}-' in f.stem]
    
    pbar = tqdm(total=len(files))
    for file in files:
        df = pd.read_csv(file)
        df = df.replace(np.nan, '', regex=True)
        make_labelings(df, mode, year, prior_thres, pattern_thres, file.stem)
        pbar.update()     
    pbar.close()

def move_images(mode, year, prior_thres, pattern_thres):
    dataset_path = Path(f'/home/ubuntu/2022_VAIV_Cho/VAIV/Yolo/Dataset/Kospi200/prior{prior_thres}_pattern{pattern_thres}/{mode.capitalize()}{year}')
    txt_path = dataset_path / 'labels'
    imgFrom_path = Path('/home/ubuntu/2022_VAIV_Cho/VAIV/Common/Image/Candle/default/1/1800x650_245_1_0.8/Kospi/images')
    imgTo_path = dataset_path / 'images'
    txt_path.mkdir(parents=True, exist_ok=True)
    imgTo_path.mkdir(parents=True, exist_ok=True)
    files = list(txt_path.iterdir())
    pbar = tqdm(total=len(files))
    for txt in files:
        name = txt.stem
        from_img = imgFrom_path / f'{name}.png'
        to_img = imgTo_path / f'{name}.png'
        shutil.copy(from_img, to_img)
        pbar.update()     
    pbar.close()

if __name__ == '__main__':
    dataset = {'train': 2019, 'valid': 2020, 'test': 2021}
    procs = []
    
    for mode, year in dataset.items():
        for prior_thres in range(3, 5):
            for pattern_thres in range(prior_thres-1, prior_thres+2):
                p = Process(target=make_all_labelings, args=(mode, year, prior_thres, pattern_thres, ))
                p.start()
                procs.append(p)
    
    for p in procs:
        p.join()
            
    for mode, year in dataset.items():
        for prior_thres in range(3, 5):
            for pattern_thres in range(prior_thres-1, prior_thres+2):      
                p = Process(target=move_images, args=(mode, year, prior_thres, pattern_thres, ))
                p.start()
        # make_all_labelings(mode, year, prior_thres)
        # move_images(mode, year, prior_thres)
