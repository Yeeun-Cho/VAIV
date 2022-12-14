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


def xyxy_to_xywh(vaiv: VAIV, xyxy):
    size = vaiv.kwargs.get('size')
    shape = (size[1], size[0], 3)
    gn = torch.tensor(shape)[[1, 0, 1, 0]]
    xywh = (
        xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn
    ).view(-1).tolist()  # normalized xywh
    return np.round(xywh, 6)


def get_xyxy(vaiv: VAIV, drange):
    pixel = vaiv.modedf.get('pixel')
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    for date in drange:
        xmin, ymin, xmax, ymax = pixel.loc[date, 'Xmin':'Ymax'].tolist()
        xmins.append(xmin)
        ymins.append(ymin)
        xmaxs.append(xmax)
        ymaxs.append(ymax)
    return [min(xmins), min(ymins), max(xmaxs), max(ymaxs)]


def make_date_labelings(vaiv:VAIV, df, mode, name):
    txt_path = vaiv.yolo.dataset.get(mode).get('labels')
    df_path = vaiv.yolo.dataset.get(mode).get('dataframes')
    # txt_path.mkdir(parents=True, exist_ok=True)
    vaiv.set_kwargs(ticker=name.split('_')[0], trade_date=name.split('_')[1])
    vaiv.load_df('pixel')

    df_list = []
    for row in df.to_dict('records'):
        drange = row.get('Range').split('/')
        label = row.get('Label')
        # if label < 5:  # buy pattern 모델
        # if label >= 5:  # sell pattern 모델
        #     label -= 5
        #     row['Label'] -= 5
        if True:  # all pattern 모델
            xyxy = get_xyxy(vaiv, drange)
            xywh = xyxy_to_xywh(vaiv, xyxy)
            line = (label, *xywh)
            with open(txt_path / f'{name}.txt', 'a') as f:
                f.write(('%s ' * len(line)).rstrip() % line + '\n')
            row = {k: [v] for k, v in row.items()}
            df_list.append(pd.DataFrame(row))
    dataframes = pd.concat(df_list)
    dataframes.to_csv(df_path / f'{name}.csv', index=False)
    
        
def make_labelings(df, mode, prior_thres, pattern_thres, name):
    txt_path = vaiv.yolo.dataset.get(mode).get('labels')
    df_path = vaiv.yolo.dataset.get(mode).get('dataframes')
    
    df_list = []
    for row in df.to_dict('records'):
        label = row.get('Label')
        x = row.get('CenterX')
        y = row.get('CenterY')
        w = row.get('Width')
        h = row.get('Height')
        # prob = row.get('Probability')
        priority = row.get('Priority')
        pattern = row.get('Pattern')
        pattern = list(filter(None, pattern.split('/')))
        xywh = [x, y, w, h]
        line = (label, *xywh)
        if (priority > prior_thres):
            continue
        if (priority > pattern_thres) & (len(pattern) == 0):
            # print(label, xmin, ymin, xmax, ymax, priority, pattern)
            continue
        with open(txt_path / f'{name}.txt', 'a') as f:
            f.write(('%s ' * len(line)).rstrip() % line + '\n')
        row = {k: [v] for k, v in row.items()}
        df_list.append(pd.DataFrame(row))
    dataframes = pd.concat(df_list)
    dataframes.to_csv(df_path / f'{name}.csv', index=False)
    
        
def make_all_labelings(vaiv:VAIV, mode, years, prior_thres, pattern_thres, labeling):
    # p = Path('/home/ubuntu/2022_VAIV_Cho/VAIV/Yolo/Labeling/Kospi/Kospi50/Merge')
    p = vaiv.yolo.labeling.get(labeling)
    files = list(p.iterdir())
    print(files[0].stem.split('_')[1].split('-')[0])
    files = [f for f in files if int(f.stem.split('_')[1].split('-')[0]) in range(years[0], years[1])]
    
    pbar = tqdm(total=len(files))
    for file in files:
        df = pd.read_csv(file)
        df = df.replace(np.nan, '', regex=True)
        if labeling == 'merge':  # min_max labeling / merge labeling 모델
            make_labelings(df, mode, prior_thres, pattern_thres, file.stem)
        else:  # pattern labeling 모델
            make_date_labelings(vaiv, df, mode, file.stem)
        pbar.update()     
    pbar.close()


def move_images(vaiv:VAIV, mode):
    # dataset_path = vaiv.yolo.dataset.get(mode)
    # dataset_path = Path(f'/home/ubuntu/2022_VAIV_Cho/VAIV/Yolo/Dataset/Kospi50/prior{prior_thres}_pattern{pattern_thres}/{mode.capitalize()}{year}')
    txt_path = vaiv.yolo.dataset.get(mode).get('labels')
    imgFrom_path = vaiv.common.image.get('images')
    # imgFrom_path = Path('/home/ubuntu/2022_VAIV_Cho/VAIV/Common/Image/Candle/default/1/1800x650_245_1_0.8/Kospi/images')
    imgTo_path = vaiv.yolo.dataset.get(mode).get('images')
    # imgTo_path = dataset_path / 'images'
    # txt_path.mkdir(parents=True, exist_ok=True)
    # imgTo_path.mkdir(parents=True, exist_ok=True)
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
    name = 'Kospi50_2006-2022'
    vaiv = VAIV(ROOT)
    kwargs = {
        'market': 'Kospi',
        'feature': {'Volume': False, 'MA': [-1], 'MACD': False},
        'offset': 1,
        'size': [1800, 650],
        'candle': 245,
        'linespace': 1,
        'candlewidth': 0.8,
        'style': 'default',
        'folder': 'yolo',
        'name': name,
    }
    vaiv.set_kwargs(**kwargs)
    vaiv.set_stock()
    vaiv.set_prediction()
    vaiv.set_image()
    vaiv.set_labeling()
    vaiv.set_kwargs(name='Kospi50_2006-2022')
    vaiv.set_dataset()
    vaiv.make_dir(yolo=True, dataset=True, labeling=True)
    
    dataset = {
        'train': (2006,2018), 
        'valid': (2018,2019), 
        'test': (2019,2022)
    }
    procs = []
    
    for mode, year in dataset.items():
        for prior_thres in range(5, 6):
            for pattern_thres in range(3, 4):
                p = Process(target=make_all_labelings, args=(vaiv, mode, year, prior_thres, pattern_thres, 'merge', ))  # pattern labeling 쓰고 싶으면 'merge'를 'pattern'으로 바꾼다
                p.start()
                procs.append(p)
    
    for p in procs:
        p.join()
            
    for mode, year in dataset.items():
        for prior_thres in range(5, 6):
            for pattern_thres in range(3, 4): 
                p = Process(target=move_images, args=(vaiv, mode, ))
                p.start()
        # make_all_labelings(mode, year, prior_thres)
        # move_images(mode, year, prior_thres)
