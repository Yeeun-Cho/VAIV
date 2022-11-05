from pathlib import Path
from tqdm import tqdm
import pandas as pd
import torch
import sys
import numpy as np
from multiprocessing import Process
sys.path.append('/home/ubuntu/2022_VAIV_Cho/VAIV/Yolo/Code/yolov7')
from utils.general import xyxy2xywh, xywh2xyxy  # noqa: E402

ROOT = Path('/home/ubuntu/2022_VAIV_Cho/VAIV')
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / 'Common' / 'Code'))

from manager import VAIV  # noqa: E402
from pattern import Bullish, Bearish  # noqa: E402


def xyxy_to_xywh(vaiv: VAIV, xyxy):
    size = vaiv.kwargs.get('size')
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
    return np.round(xywh, 6)


def date_range(date, dates, num):
    now = dates.index(date)
    if now < num-1:
        return dates[0:num]
    elif now > (len(dates) - (num-1)):
        return dates[-num:len(dates)]
    else:
        plus = num // 2
        minus = num - plus
        return dates[now-minus:now+plus]


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


def get_point(vaiv: VAIV, section):
    ticker = vaiv.kwargs.get('ticker')
    pattern = vaiv.kwargs.get('pattern')
    trade_date = vaiv.kwargs.get('trade_date')
    vaiv.set_fname('txt', ticker=ticker, trade_date=trade_date)
    vaiv.set_path(vaiv.yolo.labeling)
    txt_path = vaiv.path
    section = section.astype(int)
    max_date = section.Close.idxmax()
    min_date = section.Close.idxmin()
    return min_date, max_date


def make_span_labeling(vaiv: VAIV, span, priority):
    candle = vaiv.kwargs.get('candle')
    date = vaiv.kwargs.get('trade_date')
    stock = vaiv.modedf.get('stock')
    predict = vaiv.modedf.get('predict')

    start = predict.loc[date, 'Start']
    end = predict.loc[date, 'End']
    trade = stock.loc[start:end]
    dates = trade.index.tolist()
    
    label_list = []
    for i in range(0, candle, span):
        section = trade.iloc[i:i+span]
        buy_date, sell_date = get_point(vaiv, section)
        buy_drange = date_range(buy_date, dates, 5)
        buy_xyxy = get_xyxy(vaiv, buy_drange)
        buy_xywh = xyxy_to_xywh(vaiv, buy_xyxy)
        sell_drange = date_range(sell_date, dates, 5)
        sell_xyxy = get_xyxy(vaiv, sell_drange)
        sell_xywh = xyxy_to_xywh(vaiv, sell_xyxy)
        
        buy_pattern = []
        for date in buy_drange:
            result = Bullish(date, buy_drange, trade)
            pattern = [p for p, c in result.items() if c == True]
            buy_pattern += pattern
        
        sell_pattern = []
        for date in sell_drange:
            result = Bearish(date, sell_drange, trade)
            pattern = [p for p, c in result.items() if c == True]
            sell_pattern += pattern
        
        buy_pattern = ','.join(list(set(buy_pattern)))
        sell_pattern = ','.join(list(set(sell_pattern)))
        # buy: 1, sell: 0
        df = pd.DataFrame({
            'Label': [1, 0],
            'CenterX': [buy_xywh[0], sell_xywh[0]],
            'CenterY': [buy_xywh[1], sell_xywh[1]],
            'Width': [buy_xywh[2], sell_xywh[2]],
            'Height': [buy_xywh[3], sell_xywh[3]],
            'Priority': [priority, priority],
            'Pattern': [buy_pattern, sell_pattern],
        })
        
        # df = pd.DataFrame({
        #     'Label': [1, 0],
        #     'CenterX': [buy_xyxy[0], sell_xyxy[0]],
        #     'CenterY': [buy_xyxy[1], sell_xyxy[1]],
        #     'Width': [buy_xyxy[2], sell_xyxy[2]],
        #     'Height': [buy_xyxy[3], sell_xyxy[3]],
        #     'Priority': [priority, priority],
        #     'Pattern': [buy_pattern, sell_pattern],
        # })
        label_list.append(df)
    return label_list


def make_ticker_date_labelings(vaiv: VAIV):
    ticker = vaiv.kwargs.get('ticker')
    trade_date = vaiv.kwargs.get('trade_date')
    candle = vaiv.kwargs.get('candle')
    span = candle
    priority = 1
    
    label_list = []
    while span > 5:
        span_label_list = make_span_labeling(vaiv, span, priority)
        label_list += span_label_list
        priority += 1
        span //= 2
    labeling = pd.concat(label_list).drop_duplicates(subset=['CenterX']).reset_index(drop=True)
    vaiv.set_fname('csv', ticker=ticker, trade_date=trade_date)
    vaiv.set_path(vaiv.yolo.labeling)
    labeling.to_csv(vaiv.path, index=False)
    

def make_ticker_labelings(vaiv: VAIV, start_date, end_date):
    predict = vaiv.modedf.get('predict')
    predict = predict.loc[start_date:end_date]
    for trade_date in predict.index.tolist():
        vaiv.set_kwargs(trade_date=trade_date)
        vaiv.load_df('pixel')
        make_ticker_date_labelings(vaiv)
    return


def make_all_labelings(vaiv: VAIV, start_date='2006', end_date='z', num=968):
    market = vaiv.kwargs.get('market')
    vaiv.load_df(market)
    df = vaiv.modedf.get(market).reset_index()

    pbar = tqdm(total=num)
    for ticker in df.Ticker.tolist()[:num]:
        vaiv.set_kwargs(ticker=ticker)
        vaiv.load_df('stock')
        vaiv.load_df('predict')
        make_ticker_labelings(vaiv, start_date, end_date)
        pbar.update()
    pbar.close()


if __name__ == '__main__':
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
        'pattern': False,
        'name': 'Kospi200',
    }
    vaiv.set_kwargs(**kwargs)
    vaiv.set_stock()
    vaiv.set_prediction()
    vaiv.set_image()
    vaiv.set_labeling()
    vaiv.make_dir(yolo=True, labeling=True)
    years = ['2019-', '2020-', '2021-']
    start_mds = ['01-01', '04-01', '07-01', '10-01']
    end_mds = ['03-31', '06-30', '09-30', '12-31']
    num = 200
    for year in years:
        for start_md, end_md in zip(start_mds, end_mds):
            start_date = year + start_md
            end_date = year + end_md
            p = Process(target=make_all_labelings, args=(vaiv, start_date, end_date, num, ))
            p.start()
    # make_all_labelings(vaiv, start_date=start_date, end_date=end_date, num=50)
