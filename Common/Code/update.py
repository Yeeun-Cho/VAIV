from pathlib import Path
import sys
import argparse
import exchange_calendars as ecals
from datetime import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

ROOT = Path('/home/ubuntu/2022_VAIV_Cho/VAIV')
sys.path.append(ROOT)
sys.path.append(ROOT / 'Common' / 'Code')

from manager import VAIV  # noqa: E402
from stock import update_market, update_all_stocks  # noqa: E402
from prediction import update_all_predictions  # noqa: E402
from candlestick import update_all_candlesticks  # noqa: E402


def DateCheck(today):
    XKRX = ecals.get_calendar('XKRX')
    return XKRX.is_session(today)


def next_Date(today):
    XKRX = ecals.get_calendar('XKRX')
    return XKRX.next_session(today).strftime('%Y-%m-%d')


def update(vaiv: VAIV, stock=True, prediction=True, image=True):
    today = datetime.now()
    today = today.strftime('%Y-%m-%d')
    if DateCheck(today):
        trade_date = next_Date(today)
        if stock:
            update_market(vaiv)
            update_all_stocks(vaiv, today)
        if prediction:
            update_all_predictions(vaiv, today, trade_date)
        if image:
            update_all_candlesticks(vaiv, trade_date)


def update_cnn(stock=False, prediction=False, image=False):
    vaiv = VAIV(ROOT)
    kwargs = {
        'market': 'Kospi',
        'feature': {'Volume': False, 'MA': [-1], 'MACD': False},
        'offset': 1,
        'size': [224, 224],
        'candle': 20,
        'linespace': 1,
        'candlewidth': 0.8,
        'style': 'dark_background'  # default는 'classic'
    }
    vaiv.set_kwargs(**kwargs)
    vaiv.set_stock()
    vaiv.set_prediction()
    vaiv.set_image()
    vaiv.make_dir(stock=True, prediction=True, image=True)
    update(vaiv, stock=stock, prediction=prediction, image=image)


def update_yolo(stock=False, prediction=False, image=False):
    vaiv = VAIV(ROOT)
    kwargs = {
        'market': 'Kospi',
        'feature': {'Volume': False, 'MA': [-1], 'MACD': False},
        'offset': 1,
        'size': [1800, 650],
        'candle': 245,
        'linespace': 1,
        'candlewidth': 0.8,
        'style': 'default'  # 밝은 배경은 'default'
    }
    vaiv.set_kwargs(**kwargs)
    vaiv.set_stock()
    vaiv.set_prediction()
    vaiv.set_image()
    vaiv.make_dir(stock=True, prediction=True, image=True)
    update(vaiv, stock=stock, prediction=prediction, image=image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--yolo', action='store_true', help='update yolo files'
    )
    parser.add_argument(
        '--cnn', action='store_true', help='update cnn files'
    )
    parser.add_argument(
        '--stock', action='store_true', help='update stock data'
    )
    parser.add_argument(
        '--prediction', action='store_true', help='update prediction'
    )
    parser.add_argument(
        '--image', action='store_true', help='update images'
    )

    opt = parser.parse_args()
    print(opt)

    if opt.yolo:
        update_yolo(stock=opt.stock, prediction=opt.prediction, image=opt.image)
        opt.stock = False  # stock 중복 업데이트 방지
    if opt.cnn:
        update_cnn(stock=opt.stock, prediction=opt.prediction,  image=opt.image)
